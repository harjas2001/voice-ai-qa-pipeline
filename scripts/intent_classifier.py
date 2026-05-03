"""
Automated Intent Classifier — Dual Output

Trains an ensemble intent classifier on labeled utterance data, then applies
it to a formatted voicebot CSV to produce two outputs:

  Output 1 (Full):     All records with automated intent marking
  Output 2 (Filtered): Stratified sample for human QA review

The script is used as a base template. When invoked via app.py, the
HIGH_IMPACT_INTENTS, BRAND_LABEL, and training_file values are injected
automatically from config/brands.json before execution.

Usage (standalone):
    python intent_classifier.py <input_csv> <full_output> <filtered_output> <training_csv>
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Brand config — injected by app.py at runtime ──────────────────────────────
# These are populated from config/brands.json for each brand's pipeline run.

HIGH_IMPACT_INTENTS = [
    'intentSalesNew',
    'intentBillingEnquiry',
    'intentCancellation',
    'intentSupportHome',
    'intentSupportMobile',
]

BRAND_LABEL   = 'brand'
CHANNEL_LABEL = 'voice'


# ── Classifier ─────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Ensemble intent classifier with outlier detection.

    Architecture:
      - TF-IDF vectoriser (1–3 grams, 5000 features)
      - Voting ensemble: Logistic Regression + SVM + Random Forest
      - IsolationForest outlier detector for low-confidence suppression
    """

    def __init__(self, contamination=0.01, threshold=0.35, use_outlier_detection=True):
        self.contamination          = contamination
        self.threshold              = threshold
        self.use_outlier_detection  = use_outlier_detection
        self.vectorizer             = None
        self.label_encoder          = None
        self.voting_clf             = None
        self.outlier_detector       = None

    def load_training_data(self, training_file):
        """Load and validate training CSV. Accepts intent/ka, intent, or ka column."""
        print(f"\n📂 Loading training data: {training_file}")

        if not Path(training_file).exists():
            raise FileNotFoundError(f"Training file not found: {training_file}")

        df = pd.read_csv(training_file, encoding='utf-8')

        if 'intent/ka' in df.columns:
            intent_col = 'intent/ka'
        elif 'intent' in df.columns:
            intent_col = 'intent'
        elif 'ka' in df.columns:
            intent_col = 'ka'
        else:
            raise ValueError("Training data must have 'intent/ka', 'intent', or 'ka' column")

        if 'utterance' not in df.columns:
            raise ValueError("Training data must have 'utterance' column")

        df = df.rename(columns={intent_col: 'intent'})[['utterance', 'intent']].copy()
        df['utterance'] = df['utterance'].astype(str).str.lower().str.strip()
        df['intent']    = df['intent'].astype(str).str.strip()
        df = df[(df['utterance'] != '') & (df['intent'] != '')]

        print(f"   ✅ {len(df):,} training samples across {df['intent'].nunique()} intents")

        print(f"\n📊 Intent distribution (top 10):")
        for intent, count in df['intent'].value_counts().head(10).items():
            print(f"   {intent}: {count:,}")

        return df

    def train(self, training_data):
        """Train the TF-IDF + ensemble model with class-weighted balancing."""
        print(f"\n🔧 Training classifier...")

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(training_data['intent'])

        print("   Building TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(training_data['utterance'])
        print(f"   ✅ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = dict(zip(np.unique(y), class_weights))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")

        log_reg = LogisticRegression(class_weight=class_weights_dict, max_iter=1000, random_state=42)
        svm     = SVC(probability=True, class_weight=class_weights_dict, random_state=42)
        rf      = RandomForestClassifier(n_estimators=100, class_weight=class_weights_dict,
                                         random_state=42, n_jobs=-1)

        self.voting_clf = VotingClassifier(
            estimators=[('lr', log_reg), ('svc', svm), ('rf', rf)],
            voting='soft', n_jobs=-1
        )
        self.voting_clf.fit(X_train, y_train)
        print("   ✅ Ensemble trained (LR + SVM + RF)")

        print("\n📈 Validation performance:")
        y_pred = self.voting_clf.predict(X_val)
        print(f"   Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
        print(f"   Precision: {precision_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"   Recall:    {recall_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"   F1:        {f1_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")

        val_probabilities = self.voting_clf.predict_proba(X_val)
        self.threshold    = self._find_optimal_threshold(y_val, val_probabilities)
        print(f"   ✅ Confidence threshold: {self.threshold:.3f}")

        if self.use_outlier_detection:
            print("\n🔍 Training outlier detector (IsolationForest)...")
            self.outlier_detector = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
            self.outlier_detector.fit(X_train.toarray())
            outliers = (self.outlier_detector.predict(X_val.toarray()) == -1).sum()
            print(f"   ✅ {outliers:,} outliers detected in validation ({outliers/X_val.shape[0]*100:.1f}%)")
        else:
            print("\n⏭️  Outlier detection disabled")

        print("\n✅ Training complete")

    def _find_optimal_threshold(self, y_val, probabilities):
        """Find the confidence threshold that maximises accuracy × coverage."""
        max_probs = probabilities.max(axis=1)
        mean_p    = max_probs.mean()
        med_p     = np.median(max_probs)

        print(f"   Confidence stats — mean: {mean_p:.3f} | median: {med_p:.3f}")

        best_threshold = 0.35
        best_score     = 0

        for threshold in np.linspace(0.2, 0.5, 30):
            kept_pct = (max_probs >= threshold).sum() / max_probs.shape[0]
            if kept_pct < 0.80:
                continue
            accuracy = accuracy_score(y_val, probabilities.argmax(axis=1))
            score    = accuracy * kept_pct
            if score > best_score:
                best_threshold = threshold
                best_score     = score

        return min(best_threshold, 0.45)

    def predict(self, utterances):
        """Predict intents. Returns (labels, confidence_scores)."""
        if self.voting_clf is None:
            raise ValueError("Model not trained — call train() first.")

        utterances  = pd.Series(utterances).astype(str).str.lower().str.strip()
        X           = self.vectorizer.transform(utterances)
        probabilities = self.voting_clf.predict_proba(X)
        max_probs   = probabilities.max(axis=1)
        pred_indices = probabilities.argmax(axis=1)

        if self.use_outlier_detection and self.outlier_detector is not None:
            is_outlier = self.outlier_detector.predict(X.toarray()) == -1
        else:
            is_outlier = np.zeros(len(utterances), dtype=bool)

        predictions = []
        for pred_idx, max_prob, outlier in zip(pred_indices, max_probs, is_outlier):
            if outlier or max_prob < self.threshold:
                predictions.append('fallback')
            else:
                predictions.append(self.label_encoder.inverse_transform([pred_idx])[0])

        return predictions, max_probs


# ── Sampling rules ─────────────────────────────────────────────────────────────

def apply_sampling_rules(df, high_impact_intents):
    """
    Build a stratified QA sample from the full marked dataset.

    Sampling rules (applied in order, no overlap):
      Rule 1 — 30% of mismatches (excluding NoMatch rows)
      Rule 2 — 20% of high-impact intent matches (excluding NoMatch)
      Rule 3 — 15% of NoMatch utterances
      Rule 4 — 5%  of low-confidence matches (confidence 0.3–0.5)
      Rule 5 — 3%  of all remaining rows
    """
    print(f"\n🎲 Applying sampling rules...")
    samples     = []
    used_indices = set()

    # Rule 1
    r1 = df[
        (df['match'] == 'no') &
        (~df['ask_action_result'].astype(str).str.startswith('NoMatch')) &
        (df['ask_action_result'].notna()) &
        (df['ask_action_result'].astype(str).str.strip() != '')
    ]
    s1 = r1.sample(frac=0.30, random_state=42) if len(r1) > 0 else pd.DataFrame()
    samples.append(s1); used_indices.update(s1.index)
    print(f"   Rule 1: {len(s1):,} rows (30% of {len(r1):,} mismatches, excl. NoMatch)")

    # Rule 2
    r2 = df[
        (df['match'] == 'yes') &
        (df['intent_name'].isin(high_impact_intents)) &
        (~df['ask_action_result'].astype(str).str.startswith('NoMatch')) &
        (~df.index.isin(used_indices))
    ]
    s2 = r2.sample(frac=0.20, random_state=42) if len(r2) > 0 else pd.DataFrame()
    samples.append(s2); used_indices.update(s2.index)
    print(f"   Rule 2: {len(s2):,} rows (20% of {len(r2):,} high-impact matches)")

    # Rule 3
    r3 = df[df['ask_action_result'].astype(str).str.startswith('NoMatch')]
    s3 = r3.sample(frac=0.15, random_state=42) if len(r3) > 0 else pd.DataFrame()
    samples.append(s3); used_indices.update(s3.index)
    print(f"   Rule 3: {len(s3):,} rows (15% of {len(r3):,} NoMatch utterances)")

    # Rule 4
    r4 = df[
        (df['match'] == 'yes') &
        (df['intent_confidence'] >= 0.3) &
        (df['intent_confidence'] <= 0.5) &
        (~df.index.isin(used_indices))
    ]
    s4 = r4.sample(frac=0.05, random_state=42) if len(r4) > 0 else pd.DataFrame()
    samples.append(s4); used_indices.update(s4.index)
    print(f"   Rule 4: {len(s4):,} rows (5% of {len(r4):,} low-confidence matches)")

    # Rule 5
    r5 = df[~df.index.isin(used_indices)]
    s5 = r5.sample(frac=0.03, random_state=42) if len(r5) > 0 else pd.DataFrame()
    samples.append(s5)
    print(f"   Rule 5: {len(s5):,} rows (3% of {len(r5):,} remaining)")

    combined = pd.concat(samples, ignore_index=True)
    print(f"\n   ✅ Total sample: {len(combined):,} rows")
    return combined


# ── Pipeline ───────────────────────────────────────────────────────────────────

def process_voicebot_data(input_file, output_full_file, output_filtered_file,
                          training_file='training_data/brand_a_training_data.csv'):
    """
    Full pipeline:
      1. Train classifier on labeled data
      2. Load formatted voicebot CSV
      3. Predict intents, compare to found intent
      4. Write full output (all records)
      5. Write filtered output (stratified QA sample)
    """
    print("=" * 80)
    print("AUTOMATED INTENT MARKING — DUAL OUTPUT")
    print("=" * 80)

    classifier = IntentClassifier(contamination=0.01, threshold=0.35, use_outlier_detection=True)
    training_data = classifier.load_training_data(training_file)
    classifier.train(training_data)

    print(f"\n📂 Loading voicebot data: {input_file}")
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')

    print(f"   ✅ {len(df):,} records loaded")

    required_cols = ['user_input', 'intent_name', 'Child Intent']
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # ── Output 1: Full ────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nOUTPUT 1: FULL DATA\n{'='*80}")

    child_rows     = df[df['Child Intent'] == 'Yes'].copy()
    non_child_rows = df[df['Child Intent'] != 'Yes'].copy()

    print(f"   Child intent rows (not marked): {len(child_rows):,}")
    print(f"   Non-child rows (to mark):        {len(non_child_rows):,}")

    if len(non_child_rows) > 0:
        print(f"\n🤖 Predicting intents for {len(non_child_rows):,} rows...")
        non_child_rows['user_input'] = non_child_rows['user_input'].fillna('').astype(str)
        predicted_intents, confidence_scores = classifier.predict(non_child_rows['user_input'])

        non_child_rows['automated_intent'] = predicted_intents
        non_child_rows['match'] = np.where(
            non_child_rows['intent_name'] == non_child_rows['automated_intent'], 'yes', 'no'
        )
        non_child_rows.loc[non_child_rows['match'] == 'yes', 'True/False'] = 'TRUE'

        matches    = (non_child_rows['match'] == 'yes').sum()
        match_rate = matches / len(non_child_rows) * 100
        print(f"\n📊 Match rate: {matches:,} / {len(non_child_rows):,} ({match_rate:.1f}%)")

    if len(child_rows) > 0:
        child_rows['automated_intent'] = ''
        child_rows['match']            = ''

    df_full = pd.concat([non_child_rows, child_rows], ignore_index=True)

    final_columns = [
        'session_id', 'conversation_id', 'date_created', 'date_completed',
        'user_input', 'bot_prompts_all', 'action_name', 'action_type', 'action_number',
        'Child Intent', 'ask_action_result', 'intent_name', 'automated_intent',
        'match', 'intent_confidence', 'True/False', 'Positive/Negative',
        'Correct Intent', 'Out of scope', 'Notes'
    ]
    df_full = df_full[[c for c in final_columns if c in df_full.columns]]

    print(f"\n💾 Saving full output: {output_full_file}")
    df_full.to_csv(output_full_file, index=False, encoding='utf-8')
    print(f"   ✅ {len(df_full):,} records")

    # ── Output 2: Filtered ────────────────────────────────────────────────────
    print(f"\n{'='*80}\nOUTPUT 2: FILTERED/SAMPLED DATA\n{'='*80}")

    df_no_child = df[df['Child Intent'] != 'Yes'].copy()
    print(f"   After removing child intents: {len(df_no_child):,} rows")

    pre = len(df_no_child)
    df_no_child = df_no_child[
        ~df_no_child['ask_action_result'].isin(['SuccessConfirmationYes', 'SuccessConfirmationNo'])
    ]
    removed = pre - len(df_no_child)
    if removed:
        print(f"   🗑️  Removed {removed:,} success confirmation rows")

    nomatch_mask = df_no_child['ask_action_result'].astype(str).str.startswith('NoMatch')
    if nomatch_mask.sum() > 0:
        df_no_child.loc[nomatch_mask, 'intent_name'] = 'fallback'
        print(f"   🔄 {nomatch_mask.sum():,} NoMatch rows mapped to intent 'fallback'")

    print(f"\n🤖 Predicting intents for filtered data...")
    df_no_child['user_input'] = df_no_child['user_input'].fillna('').astype(str)
    predicted_intents, confidence_scores = classifier.predict(df_no_child['user_input'])

    df_no_child['automated_intent'] = predicted_intents
    df_no_child['match'] = np.where(
        df_no_child['intent_name'] == df_no_child['automated_intent'], 'yes', 'no'
    )

    df_sampled = apply_sampling_rules(df_no_child, HIGH_IMPACT_INTENTS)

    df_output2 = pd.DataFrame({
        'brand':            BRAND_LABEL,
        'channel':          CHANNEL_LABEL,
        'date':             df_sampled['date_created'],
        'session_id':       df_sampled['session_id'],
        'conversation_id':  df_sampled['conversation_id'],
        'utterance':        df_sampled['user_input'],
        'found_intent':     df_sampled['intent_name'],
        'automatic_intent': df_sampled['automated_intent'],
        'match':            df_sampled['match'],
        'confidence_score': df_sampled['intent_confidence'],
        'T/F':              df_sampled['True/False'],
        'P/N':              df_sampled['Positive/Negative'],
    })

    print(f"\n💾 Saving filtered output: {output_filtered_file}")
    df_output2.to_csv(output_filtered_file, index=False, encoding='utf-8')
    print(f"   ✅ {len(df_output2):,} sampled records")

    print("\n" + "=" * 80)
    print("✅ DUAL OUTPUT COMPLETE")
    print("=" * 80)
    print(f"\n   Full data:     {output_full_file} ({len(df_full):,} rows)")
    print(f"   Filtered data: {output_filtered_file} ({len(df_output2):,} rows)")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    training_file       = 'training_data/brand_a_training_data.csv'
    input_file          = 'interim_output.csv'
    output_full_file    = 'final_marked_full.csv'
    output_filtered_file = 'final_marked_filtered.csv'

    if len(sys.argv) >= 2: input_file          = sys.argv[1]
    if len(sys.argv) >= 3: output_full_file    = sys.argv[2]
    if len(sys.argv) >= 4: output_filtered_file = sys.argv[3]
    if len(sys.argv) >= 5: training_file       = sys.argv[4]

    try:
        process_voicebot_data(input_file, output_full_file, output_filtered_file, training_file)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
