"""
Automated Intent Marking for Genesys Voicebot Phrases - Dual Output Version

This script:
1. Trains a classification model on labeled training data
2. Predicts intents for voicebot user phrases
3. Produces TWO outputs:
   - Output 1 (Full): All data with automated marking (excluding child intents)
   - Output 2 (Filtered): Sampled data based on specific rules

Usage:
    python automated_intent_marking.py

Requirements:
    - Training data: {BRAND_LABEL}_voicebot_training_data.csv (utterance, intent/ka columns)
    - Input CSV: Output from interim_phrase_tuning_sheet.py
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
import re

warnings.filterwarnings('ignore')

# CONFIGURABLE PARAMETERS
HIGH_IMPACT_INTENTS = [
    'salesNew',
    'billingBill',
    'billingBillColl',
    'billingBillPay',
    'billingCcReg',
    'cancellation',
    'internetSupport',
    'mobileSupport'
]

EXCLUDED_UTTERANCES = [
    'account',
    'account enquiry',
    'account inquiry',
    'internet',
    'internet plan',
    'internet service',
    'nbn',
    'nbn plan',
    'nbn service',
]

BRAND_LABEL = ''  # Change as needed
CHANNEL_LABEL = 'voice'  # Change as needed

#Normalization helper for excluded utterances list
def normalize_utterance(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text) #remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Variable to capture and not include utterances in filtered sheet
NORMALIZED_EXCLUDED_UTTERANCES = set(normalize_utterance(u) for u in EXCLUDED_UTTERANCES)


class IntentClassifier:
    """
    Intent classification model with ensemble learning and outlier detection
    """
    
    def __init__(self, contamination=0.01, threshold=0.35, use_outlier_detection=True):
        """
        Initialize classifier
        
        Args:
            contamination: Expected proportion of outliers (default 1%)
            threshold: Confidence threshold for fallback detection (default 0.35)
            use_outlier_detection: Whether to use outlier detection (default True)
        """
        self.contamination = contamination
        self.threshold = threshold
        self.use_outlier_detection = use_outlier_detection
        self.vectorizer = None
        self.label_encoder = None
        self.voting_clf = None
        self.outlier_detector = None
        
    def load_training_data(self, training_file):
        """
        Load and validate training data
        
        Args:
            training_file: Path to training CSV file
            
        Returns:
            DataFrame with utterance and intent columns
        """
        print(f"\n📂 Loading training data: {training_file}")
        
        if not Path(training_file).exists():
            raise FileNotFoundError(f"Training file not found: {training_file}")
        
        # Load training data
        df = pd.read_csv(training_file, encoding='utf-8')
        
        # Handle different column names (intent/ka, intent, or ka)
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
        
        # Standardize column names
        df = df.rename(columns={intent_col: 'intent'})
        df = df[['utterance', 'intent']].copy()
        
        # Clean and preprocess
        df['utterance'] = df['utterance'].astype(str).str.lower().str.strip()
        df['intent'] = df['intent'].astype(str).str.strip()
        
        # Remove empty rows
        df = df[(df['utterance'] != '') & (df['intent'] != '')]
        
        print(f"   ✅ Loaded {len(df):,} training samples")
        print(f"   ✅ Unique intents: {df['intent'].nunique()}")
        
        # Show intent distribution
        print(f"\n📊 Intent Distribution (top 10):")
        intent_counts = df['intent'].value_counts().head(10)
        for intent, count in intent_counts.items():
            print(f"   {intent}: {count:,} samples")
        
        return df
    
    def train(self, training_data):
        """
        Train the intent classification model
        
        Args:
            training_data: DataFrame with utterance and intent columns
        """
        print(f"\n🔧 Training classification model...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(training_data['intent'])
        
        # Vectorize text using TF-IDF
        print("   Converting text to TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(training_data['utterance'])
        
        print(f"   ✅ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights_dict = dict(zip(np.unique(y), class_weights))
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Validation set: {X_val.shape[0]:,} samples")
        
        # Train models
        print("\n   Training ensemble models...")
        
        log_reg = LogisticRegression(
            class_weight=class_weights_dict,
            max_iter=1000,
            random_state=42
        )
        
        svm = SVC(
            probability=True,
            class_weight=class_weights_dict,
            random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights_dict,
            random_state=42,
            n_jobs=-1
        )
        
        self.voting_clf = VotingClassifier(
            estimators=[
                ('lr', log_reg),
                ('svc', svm),
                ('rf', rf)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        self.voting_clf.fit(X_train, y_train)
        print("   ✅ Ensemble model trained")
        
        # Evaluate
        print("\n📈 Model Performance on Validation Set:")
        y_pred = self.voting_clf.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        
        # Find optimal threshold
        print("\n🎯 Finding optimal confidence threshold...")
        val_probabilities = self.voting_clf.predict_proba(X_val)
        self.threshold = self._find_optimal_threshold(y_val, val_probabilities)
        print(f"   ✅ Optimal threshold: {self.threshold:.3f}")
        
        # Train outlier detector
        if self.use_outlier_detection:
            print("\n🔍 Training outlier detector...")
            self.outlier_detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.outlier_detector.fit(X_train.toarray())
            
            outliers_detected = (self.outlier_detector.predict(X_val.toarray()) == -1).sum()
            outlier_pct = (outliers_detected / X_val.shape[0]) * 100
            print(f"   ✅ Outlier detector trained (contamination={self.contamination})")
            print(f"   📊 Detected {outliers_detected:,} outliers in validation ({outlier_pct:.1f}%)")
        else:
            print("\n⏭️  Outlier detection disabled")
        
        print("\n✅ Model training complete!")
    
    def _find_optimal_threshold(self, y_val, probabilities):
        """Find optimal confidence threshold"""
        max_probs = probabilities.max(axis=1)
        
        mean_prob = max_probs.mean()
        median_prob = np.median(max_probs)
        std_prob = max_probs.std()
        
        print(f"   Validation confidence stats:")
        print(f"      Mean:   {mean_prob:.3f}")
        print(f"      Median: {median_prob:.3f}")
        print(f"      Std:    {std_prob:.3f}")
        
        thresholds = np.linspace(0.2, 0.5, 30)
        best_threshold = 0.35
        best_score = 0
        
        for threshold in thresholds:
            y_pred = probabilities.argmax(axis=1)
            kept_pct = (max_probs >= threshold).sum() / max_probs.shape[0]
            
            if kept_pct < 0.80:
                continue
            
            accuracy = accuracy_score(y_val, y_pred)
            score = accuracy * kept_pct
            
            if score > best_score:
                best_threshold = threshold
                best_score = score
        
        best_threshold = min(best_threshold, 0.45)
        
        return best_threshold
    
    def predict(self, utterances):
        """Predict intents for utterances"""
        if self.voting_clf is None:
            raise ValueError("Model not trained. Call train() first.")
        
        utterances = pd.Series(utterances).astype(str).str.lower().str.strip()
        X = self.vectorizer.transform(utterances)
        
        probabilities = self.voting_clf.predict_proba(X)
        max_probs = probabilities.max(axis=1)
        predicted_indices = probabilities.argmax(axis=1)
        
        if self.use_outlier_detection and self.outlier_detector is not None:
            is_outlier = self.outlier_detector.predict(X.toarray()) == -1
        else:
            is_outlier = np.zeros(len(utterances), dtype=bool)
        
        predictions = []
        for i, (pred_idx, max_prob, outlier) in enumerate(zip(predicted_indices, max_probs, is_outlier)):
            if outlier or max_prob < self.threshold:
                predictions.append('fallback')
            else:
                predictions.append(self.label_encoder.inverse_transform([pred_idx])[0])
        
        return predictions, max_probs


def apply_sampling_rules(df, high_impact_intents):
    """
    Apply 5 sampling rules to create filtered output
    
    Args:
        df: DataFrame with automated marking complete
        high_impact_intents: List of high-impact intent names
        
    Returns:
        DataFrame with sampled rows
    """
    print(f"\n🎲 Applying sampling rules...")
    
    samples = []
    used_indices = set()
    
    # Rule 1: 30% of mismatches (match='no'), exclude NoMatch
    rule1_df = df[
        (df['match'] == 'no') & 
        (~df['ask_action_result'].astype(str).str.startswith('NoMatch')) &
        (df['ask_action_result'].notna()) &
        (df['ask_action_result'].astype(str).str.strip() != '')
    ]
    rule1_sample = rule1_df.sample(frac=0.30, random_state=42) if len(rule1_df) > 0 else pd.DataFrame()
    samples.append(rule1_sample)
    used_indices.update(rule1_sample.index.tolist())
    print(f"   Rule 1: {len(rule1_sample):,} rows (30% of {len(rule1_df):,} mismatches, excluding NoMatch)")
    
    # Rule 2: 20% of high-impact matches, exclude NoMatch
    rule2_df = df[
        (df['match'] == 'yes') &
        (df['intent_name'].isin(high_impact_intents)) &
        (~df['ask_action_result'].astype(str).str.startswith('NoMatch')) &
        (df['ask_action_result'].notna()) &
        (df['ask_action_result'].astype(str).str.strip() != '')
    ]
    rule2_sample = rule2_df.sample(frac=0.20, random_state=42) if len(rule2_df) > 0 else pd.DataFrame()
    samples.append(rule2_sample)
    used_indices.update(rule2_sample.index.tolist())
    print(f"   Rule 2: {len(rule2_sample):,} rows (20% of {len(rule2_df):,} high-impact matches, excluding NoMatch)")
    
    # Rule 3: 15% of NoMatch utterances (not empty values)
    rule3_df = df[
        df['ask_action_result'].astype(str).str.startswith('NoMatch')
    ]
    rule3_sample = rule3_df.sample(frac=0.15, random_state=42) if len(rule3_df) > 0 else pd.DataFrame()
    samples.append(rule3_sample)
    used_indices.update(rule3_sample.index.tolist())
    print(f"   Rule 3: {len(rule3_sample):,} rows (15% of {len(rule3_df):,} NoMatch utterances)")
    
    # Rule 4: 5% of low confidence matches (0.3-0.5), not in rules 1-2
    rule4_df = df[
        (df['match'] == 'yes') &
        (df['intent_confidence'] >= 0.3) &
        (df['intent_confidence'] <= 0.5) &
        (~df.index.isin(used_indices))
    ]
    rule4_sample = rule4_df.sample(frac=0.05, random_state=42) if len(rule4_df) > 0 else pd.DataFrame()
    samples.append(rule4_sample)
    used_indices.update(rule4_sample.index.tolist())
    print(f"   Rule 4: {len(rule4_sample):,} rows (5% of {len(rule4_df):,} low confidence matches)")
    
    # Rule 5: 3% of everything else not in rules 1-4
    rule5_df = df[~df.index.isin(used_indices)]
    rule5_sample = rule5_df.sample(frac=0.03, random_state=42) if len(rule5_df) > 0 else pd.DataFrame()
    samples.append(rule5_sample)
    print(f"   Rule 5: {len(rule5_sample):,} rows (3% of {len(rule5_df):,} remaining utterances)")
    
    # Combine all samples
    combined = pd.concat(samples, ignore_index=True)
    print(f"\n   ✅ Total sampled rows: {len(combined):,}")
    
    return combined


def process_voicebot_data(input_file, output_full_file, output_filtered_file, training_file='{BRAND_LABEL}_voicebot_training_data.csv'):
    """
    Main processing pipeline with dual output
    
    Args:
        input_file: Path to CSV from interim_phrase_tuning_sheet.py
        output_full_file: Path for full output CSV
        output_filtered_file: Path for filtered/sampled output CSV
        training_file: Path to training data CSV
    """
    print("=" * 80)
    print("AUTOMATED INTENT MARKING - DUAL OUTPUT")
    print("=" * 80)
    
    # Initialize classifier
    classifier = IntentClassifier(
        contamination=0.01,
        threshold=0.35,
        use_outlier_detection=True
    )
    
    # Load and train model
    training_data = classifier.load_training_data(training_file)
    classifier.train(training_data)
    
    # Load voicebot data
    print(f"\n📂 Loading voicebot data: {input_file}")
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
    
    print(f"   ✅ Loaded {len(df):,} records")
    
    # Validate required columns
    required_cols = ['user_input', 'intent_name', 'Child Intent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # ====================
    # OUTPUT 1: FULL DATA
    # ====================
    print(f"\n" + "=" * 80)
    print("OUTPUT 1: FULL DATA (Excluding Child Intents from Marking)")
    print("=" * 80)
    
    # Separate child and non-child rows
    child_rows = df[df['Child Intent'] == 'Yes'].copy()
    non_child_rows = df[df['Child Intent'] != 'Yes'].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Child intent rows: {len(child_rows):,} (will NOT be marked)")
    print(f"   Non-child rows: {len(non_child_rows):,} (will be marked)")
    
    # Predict intents for non-child rows only
    if len(non_child_rows) > 0:
        print(f"\n🤖 Predicting intents for {len(non_child_rows):,} non-child phrases...")
        non_child_rows['user_input'] = non_child_rows['user_input'].fillna('').astype(str)
        
        predicted_intents, confidence_scores = classifier.predict(non_child_rows['user_input'])
        
        # Add predictions to non-child rows
        non_child_rows['automated_intent'] = predicted_intents
        
        # Compare predictions
        non_child_rows['match'] = np.where(
            non_child_rows['intent_name'] == non_child_rows['automated_intent'],
            'yes',
            'no'
        )
        
        # Populate True/False
        non_child_rows.loc[non_child_rows['match'] == 'yes', 'True/False'] = 'TRUE'
        
        # Stats
        matches = (non_child_rows['match'] == 'yes').sum()
        match_rate = (matches / len(non_child_rows)) * 100 if len(non_child_rows) > 0 else 0
        
        print(f"\n📊 Non-Child Results:")
        print(f"   Matches (yes):   {matches:,} ({match_rate:.1f}%)")
        print(f"   Mismatches (no): {len(non_child_rows) - matches:,} ({100 - match_rate:.1f}%)")
    
    # Add empty columns to child rows
    if len(child_rows) > 0:
        child_rows['automated_intent'] = ''
        child_rows['match'] = ''
    
    # Combine back
    df_full = pd.concat([non_child_rows, child_rows], ignore_index=True)
    
    # Select final columns (from your specification)
    final_columns_full = [
        'session_id',
        'conversation_id',
        'date_created',
        'date_completed',
        'user_input',
        'bot_prompts_all',
        'action_name',
        'action_type',
        'action_number',
        'Child Intent',
        'ask_action_result',
        'intent_name',
        'automated_intent',
        'match',
        'intent_confidence',
        'True/False',
        'Positive/Negative',
        'Correct Intent',
        'Out of scope',
        'Notes'
    ]
    
    available_cols_full = [col for col in final_columns_full if col in df_full.columns]
    df_full = df_full[available_cols_full]
    
    # Save Output 1
    print(f"\n💾 Saving Output 1: {output_full_file}")
    df_full.to_csv(output_full_file, index=False, encoding='utf-8')
    print(f"   ✅ Saved {len(df_full):,} records")
    
    # ====================
    # OUTPUT 2: FILTERED
    # ====================
    print(f"\n" + "=" * 80)
    print("OUTPUT 2: FILTERED/SAMPLED DATA")
    print("=" * 80)
    
    # Remove child intents for Output 2
    df_no_child = df[df['Child Intent'] != 'Yes'].copy()
    print(f"\n📊 Removed child intents: {len(df_no_child):,} rows remaining")
    
    # Filter out success confirmations
    pre_filter_count = len(df_no_child)
    df_no_child = df_no_child[
        ~df_no_child['ask_action_result'].isin(['SuccessConfirmationYes', 'SuccessConfirmationNo'])
    ]
    removed_success = pre_filter_count - len(df_no_child)
    if removed_success > 0:
        print(f"🗑️  Removed {removed_success:,} success confirmation rows")
    print(f"   {len(df_no_child):,} rows remaining after filtering")
    
    #handling of excluded utterances in output 2 file
    pre_exclude_count = len(df_no_child)
    normalize_utterances = df_no_child['user_input'].apply(normalize_utterance)
    df_no_child = df_no_child[~normalize_utterances.isin(NORMALIZED_EXCLUDED_UTTERANCES)]
    removed_excluded = pre_exclude_count - len(df_no_child)
    if removed_excluded > 0:
        print(f"🚫 Removed {removed_excluded:,} rows matching excluded utterances list")

    # Update found_intent to 'fallback' where ask_action_result starts with 'NoMatch'
    nomatch_mask = df_no_child['ask_action_result'].astype(str).str.startswith('NoMatch')
    nomatch_count = nomatch_mask.sum()
    if nomatch_count > 0:
        df_no_child.loc[nomatch_mask, 'intent_name'] = 'fallback'
        print(f"🔄 Updated {nomatch_count:,} NoMatch rows to have found_intent='fallback'")
    
    # Predict intents
    print(f"\n🤖 Predicting intents for filtered data...")
    df_no_child['user_input'] = df_no_child['user_input'].fillna('').astype(str)
    
    predicted_intents, confidence_scores = classifier.predict(df_no_child['user_input'])
    
    df_no_child['automated_intent'] = predicted_intents
    df_no_child['match'] = np.where(
        df_no_child['intent_name'] == df_no_child['automated_intent'],
        'yes',
        'no'
    )
    
    print(f"   ✅ Predictions complete")
    
    # Apply sampling rules
    df_sampled = apply_sampling_rules(df_no_child, HIGH_IMPACT_INTENTS)
    
    # Prepare Output 2 with custom columns
    df_output2 = pd.DataFrame({
        'brand': BRAND_LABEL,
        'channel': CHANNEL_LABEL,
        'date': df_sampled['date_created'],
        'session_id': df_sampled['session_id'],
        'conversation_id': df_sampled['conversation_id'],
        'utterance': df_sampled['user_input'],
        'found_intent': df_sampled['intent_name'],
        'automatic_intent': df_sampled['automated_intent'],
        'match': df_sampled['match'],
        'confidence_score': df_sampled['intent_confidence'],
        'T/F': df_sampled['True/False'],
        'P/N': df_sampled['Positive/Negative']
    })
    
    # Save Output 2
    print(f"\n💾 Saving Output 2: {output_filtered_file}")
    df_output2.to_csv(output_filtered_file, index=False, encoding='utf-8')
    print(f"   ✅ Saved {len(df_output2):,} sampled records")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ DUAL OUTPUT COMPLETE")
    print("=" * 80)
    print(f"\n📄 Output Files:")
    print(f"   Full data:     {output_full_file} ({len(df_full):,} rows)")
    print(f"   Filtered data: {output_filtered_file} ({len(df_output2):,} rows)")
    print("\n💡 Next steps:")
    print(f"   1. Review {output_full_file} for complete marked data")
    print(f"   2. Review {output_filtered_file} for sampled QA data")
    print(f"   3. Update HIGH_IMPACT_INTENTS in script as needed")
    print()
    print("=" * 80)


def main():
    """Main entry point"""
    
    # Default file paths
    training_file = '{BRAND_LABEL}_voicebot_training_data.csv'
    input_file = 'interim_output.csv'
    output_full_file = 'final_marked_output_full.csv'
    output_filtered_file = 'final_output_filtered.csv'
    
    # Check command line arguments
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_full_file = sys.argv[2]
    if len(sys.argv) >= 4:
        output_filtered_file = sys.argv[3]
    if len(sys.argv) >= 5:
        training_file = sys.argv[4]
    
    # Process
    try:
        process_voicebot_data(input_file, output_full_file, output_filtered_file, training_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nUsage:")
        print(f"   python {sys.argv[0]} <input_csv> [output_full_csv] [output_filtered_csv] [training_csv]")
        print("\nDefaults:")
        print(f"   input_csv:          {input_file}")
        print(f"   output_full_csv:    {output_full_file}")
        print(f"   output_filtered_csv: {output_filtered_file}")
        print(f"   training_csv:       {training_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
