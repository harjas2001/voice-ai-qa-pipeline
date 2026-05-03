# voice-ai-qa-pipeline

A Flask-based pipeline that extracts voice AI conversation data from Genesys Cloud, runs automated intent classification using an ML ensemble, and produces QA-ready outputs.

---

## Background

Built as a solo-led project between December 2025 and February 2026 to replace a fully manual voice AI QA process across multiple enterprise deployments.

The pipeline processes 100,000–150,000 voice conversation turns per week. Prior to this, analysts pulled raw data manually and applied intent labels by hand — a process that consumed several hours of analyst time each weekly cycle and introduced inconsistency across different reviewers.

The pipeline reduced that overhead significantly by automating the classification step, generating a stratified QA sample surfacing the highest-signal rows for human review (mismatches, high-impact intents, low-confidence predictions, NoMatch utterances), and delivering outputs directly in the browser.

---

## Pipeline Overview

The pipeline runs as four sequential steps, streamed live to the browser console via Server-Sent Events:

```
Genesys Cloud API
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ STEP 1 │ genesys_extractor.py                       │
│        │ Cursor-paginated pull from Botflows API     │
│        │ Handles interrupts, checkpoints every N pg  │
└────────┴──────────────────┬────────────────────────┘
                            │ raw JSON
                            ▼
┌─────────────────────────────────────────────────────┐
│ STEP 2 │ json_to_csv.py                             │
│        │ Flattens nested schema → flat CSV           │
│        │ Unpacks: conversation, askAction, intent,   │
│        │          botPrompts array                   │
└────────┴──────────────────┬────────────────────────┘
                            │ initial CSV
                            ▼
┌─────────────────────────────────────────────────────┐
│ STEP 3 │ interim_phrase_tuning_sheet_{brand}.py      │
│        │ Generated per-brand from config             │
│        │ Adds: Child Intent flag, True/False,        │
│        │       Positive/Negative, annotation cols    │
└────────┴──────────────────┬────────────────────────┘
                            │ formatted CSV
                            ▼
┌─────────────────────────────────────────────────────┐
│ STEP 4 │ intent_classifier_{brand}.py               │
│        │ TF-IDF + Ensemble (LR + SVM + RF)          │
│        │ IsolationForest outlier detection           │
│        │ Stratified sampling rules → 2 outputs       │
└────────┴──────────────────┬────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       final_full.csv           final_filtered.csv
    All records + marking    Stratified QA sample
```

### Script Reference

| Script | Role |
|---|---|
| `app.py` | Flask orchestrator — routes, pipeline execution, SSE log streaming |
| `index.html` | Dark console UI — brand selector, date picker, live logs, file download |
| `scripts/genesys_extractor.py` | Genesys Cloud Botflows API client (cursor pagination) |
| `scripts/json_to_csv.py` | Nested JSON → flat CSV converter |
| `scripts/intent_classifier.py` | ML classifier base (patched per-brand at runtime) |
| `scripts/intent_extractor.py` | Genesys YAML export → training data CSV |
| `config/brands.json` | Per-brand config: intent lists, child intent ID, training file |
| `scripts/credentials.json` | API credentials (gitignored, managed via UI) |

---

## Setup

```bash
git clone https://github.com/yourusername/voice-ai-qa-pipeline
cd voice-ai-qa-pipeline

python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure brands
cp config/brands.example.json config/brands.json
# Edit config/brands.json — add your brand keys, intent lists, child intent IDs

# Run the setup checker
python setup.py

# Start the app
python app.py
# → http://localhost:5000
```

**Windows:** edit the path in `launch.bat` and double-click to launch.

---

## Configuration

### Brand config — `config/brands.json`

```json
{
  "brand_a": {
    "label": "Brand A",
    "brand_label": "Brand A",
    "child_intent_id": "1234",
    "high_impact_intents": ["intentSalesNew", "intentCancellation"],
    "training_file": "training_data/brand_a_training_data.csv",
    "color": "#0099CC",
    "color_dark": "#006699"
  }
}
```

| Field | Description |
|---|---|
| `child_intent_id` | Action number that identifies a child intent turn (excluded from marking) |
| `high_impact_intents` | Intent names that receive higher sampling weight in filtered output |
| `training_file` | Path (relative to `scripts/`) to the labeled training CSV |
| `color` / `color_dark` | Brand accent colour shown in the UI |

### Credentials — set via UI

Credentials are entered once through the browser (⚙ Configure credentials) and stored in `scripts/credentials.json`. Each brand requires its own Botflow ID; all brands share Client ID, Secret, and Region.

| Field | Description |
|---|---|
| `client_id` | Genesys Cloud OAuth client ID |
| `client_secret` | Genesys Cloud OAuth client secret |
| `region` | Genesys Cloud region (e.g. `mypurecloud.com.au`) |
| `botflow_id_{key}` | Botflow UUID per brand key (e.g. `botflow_id_brand_a`) |

### Environment variables — `.env`

| Variable | Default | Description |
|---|---|---|
| `FLASK_PORT` | `5000` | Port to run Flask on |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `BRANDS_CONFIG` | `config/brands.json` | Override brands config path |
| `CREDENTIALS_FILE` | `scripts/credentials.json` | Override credentials file path |

---

## Training Data

Training data is extracted from a Genesys botflow YAML export using `intent_extractor.py`:

```bash
python scripts/intent_extractor.py botflow_export.yaml scripts/training_data/brand_a_training_data.csv
```

The output CSV has two columns: `utterance` and `intent`. Place it in `scripts/training_data/` and reference it in `config/brands.json` under `training_file`.

---

## Classifier

The intent classifier in `scripts/intent_classifier.py` uses a soft-voting ensemble:

| Component | Detail |
|---|---|
| Vectoriser | TF-IDF, 1–3 grams, 5000 features, English stop words |
| Logistic Regression | Class-weighted, `max_iter=1000` |
| SVM | Probability-enabled, class-weighted |
| Random Forest | 100 estimators, class-weighted |
| Outlier detector | IsolationForest (`contamination=0.01`) |
| Threshold | Optimal confidence threshold found on validation set (0.2–0.45 range) |

Predictions with confidence below the threshold, or flagged as outliers, are assigned the label `fallback`. Class weights are automatically balanced to handle imbalanced intent distributions.

### Filtered output sampling rules

| Rule | Description | Sample rate |
|---|---|---|
| 1 | Mismatches (model ≠ found intent), non-NoMatch | 30% |
| 2 | Matches on high-impact intents, non-NoMatch | 20% |
| 3 | NoMatch utterances | 15% |
| 4 | Low-confidence matches (0.30–0.50), not in rules 1–2 | 5% |
| 5 | All remaining rows | 3% |

---

## Stack

`Python · Flask · pandas · scikit-learn · TF-IDF · Genesys Cloud API · Server-Sent Events`
