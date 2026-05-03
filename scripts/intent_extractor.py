"""
Genesys Botflow YAML → Training Data Extractor

Parses a Genesys botflow YAML export and extracts all NLU intent utterances
into a CSV suitable for training the intent classifier.

The output CSV has two columns:
  utterance   — the raw training phrase
  intent      — the intent name from the NLU domain

Usage:
    python intent_extractor.py <input.yaml> <output.csv>
    python intent_extractor.py botflow_export.yaml training_data/brand_a_training_data.csv
"""

import yaml
import csv
import sys
from pathlib import Path


def extract_intents_to_csv(yaml_file_path, output_csv_path):
    """
    Extract all NLU intent utterances from a Genesys botflow YAML export.

    Navigates:
      botFlow → settingsNaturalLanguageUnderstanding
             → nluDomainVersion → intents → utterances → segments
    """
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    intents = (
        data['botFlow']
            ['settingsNaturalLanguageUnderstanding']
            ['nluDomainVersion']
            ['intents']
    )

    rows = []
    for intent in intents:
        intent_name = intent['name']
        print(f"  Processing: {intent_name}")

        for utterance in intent.get('utterances', []):
            for segment in utterance.get('segments', []):
                text = segment.get('text', '').strip()
                if text:
                    rows.append({'utterance': text, 'intent': intent_name})

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['utterance', 'intent'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Extracted {len(rows):,} utterances from {len(intents)} intents")
    print(f"   Saved to: {output_csv_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python intent_extractor.py <input.yaml> [output.csv]")
        print("  input.yaml  — Genesys botflow YAML export")
        print("  output.csv  — destination for training data (default: training_data.csv)")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'training_data.csv'

    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    print(f"📂 Loading: {input_file}")
    extract_intents_to_csv(input_file, output_file)


if __name__ == "__main__":
    main()
