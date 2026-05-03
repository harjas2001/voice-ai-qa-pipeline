"""
Genesys Botflow JSON → CSV Converter

Flattens the nested Genesys reporting turn schema into a flat CSV.
Handles nested objects (conversation, askAction, intent) and arrays (botPrompts).
Filters rows with empty user_input before writing output.

Usage:
    python json_to_csv.py input.json output.csv
"""

import json
import csv
import sys
import argparse
from pathlib import Path


def flatten_turn_data(turn):
    """
    Flatten a single reporting turn entity into a CSV-compatible row.

    Genesys nests the following fields — all are unpacked here:
      - conversation.id
      - askAction.actionName / actionType / actionNumber
      - intent.name / confidence
      - botPrompts (array of strings)
    """
    flattened = {
        'session_id':      turn.get('sessionId', ''),
        'conversation_id': turn.get('conversation', {}).get('id', ''),
        'date_created':    turn.get('dateCreated', ''),
        'date_completed':  turn.get('dateCompleted', ''),
        'user_input':      turn.get('userInput', ''),
    }

    # Bot prompts (array)
    bot_prompts = turn.get('botPrompts', [])
    if isinstance(bot_prompts, list) and bot_prompts:
        flattened['bot_prompts_all']   = ' | '.join([str(p).strip() for p in bot_prompts if p])
        flattened['bot_prompt_first']  = str(bot_prompts[0]).strip()
        flattened['bot_prompt_count']  = len(bot_prompts)
    else:
        flattened['bot_prompts_all']  = ''
        flattened['bot_prompt_first'] = ''
        flattened['bot_prompt_count'] = 0

    # Ask action (nested object)
    ask_action = turn.get('askAction', {})
    if isinstance(ask_action, dict):
        flattened['action_name']   = ask_action.get('actionName', '')
        flattened['action_type']   = ask_action.get('actionType', '')
        flattened['action_number'] = ask_action.get('actionNumber', '')
    else:
        flattened['action_name']   = ''
        flattened['action_type']   = ''
        flattened['action_number'] = ''

    flattened['ask_action_result'] = turn.get('askActionResult', '')

    # Intent (nested object)
    intent = turn.get('intent', {})
    if isinstance(intent, dict) and intent:
        flattened['intent_name']       = intent.get('name', '')
        flattened['intent_confidence'] = intent.get('confidence', '')
    else:
        flattened['intent_name']       = ''
        flattened['intent_confidence'] = ''

    return flattened


def json_to_csv(json_file, csv_file):
    """Load JSON, flatten all records, filter empty inputs, write CSV."""
    print(f"📄 Loading: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

    if not data:
        print("⚠️  JSON file is empty")
        return False

    if not isinstance(data, list):
        print("❌ Expected a JSON array at root level")
        return False

    print(f"   ✅ Loaded {len(data):,} records")

    # Flatten
    print(f"\n🔄 Flattening records...")
    flattened_data = []
    for i, turn in enumerate(data, 1):
        try:
            flattened_data.append(flatten_turn_data(turn))
            if i % 1000 == 0:
                print(f"   Processed {i:,} records…")
        except Exception as e:
            print(f"   ⚠️  Skipped record {i}: {e}")

    print(f"   ✅ Flattened {len(flattened_data):,} records")

    if not flattened_data:
        print("❌ No records to export")
        return False

    # Filter empty user_input
    print(f"\n🔍 Filtering empty user_input rows…")
    original_count = len(flattened_data)
    flattened_data = [r for r in flattened_data if r.get('user_input', '').strip()]
    removed_count  = original_count - len(flattened_data)

    print(f"   ✅ Kept {len(flattened_data):,} rows with user input")
    if removed_count:
        pct = removed_count / original_count * 100
        print(f"   🗑️  Removed {removed_count:,} empty rows ({pct:.1f}%)")

    if not flattened_data:
        print("❌ No records remaining after filtering")
        return False

    # Write CSV
    print(f"\n💾 Writing: {csv_file}")
    try:
        fieldnames = list(flattened_data[0].keys())
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)

        print(f"   ✅ {len(flattened_data):,} rows · {len(fieldnames)} columns")
        print(f"\n📋 Columns:")
        for col in fieldnames:
            print(f"   - {col}")
        return True

    except Exception as e:
        print(f"❌ Error writing CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Flatten Genesys botflow JSON to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python json_to_csv.py raw_extract.json initial_extract.csv
        """
    )
    parser.add_argument('input',  help='Input JSON file')
    parser.add_argument('output', help='Output CSV file')
    args = parser.parse_args()

    print("=" * 80)
    print("JSON → CSV CONVERTER")
    print("=" * 80)
    print()

    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    if Path(args.output).exists():
        response = input(f"⚠️  {args.output} already exists. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            sys.exit(0)

    success = json_to_csv(args.input, args.output)

    if success:
        print("\n" + "=" * 80)
        print("✅ CONVERSION COMPLETE")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n❌ Conversion failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
