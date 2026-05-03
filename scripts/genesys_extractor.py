"""
Genesys Cloud Botflows Reporting — Data Extractor

Extracts conversation turn data from the Genesys Cloud Botflows Reporting API
using cursor-based pagination (after/nextUri).

Usage:
    python genesys_extractor.py \\
        --start-date 2025-01-06T00:00:00Z \\
        --end-date   2025-01-06T23:59:59Z \\
        --all \\
        --output raw_extract.json \\
        --credentials credentials.json
"""

import json
import requests
from datetime import datetime
import sys
import argparse
import os
import signal
from urllib.parse import urlparse, parse_qs


# ── Global state for graceful interrupt handling ───────────────────────────────

extraction_state = {
    'entities':       [],
    'interrupted':    False,
    'output_filename': None,
    'pages_scanned':  0
}


def signal_handler(sig, frame):
    print("\n\n⚠️  Interrupt received (Ctrl+C)")
    print("💾 Saving collected data before exit...")
    extraction_state['interrupted'] = True


signal.signal(signal.SIGINT, signal_handler)


# ── Credentials ────────────────────────────────────────────────────────────────

def load_credentials(filepath='credentials.json'):
    """Load and validate credentials from JSON file."""
    try:
        with open(filepath, 'r') as f:
            creds = json.load(f)

        required = ['client_id', 'client_secret', 'region', 'botflow_id']
        missing  = [f for f in required if f not in creds]

        if missing:
            print(f"❌ Missing required credential fields: {', '.join(missing)}")
            sys.exit(1)

        creds.setdefault('verify_cert', True)
        return creds

    except FileNotFoundError:
        print(f"❌ Credentials file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in credentials file: {e}")
        sys.exit(1)


# ── Authentication ─────────────────────────────────────────────────────────────

def authenticate(creds):
    """Obtain a Genesys Cloud OAuth access token."""
    print("🔐 Authenticating...")

    token_url = f"https://login.{creds['region']}/oauth/token"

    try:
        response = requests.post(
            token_url,
            data={
                'grant_type': 'client_credentials',
                'scope': 'analytics:botFlowDivisionAwareReportingTurn:view'
            },
            auth=(creds['client_id'], creds['client_secret']),
            timeout=30,
            verify=creds['verify_cert']
        )
        response.raise_for_status()
        print("✅ Authentication successful\n")
        return response.json()['access_token']

    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_raw_data(creds, access_token, start_date_str, end_date_str,
                     max_pages=10, checkpoint_interval=50, output_filename=None):
    """
    Pull turn data using cursor-based pagination (after + nextUri).

    Each page returns up to 250 records. Pagination follows the nextUri
    returned in the response body — pageNumber is not supported by this API.
    """
    base_url = f"https://api.{creds['region']}"
    endpoint = f"/api/v2/analytics/botflows/{creds['botflow_id']}/divisions/reportingturns"

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type':  'application/json'
    }

    all_entities     = []
    page_count       = 0
    checkpoint_count = 0
    next_uri         = None

    extraction_state['output_filename'] = output_filename

    print(f"📊 Starting extraction")
    print(f"   Date range:  {start_date_str} to {end_date_str}")
    print(f"   Bot flow ID: {creds['botflow_id']}")
    print(f"   Max pages:   {'unlimited' if max_pages > 9000 else max_pages}")
    print(f"\n   💡 Press Ctrl+C anytime to save collected data and exit\n")

    while page_count < max_pages:
        page_count += 1

        if extraction_state['interrupted']:
            print(f"\n   ⚠️  Interrupted at page {page_count}")
            break

        # Build request params
        if next_uri:
            parsed = urlparse(next_uri)
            params = parse_qs(parsed.query)
            params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        else:
            params = {
                'pageSize': '250',
                'interval': f"{start_date_str}/{end_date_str}"
            }

        try:
            response = requests.get(
                base_url + endpoint,
                headers=headers,
                params=params,
                timeout=120,
                verify=creds['verify_cert']
            )

            if response.status_code != 200:
                print(f"   ❌ HTTP {response.status_code}")
                if response.status_code == 404:
                    print("   Bot flow ID not found or access denied.")
                elif response.status_code == 400:
                    print(f"   Bad request: {response.text[:300]}")
                break

            data     = response.json()
            entities = data.get('entities', [])

            if not entities:
                print(f"\n   ✅ No more data at page {page_count}")
                break

            all_entities.extend(entities)
            extraction_state['entities']      = all_entities
            extraction_state['pages_scanned'] = page_count

            print(f"   Page {page_count}: {len(entities)} records | Total: {len(all_entities):,}")

            if checkpoint_interval > 0 and page_count % checkpoint_interval == 0:
                checkpoint_count += 1
                if output_filename:
                    save_checkpoint(all_entities, output_filename, checkpoint_count)

            next_uri = data.get('nextUri')
            if not next_uri:
                print(f"\n   ✅ No more pages")
                break

        except requests.exceptions.Timeout:
            print(f"   ⏱️  Request timeout on page {page_count}")
            if output_filename:
                save_checkpoint(all_entities, output_filename, checkpoint_count + 1)
            break
        except Exception as e:
            print(f"   ❌ Error on page {page_count}: {e}")
            if output_filename:
                save_checkpoint(all_entities, output_filename, checkpoint_count + 1)
            break

    print(f"\n📊 Extraction complete")
    print(f"   Pages fetched:  {page_count}")
    print(f"   Total records:  {len(all_entities):,}")

    return all_entities


# ── Checkpoint / save ──────────────────────────────────────────────────────────

def save_checkpoint(entities, filename, checkpoint_num=None):
    """Write a checkpoint file mid-extraction."""
    try:
        if checkpoint_num is not None:
            base, ext     = os.path.splitext(filename)
            checkpoint_file = f"{base}_checkpoint{checkpoint_num}{ext}"
        else:
            checkpoint_file = filename

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, default=str)

        file_size = os.path.getsize(checkpoint_file)
        if checkpoint_num is not None:
            print(f"\n   💾 Checkpoint saved: {checkpoint_file} "
                  f"({len(entities):,} records, {file_size/1024/1024:.2f} MB)")

        return checkpoint_file

    except Exception as e:
        print(f"\n   ❌ Error saving checkpoint: {e}")
        return None


def save_json(entities, filename):
    """Write the final output file."""
    print(f"\n💾 Saving: {filename}...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, default=str)
        file_size = os.path.getsize(filename)
        print(f"✅ Saved {len(entities):,} records ({file_size/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


# ── Structure analysis (optional) ─────────────────────────────────────────────

def analyze_structure(entities):
    """Print a summary of the JSON field structure for debugging."""
    if not entities:
        print("\n⚠️  No data to analyse")
        return

    print("\n" + "=" * 80)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 80)

    all_fields = set()
    for entity in entities:
        all_fields.update(entity.keys())

    print(f"\n📋 Unique fields: {len(all_fields)}")
    for field in sorted(all_fields):
        print(f"   - {field}")

    first = entities[0]
    print(f"\n🔎 Nested objects/arrays:")
    for key, value in first.items():
        if isinstance(value, dict):
            print(f"   📦 {key} (object) → {list(value.keys())}")
        elif isinstance(value, list) and value:
            t = 'array of objects' if isinstance(value[0], dict) else 'array'
            print(f"   📦 {key} ({t})")


def print_sample_json(entities, num_samples=2):
    if not entities or num_samples == 0:
        return
    print(f"\n{'─' * 80}")
    print(f"SAMPLE RECORDS (first {min(num_samples, len(entities))})")
    print(f"{'─' * 80}")
    for i, entity in enumerate(entities[:num_samples], 1):
        print(f"\nRecord {i}:")
        print(json.dumps(entity, indent=2, default=str))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract Genesys Cloud botflow reporting turn data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python genesys_extractor.py --start-date 2025-01-06T00:00:00Z --end-date 2025-01-06T23:59:59Z --all
  python genesys_extractor.py --start-date 2025-01-01T00:00:00Z --end-date 2025-01-07T23:59:59Z --all --output week.json
        """
    )
    parser.add_argument('--start-date', required=True,
                        help='Start datetime (ISO 8601, e.g. 2025-01-06T00:00:00Z)')
    parser.add_argument('--end-date', required=True,
                        help='End datetime (ISO 8601, e.g. 2025-01-06T23:59:59Z)')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum pages to fetch (default: 10)')
    parser.add_argument('--all', action='store_true',
                        help='Fetch all pages (overrides --max-pages)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Checkpoint every N pages (default: 50, 0 to disable)')
    parser.add_argument('--output',
                        help='Output JSON file path (default: auto-generated)')
    parser.add_argument('--credentials', default='credentials.json',
                        help='Path to credentials JSON file (default: credentials.json)')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip field structure analysis')
    parser.add_argument('--samples', type=int, default=2,
                        help='Sample records to print (default: 2, 0 to skip)')

    args = parser.parse_args()

    print("=" * 80)
    print("GENESYS CLOUD — BOTFLOW REPORTING EXTRACTOR")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Press Ctrl+C anytime to save collected data and exit\n")

    max_pages = 999999 if args.all else args.max_pages
    creds     = load_credentials(args.credentials)
    token     = authenticate(creds)

    if args.output:
        output_filename = args.output
    else:
        start_dt = datetime.fromisoformat(args.start_date.replace('Z', '+00:00'))
        output_filename = f"raw_extract_{start_dt.strftime('%Y%m%d')}.json"

    try:
        entities = extract_raw_data(
            creds, token,
            args.start_date, args.end_date,
            max_pages,
            checkpoint_interval=args.checkpoint_interval,
            output_filename=output_filename
        )
    except Exception as e:
        print(f"\n❌ Extraction error: {e}")
        entities = extraction_state.get('entities', [])
        if entities:
            print(f"💾 Saving {len(entities):,} partial records...")

    if extraction_state['entities']:
        entities = extraction_state['entities']

    if not entities:
        print("\n⚠️  No data collected")
        sys.exit(0)

    if not args.no_analysis:
        analyze_structure(entities)

    save_json(entities, output_filename)

    if args.samples > 0:
        print_sample_json(entities, num_samples=args.samples)

    print("\n" + "=" * 80)
    if extraction_state['interrupted']:
        print("⚠️  EXTRACTION INTERRUPTED — PARTIAL DATA SAVED")
    else:
        print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\n📄 Output: {output_filename}")
    print(f"📊 Records: {len(entities):,}")
    print(f"📑 Pages:   {extraction_state['pages_scanned']}")

    return 0 if not extraction_state['interrupted'] else 130


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        if extraction_state['entities'] and extraction_state['output_filename']:
            print("\n\n💾 Final save before exit...")
            try:
                save_json(extraction_state['entities'], extraction_state['output_filename'])
            except Exception:
                print("❌ Could not save data")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if extraction_state['entities'] and extraction_state['output_filename']:
            try:
                save_json(extraction_state['entities'], extraction_state['output_filename'])
            except Exception:
                pass
        sys.exit(1)
