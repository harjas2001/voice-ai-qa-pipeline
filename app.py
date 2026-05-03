"""
Voice AI QA Pipeline — Flask Backend

Orchestrates a 4-step pipeline:
  1. Extract conversation turn data from the Genesys Cloud Botflows API
  2. Flatten nested JSON to CSV
  3. Format CSV and add QA annotation columns
  4. Auto-mark intent accuracy using a trained ML ensemble classifier

Brand configuration is loaded from config/brands.json (gitignored).
See config/brands.example.json for the required structure.
"""

from flask import Flask, request, jsonify, Response, send_file
import subprocess
import sys
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
import re
import os

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
SCRIPTS_DIR    = BASE_DIR / 'scripts'
OUTPUTS_DIR    = BASE_DIR / 'outputs'
BRANDS_CONFIG  = Path(os.getenv('BRANDS_CONFIG', BASE_DIR / 'config' / 'brands.json'))
CREDENTIALS_FILE = Path(os.getenv('CREDENTIALS_FILE', SCRIPTS_DIR / 'credentials.json'))

OUTPUTS_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR.mkdir(exist_ok=True)


# ── Brand config ───────────────────────────────────────────────────────────────

def load_brand_config():
    if not BRANDS_CONFIG.exists():
        raise FileNotFoundError(
            f"\n[config] Brand config not found: {BRANDS_CONFIG}\n"
            f"  Copy config/brands.example.json → config/brands.json and populate it.\n"
        )
    with open(BRANDS_CONFIG, encoding='utf-8') as f:
        return json.load(f)


try:
    BRAND_CONFIG = load_brand_config()
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    BRAND_CONFIG = {}


# ── Active extraction state ────────────────────────────────────────────────────

extraction_state = {
    'running':      False,
    'log_queue':    queue.Queue(),
    'process':      None,
    'output_files': {},
    'brand':        None,
    'started_at':   None,
    'error':        None,
}


# ── Credential helpers ─────────────────────────────────────────────────────────

def load_credentials():
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        with open(CREDENTIALS_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_brand_credentials(brand):
    """Return credentials dict with the correct botflow_id for the given brand."""
    creds = load_credentials()
    if not creds:
        return None
    botflow_id = creds.get(f'botflow_id_{brand}', '')
    if not botflow_id:
        return None
    return {
        'client_id':     creds.get('client_id', ''),
        'client_secret': creds.get('client_secret', ''),
        'region':        creds.get('region', ''),
        'botflow_id':    botflow_id,
        'verify_cert':   creds.get('verify_cert', True),
    }


def write_temp_credentials(brand):
    """Write a temporary per-brand credentials file for the extractor script."""
    brand_creds = get_brand_credentials(brand)
    if not brand_creds:
        raise ValueError(f'No credentials configured for brand: {brand}')
    temp_path = SCRIPTS_DIR / f'credentials_{brand}_temp.json'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(brand_creds, f, indent=2)
    return temp_path


# ── Script generation ──────────────────────────────────────────────────────────

def generate_interim_script(brand_key):
    """Generate the CSV formatting script for a specific brand."""
    config   = BRAND_CONFIG[brand_key]
    child_id = config['child_intent_id']
    label    = config['label']

    return f'''"""
Auto-generated interim formatter for: {label}
Adds QA annotation columns and classifies child vs parent intents.
"""
import pandas as pd
import sys


def parse_and_reformat_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    print(f"Loaded {{len(df):,}} rows")

    ask_action_number_idx = df.columns.get_loc('action_number')

    df['Child Intent'] = df['action_number'].apply(
        lambda x: 'No' if str(x) == '{child_id}' else 'Yes'
    )

    cols = df.columns.tolist()
    cols.remove('Child Intent')
    cols.insert(ask_action_number_idx + 1, 'Child Intent')
    df = df[cols]

    df['True/False']       = ''
    df['Positive/Negative'] = ''
    df['Correct Intent']   = ''
    df['Out of scope']     = ''
    df['Notes']            = ''

    df['Positive/Negative'] = df['ask_action_result'].apply(
        lambda x: 'Negative' if str(x).startswith('NoMatch') else 'Positive'
    )

    df.to_csv(output_file, index=False)
    print(f"Saved output to: {{output_file}}")
    print(f"Total rows: {{len(df)}}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_csv, output_csv = sys.argv[1], sys.argv[2]
    else:
        input_csv, output_csv = 'input.csv', 'output.csv'
    try:
        parse_and_reformat_csv(input_csv, output_csv)
    except FileNotFoundError:
        print("Error: Input file not found!")
    except KeyError as e:
        print(f"Error: Column {{e}} not found!")
    except Exception as e:
        print(f"An error occurred: {{str(e)}}")
'''


def generate_marking_script(brand_key):
    """Patch the base intent_classifier.py with brand-specific config."""
    config        = BRAND_CONFIG[brand_key]
    brand_label   = config['brand_label']
    training_file = config['training_file']
    intents_list  = config['high_impact_intents']

    base_path = SCRIPTS_DIR / 'intent_classifier.py'
    if not base_path.exists():
        raise FileNotFoundError(f'Base classifier not found: {base_path}')

    text  = base_path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    # Replace HIGH_IMPACT_INTENTS block
    new_lines = []
    skip = False
    for line in lines:
        if re.match(r'HIGH_IMPACT_INTENTS\s*=\s*\[', line):
            new_lines.append('HIGH_IMPACT_INTENTS = [\n')
            for intent in intents_list:
                new_lines.append(f'    {repr(intent)},\n')
            new_lines.append(']\n')
            skip = True
            continue
        if skip:
            if line.strip() == ']' or line.strip().endswith(']'):
                skip = False
            continue
        new_lines.append(line)
    text = ''.join(new_lines)

    # Replace BRAND_LABEL
    text = re.sub(
        r"BRAND_LABEL\s*=\s*'[^']*'",
        f"BRAND_LABEL = '{brand_label}'",
        text
    )

    # Replace default training file reference in main()
    text = re.sub(
        r"training_file\s*=\s*'[^']*_training_data\.csv'",
        f"training_file = '{training_file}'",
        text
    )

    return text


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    with open(BASE_DIR / 'index.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/brands')
def get_brands():
    """Return available brand keys and display config for the frontend."""
    return jsonify({
        key: {
            'label':      cfg.get('label', key),
            'color':      cfg.get('color', '#0099FF'),
            'color_dark': cfg.get('color_dark', '#006699'),
        }
        for key, cfg in BRAND_CONFIG.items()
    })


@app.route('/api/status')
def status():
    return jsonify({
        'running':      extraction_state['running'],
        'brand':        extraction_state['brand'],
        'started_at':   extraction_state['started_at'],
        'output_files': extraction_state['output_files'],
        'error':        extraction_state['error'],
    })


@app.route('/api/credentials', methods=['GET'])
def get_credentials_route():
    creds = load_credentials()
    if not creds:
        return jsonify({'exists': False})

    response = {
        'exists':          True,
        'region':          creds.get('region', ''),
        'has_client_id':   bool(creds.get('client_id')),
        'has_client_secret': bool(creds.get('client_secret')),
    }
    for key in BRAND_CONFIG:
        response[f'botflow_id_{key}'] = creds.get(f'botflow_id_{key}', '')

    return jsonify(response)


@app.route('/api/credentials', methods=['POST'])
def save_credentials_route():
    data     = request.json
    required = ['client_id', 'client_secret', 'region']
    required += [f'botflow_id_{key}' for key in BRAND_CONFIG]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    payload = {
        'client_id':     data['client_id'],
        'client_secret': data['client_secret'],
        'region':        data['region'],
        'verify_cert':   data.get('verify_cert', True),
    }
    for key in BRAND_CONFIG:
        payload[f'botflow_id_{key}'] = data[f'botflow_id_{key}']

    with open(CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return jsonify({'success': True})


@app.route('/api/extract', methods=['POST'])
def start_extraction():
    global extraction_state

    if not BRAND_CONFIG:
        return jsonify({'error': 'Brand config not loaded. Check config/brands.json.'}), 500

    if extraction_state['running']:
        return jsonify({'error': 'Extraction already running'}), 409

    data       = request.json
    brand      = data.get('brand')
    start_date = data.get('start_date')
    end_date   = data.get('end_date')
    date_label = data.get('date_label', '')

    if brand not in BRAND_CONFIG:
        return jsonify({'error': f'Unknown brand: {brand}'}), 400
    if not start_date or not end_date:
        return jsonify({'error': 'Missing dates'}), 400
    if not get_brand_credentials(brand):
        return jsonify({'error': 'No credentials configured. Set up credentials first.'}), 400

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', date_label) if date_label else timestamp
    prefix     = f"{brand}_{safe_label}"

    run_dir = OUTPUTS_DIR / f"run_{timestamp}_{brand}"
    run_dir.mkdir(exist_ok=True)

    files = {
        'json':           str(run_dir / f'{prefix}_raw.json'),
        'initial_csv':    str(run_dir / f'{prefix}_initial.csv'),
        'interim_csv':    str(run_dir / f'{prefix}_interim.csv'),
        'final_full':     str(run_dir / f'{prefix}_final_full.csv'),
        'final_filtered': str(run_dir / f'{prefix}_final_filtered.csv'),
    }

    config = BRAND_CONFIG[brand]

    # Write brand-specific generated scripts
    interim_script  = SCRIPTS_DIR / f'interim_phrase_tuning_sheet_{brand}.py'
    marking_script  = SCRIPTS_DIR / f'intent_classifier_{brand}.py'
    interim_script.write_text(generate_interim_script(brand), encoding='utf-8')
    marking_script.write_text(generate_marking_script(brand), encoding='utf-8')

    extraction_state = {
        'running':      True,
        'log_queue':    queue.Queue(),
        'process':      None,
        'output_files': files,
        'brand':        brand,
        'started_at':   datetime.now().isoformat(),
        'error':        None,
        'run_dir':      str(run_dir),
    }

    def run_pipeline():
        try:
            _run_pipeline_steps(brand, start_date, end_date, files, config)
        except Exception as e:
            extraction_state['error'] = str(e)
            extraction_state['log_queue'].put(f'\n❌ Fatal error: {e}\n')
        finally:
            extraction_state['running'] = False
            extraction_state['log_queue'].put('__DONE__')

    threading.Thread(target=run_pipeline, daemon=True).start()
    return jsonify({'success': True, 'run_id': timestamp, 'files': files})


def _run_pipeline_steps(brand, start_date, end_date, files, config):
    q = extraction_state['log_queue']

    def stream_process(cmd, step_label):
        q.put(f'\n{"="*60}\n🔄 {step_label}\n{"="*60}\n')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            cwd=str(SCRIPTS_DIR),
        )
        extraction_state['process'] = proc
        for line in proc.stdout:
            q.put(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'{step_label} failed (exit code {proc.returncode})')
        q.put(f'✅ {step_label} complete\n')

    cred_file = write_temp_credentials(brand)

    try:
        stream_process([
            sys.executable,
            str(SCRIPTS_DIR / 'genesys_extractor.py'),
            '--start-date', start_date,
            '--end-date',   end_date,
            '--all',
            '--output',      files['json'],
            '--credentials', str(cred_file),
            '--no-analysis',
            '--samples', '0',
        ], 'STEP 1: Extract from Genesys Cloud API')

        stream_process([
            sys.executable,
            str(SCRIPTS_DIR / 'json_to_csv.py'),
            files['json'],
            files['initial_csv'],
        ], 'STEP 2: Convert JSON to CSV')

        stream_process([
            sys.executable,
            str(SCRIPTS_DIR / f'interim_phrase_tuning_sheet_{brand}.py'),
            files['initial_csv'],
            files['interim_csv'],
        ], 'STEP 3: Format & Add QA Columns')

        training_file = str(SCRIPTS_DIR / config['training_file'])
        stream_process([
            sys.executable,
            str(SCRIPTS_DIR / f'intent_classifier_{brand}.py'),
            files['interim_csv'],
            files['final_full'],
            files['final_filtered'],
            training_file,
        ], 'STEP 4: Automated Intent Marking')

    finally:
        try:
            cred_file.unlink()
        except Exception:
            pass


@app.route('/api/logs')
def stream_logs():
    def generate():
        while True:
            try:
                msg = extraction_state['log_queue'].get(timeout=30)
                if msg == '__DONE__':
                    yield 'data: __DONE__\n\n'
                    break
                yield f'data: {msg.replace(chr(10), "<br>")}\n\n'
            except queue.Empty:
                yield 'data: __HEARTBEAT__\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/download/<file_type>')
def download_file(file_type):
    if file_type not in ('final_full', 'final_filtered'):
        return jsonify({'error': 'Invalid file type'}), 400
    filepath = extraction_state['output_files'].get(file_type)
    if not filepath or not Path(filepath).exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(filepath, as_attachment=True, download_name=Path(filepath).name)


@app.route('/api/cancel', methods=['POST'])
def cancel_extraction():
    proc = extraction_state.get('process')
    if proc and extraction_state['running']:
        proc.terminate()
        extraction_state['running'] = False
        extraction_state['log_queue'].put('\n⚠️  Extraction cancelled by user\n')
        extraction_state['log_queue'].put('__DONE__')
        return jsonify({'success': True})
    return jsonify({'error': 'No active extraction'}), 400


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    print("=" * 60)
    print("  Voice AI QA Pipeline")
    print(f"  http://localhost:{port}")
    print(f"  Brands loaded: {list(BRAND_CONFIG.keys()) or '⚠️  none (check config/brands.json)'}")
    print("=" * 60)
    app.run(debug=debug, port=port, threaded=True)
