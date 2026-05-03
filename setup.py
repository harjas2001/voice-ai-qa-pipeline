"""
Setup script for Voice AI QA Pipeline.
Run once to verify dependencies, directory structure, and config.

Usage:
    python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

APP_DIR    = Path(__file__).parent
SCRIPTS_DIR = APP_DIR / 'scripts'
OUTPUTS_DIR = APP_DIR / 'outputs'
DATA_DIR   = APP_DIR / 'data'
CONFIG_DIR = APP_DIR / 'config'
ML_DIR     = SCRIPTS_DIR / 'training_data'

REQUIRED_SCRIPTS = [
    'genesys_extractor.py',
    'json_to_csv.py',
    'intent_classifier.py',
    'intent_extractor.py',
]

REQUIRED_PACKAGES = [
    'flask',
    'pandas',
    'numpy',
    'scikit-learn',
    'requests',
    'pyyaml',
]


def check_package(pkg):
    try:
        __import__(pkg.replace('-', '_').split('[')[0])
        return True
    except ImportError:
        return False


def main():
    print("=" * 60)
    print("  Voice AI QA Pipeline — Setup")
    print("=" * 60)
    print()

    # Create required directories
    for d in [SCRIPTS_DIR, OUTPUTS_DIR, DATA_DIR, CONFIG_DIR, ML_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Directories verified")

    # Check packages
    print("\n📦 Checking Python packages...")
    missing_pkgs = []
    for pkg in REQUIRED_PACKAGES:
        installed = check_package(pkg)
        print(f"   {'✅' if installed else '❌'} {pkg}")
        if not installed:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        print(f"\n   Installing: {', '.join(missing_pkgs)}")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '--break-system-packages', '--quiet'
        ] + missing_pkgs)
        print("   ✅ Packages installed")

    # Check script files
    print("\n📄 Checking scripts/...")
    missing_scripts = []
    for script in REQUIRED_SCRIPTS:
        path = SCRIPTS_DIR / script
        exists = path.exists()
        print(f"   {'✅' if exists else '❌'} {script}")
        if not exists:
            missing_scripts.append(script)

    if missing_scripts:
        print(f"\n⚠️  Missing scripts — copy these to scripts/:")
        for s in missing_scripts:
            print(f"   • {s}")

    # Check config
    print("\n⚙️  Checking config/...")
    brands_cfg = CONFIG_DIR / 'brands.json'
    if brands_cfg.exists():
        print("   ✅ config/brands.json")
    else:
        print("   ❌ config/brands.json  ← MISSING")
        print("      Copy config/brands.example.json → config/brands.json and populate it")

    cred_file = SCRIPTS_DIR / 'credentials.json'
    if cred_file.exists():
        print("   ✅ scripts/credentials.json")
    else:
        print("   ❌ scripts/credentials.json  ← not yet configured (use the UI)")

    # Check training data
    print("\n📊 Checking training data...")
    td_files = list(ML_DIR.glob('*.csv'))
    if td_files:
        for f in td_files:
            print(f"   ✅ {f.name}")
    else:
        print("   ⚠️  No training CSV files found in scripts/training_data/")
        print("      Generate with:  python scripts/intent_extractor.py <botflow.yaml> <output.csv>")

    print()
    print("=" * 60)
    print("  Launch the app:")
    print()
    print("    python app.py")
    print()
    print("  Then open:  http://localhost:5000")
    print("=" * 60)


if __name__ == '__main__':
    main()
