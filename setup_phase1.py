"""
scripts/setup_phase1.py -- Phase 1 Environment Setup & Validation

Run this FIRST. It:
  1. Creates the new directory structure alongside your existing files
  2. Checks Python version and dependencies
  3. Validates your existing Kalshi auth still works
  4. Validates NWS API access
  5. Reports what's ready and what needs action

Usage:
    cd C:\\Users\\Unitye\\Desktop\\kalshi-weather-bot
    py scripts\\setup_phase1.py
"""

import sys
import os
import importlib
from pathlib import Path

# We run from the project root, so add it to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    print("[1/5] Python version...")
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        print("  ✅ Python %d.%d.%d" % (v.major, v.minor, v.micro))
        return True
    else:
        print("  ⚠️  Python %d.%d.%d — recommend 3.10+" % (v.major, v.minor, v.micro))
        return True  # Still works, just warn


def check_dependencies():
    print("\n[2/5] Dependencies...")
    required = {
        "requests": "HTTP client (existing)",
        "cryptography": "Kalshi RSA-PSS auth (existing)",
        "pandas": "DataFrames (NEW — Phase 1)",
        "pyarrow": "Parquet I/O (NEW — Phase 1)",
        "duckdb": "Local SQL engine (NEW — Phase 1)",
        "tqdm": "Progress bars (NEW — Phase 1)",
    }
    
    missing = []
    for pkg, desc in required.items():
        try:
            importlib.import_module(pkg)
            print("  ✅ %-15s — %s" % (pkg, desc))
        except ImportError:
            print("  ❌ %-15s — %s — MISSING" % (pkg, desc))
            missing.append(pkg)
    
    if missing:
        print("\n  To install missing packages:")
        print("    pip install %s" % " ".join(missing))
        print("  Or install everything:")
        print("    pip install -r requirements.txt")
        return False
    return True


def create_directories():
    print("\n[3/5] Directory structure...")
    dirs = [
        "data/raw/kalshi/trades",
        "data/raw/kalshi/orderbook_snaps",
        "data/raw/kalshi/markets",
        "data/raw/weather/nws_snapshots",
        "data/raw/weather/observations",
        "data/raw/weather/cli_reports",
        "data/curated/features",
        "data/curated/labels",
        "data/curated/calibration",
        "data/reference/contract_terms",
        "models/calibration_params",
        "models/model_cards",
        "reports/backtests",
        "reports/daily",
        "notebooks",
        "scripts",
    ]
    
    created = 0
    existed = 0
    for d in dirs:
        p = PROJECT_ROOT / d
        if p.exists():
            existed += 1
        else:
            p.mkdir(parents=True, exist_ok=True)
            created += 1
    
    # Keep existing dirs too (don't break anything)
    for legacy in ["data/forecast_snapshots", "data/observations", "keys", "logs"]:
        lp = PROJECT_ROOT / legacy
        if lp.exists():
            print("  📁 Keeping existing: %s" % legacy)
    
    print("  ✅ %d directories created, %d already existed" % (created, existed))
    return True


def check_kalshi_auth():
    print("\n[4/5] Kalshi API auth...")
    try:
        import config
        from kalshi_auth import KalshiAuth
        
        key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
        if not key_path.exists():
            print("  ❌ Private key not found: %s" % key_path)
            return False
        
        auth = KalshiAuth(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)
        test_headers = auth.headers("GET", "/trade-api/v2/portfolio/balance")
        print("  ✅ RSA key loaded, signature generated")
        print("     Key ID: %s..." % config.KALSHI_API_KEY_ID[:12])
        
        # Quick live test
        import requests
        url = config.KALSHI_PROD_URL + config.KALSHI_API_PATH + "/portfolio/balance"
        resp = requests.get(url, headers=test_headers, timeout=10)
        if resp.status_code == 200:
            balance = resp.json().get("balance", 0) / 100.0
            print("  ✅ API connection OK — balance: $%.2f" % balance)
        else:
            print("  ⚠️  API returned status %d (auth may need refresh)" % resp.status_code)
            print("     Response: %s" % resp.text[:200])
        
        return True
    except Exception as e:
        print("  ❌ Auth check failed: %s" % e)
        return False


def check_nws_api():
    print("\n[5/5] NWS API access...")
    try:
        import requests
        import config
        
        url = "https://api.weather.gov/gridpoints/%s/%d,%d/forecast" % (
            config.NWS_GRID_OFFICE, config.NWS_GRID_X, config.NWS_GRID_Y
        )
        headers = {
            "User-Agent": config.NWS_USER_AGENT,
            "Accept": "application/geo+json",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            periods = data.get("properties", {}).get("periods", [])
            if periods:
                first = periods[0]
                print("  ✅ NWS API OK — %s: %d°F" % (first["name"], first["temperature"]))
            else:
                print("  ⚠️  NWS returned 200 but no forecast periods")
        else:
            print("  ❌ NWS returned status %d" % resp.status_code)
        
        return True
    except Exception as e:
        print("  ❌ NWS check failed: %s" % e)
        return False


def check_existing_archives():
    """Report what archived data already exists."""
    print("\n[Bonus] Existing data archives...")
    
    import config
    
    # Check existing forecast snapshots
    snap_dir = config.FORECAST_SNAPSHOTS_DIR
    if snap_dir.exists():
        files = list(snap_dir.glob("*.json"))
        if files:
            dates = sorted(set(f.stem.split("_")[1][:8] for f in files if "_" in f.stem))
            print("  📊 Forecast snapshots: %d files" % len(files))
            if dates:
                print("     Date range: %s to %s" % (dates[0], dates[-1]))
                print("     Unique dates: %d" % len(dates))
        else:
            print("  📊 Forecast snapshots dir exists but empty")
    else:
        print("  📊 No forecast snapshots directory found")
    
    # Check existing observations
    obs_dir = config.OBSERVATIONS_DIR
    if obs_dir.exists():
        files = list(obs_dir.glob("*.json"))
        print("  📊 Observation snapshots: %d files" % len(files))
    else:
        print("  📊 No observations directory found")


def main():
    print("=" * 60)
    print("PHASE 1 SETUP — Kalshi Weather Bot")
    print("=" * 60)
    print("Project root: %s" % PROJECT_ROOT)
    print()
    
    results = {}
    results["python"] = check_python_version()
    results["deps"] = check_dependencies()
    results["dirs"] = create_directories()
    results["kalshi"] = check_kalshi_auth()
    results["nws"] = check_nws_api()
    check_existing_archives()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = all(results.values())
    for name, ok in results.items():
        print("  %s %s" % ("✅" if ok else "❌", name))
    
    if not results["deps"]:
        print("\n⚠️  NEXT STEP: Install missing dependencies:")
        print("   cd %s" % PROJECT_ROOT)
        print("   pip install -r requirements.txt")
    elif all_ok:
        print("\n✅ ALL CHECKS PASSED — Ready for Phase 1 data acquisition!")
        print("\nNext steps:")
        print("  1. Register for NCEI CDO API token:")
        print("     https://www.ncdc.noaa.gov/cdo-web/token")
        print("  2. Once you have the token, run:")
        print("     py scripts\\fetch_ghcn_daily.py --token YOUR_TOKEN_HERE")
        print("  3. Then pull Kalshi trade history:")
        print("     py scripts\\fetch_kalshi_trades.py")
    else:
        print("\n⚠️  Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
