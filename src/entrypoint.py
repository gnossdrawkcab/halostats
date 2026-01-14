import json
import time
import subprocess
import os
from pathlib import Path
from halo_paths import data_path

TOKEN_FILE = data_path("tokens.json")
REQUIRED_KEYS = [
    "access_token", "refresh_token", "user_token",
    "xuid", "xsts_token", "spartan_token", "clearance_token"
]

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def load_tokens():
    try:
        with TOKEN_FILE.open('r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load tokens.json: {e}")
        return None

def tokens_valid():
    if not TOKEN_FILE.exists():
        print("❌ tokens.json does not exist.")
        return False

    tokens = load_tokens()
    if not tokens:
        return False

    for key in REQUIRED_KEYS:
        if not tokens.get(key):
            print(f"❌ Missing or empty token: {key}")
            return False

    if time.time() > tokens.get("expires_at", 0):
        print("⚠️ Token expired. Will run auth to refresh.")
        return False

    print("✅ tokens.json is valid and all tokens are present.")
    return True

def run_script(script_name):
    try:
        script_path = SCRIPT_DIR / script_name
        print(f"▶️ Running {script_name}...")
        subprocess.run(["python", str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")

def get_update_interval():
    """Load update interval from settings file or fall back to environment variable."""
    from halo_paths import data_path
    
    default_interval = int(os.getenv("HALO_UPDATE_INTERVAL", "60"))
    settings_path = data_path("settings.json")
    
    if settings_path.exists():
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
                return settings.get("update_interval", default_interval)
        except Exception as e:
            print(f"Warning: Could not load settings file: {e}")
    
    return default_interval

def main():
    while True:
        sleep_seconds = get_update_interval()
        
        if not tokens_valid():
            run_script("auth.py")

        if tokens_valid():
            run_script("stats.py")
        else:
            print("❌ Unable to validate tokens after auth. Skipping this cycle.")

        print(f"Sleeping {sleep_seconds} seconds before next run...")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    main()
