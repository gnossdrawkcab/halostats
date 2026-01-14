import json
import asyncio
import os
from datetime import datetime, timezone as dt_timezone
from aiohttp import ClientSession, ClientResponseError
from spnkr.client import HaloInfiniteClient
import pandas as pd
from pytz import timezone
import random
from sqlalchemy import create_engine, inspect, text
from halo_paths import data_path

update_status = {
    "new_rows_added": False,
    "new_row_count": 0,
    "last_update": None
}

TOKENS_PATH = data_path("tokens.json")
UPDATE_STATUS_PATH = data_path("update_status.json")
SETTINGS_PATH = data_path("settings.json")
DB_NAME = os.getenv("HALO_DB_NAME", "halostatsapi")
DB_USER = os.getenv("HALO_DB_USER", "postgres")
DB_PASSWORD = os.getenv("HALO_DB_PASSWORD")
DB_HOST = os.getenv("HALO_DB_HOST", "halostatsapi")
DB_PORT = os.getenv("HALO_DB_PORT", "5432")

_RUNTIME_SETTINGS_CACHE: dict = {"mtime": None, "settings": {}}
_LAST_LOGGED_MATCH_LIMIT: int | None = None
_LAST_LOGGED_FORCE_REFRESH: bool | None = None


def load_runtime_settings() -> dict:
    """Load settings.json with a lightweight mtime cache.

    This is used by the scraper so Settings page changes can apply mid-run
    without restarting the container.
    """

    if not SETTINGS_PATH.exists():
        _RUNTIME_SETTINGS_CACHE["mtime"] = None
        _RUNTIME_SETTINGS_CACHE["settings"] = {}
        return {}

    try:
        mtime = SETTINGS_PATH.stat().st_mtime
    except Exception:
        mtime = None

    if _RUNTIME_SETTINGS_CACHE.get("mtime") == mtime and isinstance(
        _RUNTIME_SETTINGS_CACHE.get("settings"), dict
    ):
        return _RUNTIME_SETTINGS_CACHE["settings"]

    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        if not isinstance(settings, dict):
            settings = {}
    except Exception:
        settings = {}

    _RUNTIME_SETTINGS_CACHE["mtime"] = mtime
    _RUNTIME_SETTINGS_CACHE["settings"] = settings
    return settings

def get_match_limit():
    """Load match limit from settings file or fall back to environment variable.
    
    This controls how many matches to scan per API call, but the system will
    automatically fetch all historical matches until the database is complete.
    """
    default_limit = int(os.getenv("HALO_MATCH_LIMIT", "500"))

    settings = load_runtime_settings()
    chosen = settings.get("match_limit", default_limit)
    try:
        chosen_int = int(chosen)
    except Exception:
        chosen_int = default_limit

    # Interpret 0 or negative as unlimited
    unlimited = False
    if chosen_int <= 0:
        unlimited = True
        chosen_int = None

    global _LAST_LOGGED_MATCH_LIMIT
    if _LAST_LOGGED_MATCH_LIMIT != chosen_int:
        source = "settings.json" if "match_limit" in settings else "env HALO_MATCH_LIMIT"
        if unlimited:
            print(f"✅ Using match_limit from {source}: unlimited")
        else:
            print(f"✅ Using match_limit from {source}: {chosen_int}")
        _LAST_LOGGED_MATCH_LIMIT = chosen_int

    return chosen_int



def get_force_refresh() -> bool:
    """Always return False - system automatically skips existing matches."""
    return False


def consume_force_refresh_setting() -> bool:
    """Always return False - force refresh is disabled."""
    return False


def get_update_interval():
    """Load update interval from settings or env (seconds)."""
    default_interval = int(os.getenv("HALO_UPDATE_INTERVAL", "120"))

    settings = load_runtime_settings()
    chosen = settings.get("update_interval", default_interval)
    try:
        chosen_int = int(chosen)
    except Exception:
        chosen_int = default_interval

    return chosen_int


TEXT_COLUMNS = {
    "player_gamertag",
    "player_xuid",
    "match_id",
    "playlist",
    "playlist_id",
    "map",
    "game_type",
    "outcome",
    "raw_json",
}

EXTRA_COLUMNS = [
    # Stores the un-normalized match payload (API-derived fields) as JSON text
    "raw_json",
    # Best-effort timestamp for when the row was scraped
    "scraped_at",
]


def all_db_columns() -> list[str]:
    # Keep stable ordering: FINAL_COLUMNS first, then any extras.
    cols = list(FINAL_COLUMNS)
    for col in EXTRA_COLUMNS:
        if col not in cols:
            cols.append(col)
    return cols


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_sql_type(col: str) -> str:
    # Preserve the current data model behavior:
    # - TEXT_COLUMNS stay as text
    # - date/scraped_at are timestamps
    # - everything else is stored as DOUBLE PRECISION (pandas will coerce)
    if col in ("date", "scraped_at"):
        return "TIMESTAMPTZ"
    if col in TEXT_COLUMNS:
        return "TEXT"
    return "DOUBLE PRECISION"


def ensure_schema(engine) -> None:
    """Create the halo_match_stats table with a stable, explicit schema.

    This avoids schema drift from pandas' inferred types and also ensures
    that columns with special characters (e.g. dmg/ka) are consistently
    present across deployments.
    """

    cols = all_db_columns()
    col_defs = [f"{_quote_ident(col)} {_column_sql_type(col)}" for col in cols]

    ddl = "CREATE TABLE IF NOT EXISTS halo_match_stats (\n  " + ",\n  ".join(col_defs) + "\n);"
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # If the table already existed (e.g. older deployments), add any missing columns.
    try:
        existing_cols = {c.get("name") for c in inspect(engine).get_columns("halo_match_stats")}
        desired_cols = set(cols)
        missing = [c for c in cols if c not in existing_cols]
        if missing:
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(
                        text(
                            f"ALTER TABLE halo_match_stats ADD COLUMN IF NOT EXISTS {_quote_ident(col)} {_column_sql_type(col)}"
                        )
                    )
    except Exception as exc:
        print(f"Warning: failed to reconcile halo_match_stats columns: {exc}")

    # Ensure indexes + KV schema after table exists.
    ensure_indexes(engine)
    ensure_kv_schema(engine)


def ensure_kv_schema(engine) -> None:
    """Create tables for auto-discovered stats.

    - halo_match_stats_kv: one row per (player_xuid, match_id, key)
    - halo_stat_keys: registry of all keys ever observed

    This lets the scraper store new/unknown stats without code changes.
    """

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS halo_match_stats_kv (
                    player_xuid TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json JSONB,
                    value_text TEXT,
                    value_num DOUBLE PRECISION,
                    value_type TEXT,
                    scraped_at TIMESTAMPTZ,
                    PRIMARY KEY (player_xuid, match_id, key)
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS halo_stat_keys (
                    key TEXT PRIMARY KEY,
                    first_seen TIMESTAMPTZ,
                    last_seen TIMESTAMPTZ,
                    inferred_type TEXT,
                    example_json JSONB
                );
                """
            )
        )

    ensure_kv_indexes(engine)


def ensure_kv_indexes(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_halo_match_stats_kv_key ON halo_match_stats_kv (key)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_halo_match_stats_kv_match ON halo_match_stats_kv (match_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_halo_match_stats_kv_player ON halo_match_stats_kv (player_xuid)"
            )
        )


def _infer_value_type(value) -> tuple[str, float | None, str | None, str | None]:
    """Return (value_type, value_num, value_text, value_json_str)."""

    if value is None:
        return "null", None, None, "null"

    # Normalize bool before int/float checks.
    if isinstance(value, bool):
        return "bool", 1.0 if value else 0.0, "true" if value else "false", json.dumps(value)

    if isinstance(value, (int, float)):
        # Keep as numeric + json
        return "number", float(value), None, json.dumps(value)

    if isinstance(value, str):
        # Some API fields might be huge; store text and json.
        return "text", None, value, json.dumps(value)

    # Lists/dicts/objects: store as json only.
    try:
        return "json", None, None, json.dumps(value, default=str)
    except Exception:
        # Last resort
        return "text", None, str(value), json.dumps(str(value))


def write_extra_stats_to_kv(
    engine,
    player_xuid: str,
    match_id: str,
    payload: dict,
    scraped_at_iso: str | None = None,
) -> None:
    """Persist any fields not present in the stable wide schema into KV tables."""

    if not payload:
        return

    known_cols = set(all_db_columns())
    extras = {k: v for k, v in payload.items() if k not in known_cols}
    if not extras:
        return

    scraped_at = scraped_at_iso or datetime.now(dt_timezone.utc).isoformat()

    rows = []
    key_rows = []
    for key, value in extras.items():
        value_type, value_num, value_text, value_json_str = _infer_value_type(value)
        rows.append(
            {
                "player_xuid": str(player_xuid),
                "match_id": str(match_id),
                "key": str(key),
                "value_json": value_json_str,
                "value_text": value_text,
                "value_num": value_num,
                "value_type": value_type,
                "scraped_at": scraped_at,
            }
        )
        key_rows.append(
            {
                "key": str(key),
                "ts": scraped_at,
                "inferred_type": value_type,
                "example_json": value_json_str,
            }
        )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO halo_match_stats_kv (
                    player_xuid, match_id, key,
                    value_json, value_text, value_num, value_type, scraped_at
                )
                VALUES (
                    :player_xuid, :match_id, :key,
                    (:value_json)::jsonb, :value_text, :value_num, :value_type, (:scraped_at)::timestamptz
                )
                ON CONFLICT (player_xuid, match_id, key)
                DO UPDATE SET
                    value_json = EXCLUDED.value_json,
                    value_text = EXCLUDED.value_text,
                    value_num = EXCLUDED.value_num,
                    value_type = EXCLUDED.value_type,
                    scraped_at = EXCLUDED.scraped_at
                """
            ),
            rows,
        )

        conn.execute(
            text(
                """
                INSERT INTO halo_stat_keys (key, first_seen, last_seen, inferred_type, example_json)
                VALUES (:key, (:ts)::timestamptz, (:ts)::timestamptz, :inferred_type, (:example_json)::jsonb)
                ON CONFLICT (key)
                DO UPDATE SET
                    last_seen = EXCLUDED.last_seen
                """
            ),
            key_rows,
        )

# Load players to track from environment variable
def load_players_from_env():
    """Load players from environment variable HALO_TRACKED_PLAYERS.
    
    Format: JSON array of objects with gamertag and xuid
    Example: '[{"gamertag": "PlayerOne", "xuid": "123456"}, {"gamertag": "PlayerTwo", "xuid": "789012"}]'
    
    Falls back to default players if env var is not set.
    """
    default_players = [
        {"gamertag": "l 0cty l", "xuid": "2533274818160056"},
        {"gamertag": "Zaidster7", "xuid": "2533274965035069"},
        {"gamertag": "l P1N1 l", "xuid": "2533274804338345"},
        {"gamertag": "l Viper18 l", "xuid": "2535430400255009"},
        {"gamertag": "l Jordo l", "xuid": "2533274797008163"},
    ]
    
    players_json = os.getenv("HALO_TRACKED_PLAYERS", "").strip()
    
    if not players_json:
        print(f"✅ No HALO_TRACKED_PLAYERS set. Using {len(default_players)} default players.")
        return default_players
    
    try:
        players = json.loads(players_json)
        if not isinstance(players, list):
            print("⚠️ HALO_TRACKED_PLAYERS must be a JSON array. Using defaults.")
            return default_players
        
        # Validate each player has required fields
        for player in players:
            if not isinstance(player, dict) or 'gamertag' not in player or 'xuid' not in player:
                print("⚠️ Invalid player format. Each player needs 'gamertag' and 'xuid'. Using defaults.")
                return default_players
        
        print(f"✅ Loaded {len(players)} players from HALO_TRACKED_PLAYERS")
        for player in players:
            print(f"   - {player['gamertag']} ({player['xuid']})")
        return players
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse HALO_TRACKED_PLAYERS: {e}. Using defaults.")
        return default_players


# Define the players to track
PLAYERS = load_players_from_env()

FINAL_COLUMNS = [
    'player_gamertag', 'player_xuid', 'match_id', 'date', 'duration', 'game_type', 'map', 'playlist','playlist_id', 'outcome', 'team_id', 'team_rank', 'kills', 'deaths', 'assists', 'kda', 'accuracy','score', 'medal_count', 'dmg/ka', 'dmg/death', 'dmg/min', 'dmg_difference', 'pre_match_csr', 'post_match_csr', 'medal_360','medal_Achilles_Spine','medal_Always_Rotating','medal_Back_Smack','medal_Ballista','medal_Bank_Shot','medal_Blind_Fire','medal_Bodyguard','medal_Bomber','medal_Boom_Block','medal_Boxer','medal_Breacher','medal_Bulltrue','medal_Call_Blocked','medal_Chain_Reaction','medal_Clear_Reception','medal_Clock_Stop','medal_Cluster_Luck','medal_Combat_Evolved','medal_Counter_snipe','medal_Deadly_Catch','medal_Double_Kill','medal_Extermination','medal_Fastball','medal_Flag_Joust','medal_Flawless_Victory','medal_From_the_Grave','medal_Fumble','medal_Goal_Line_Stand','medal_Grenadier','medal_Guardian_Angel','medal_Gunslinger','medal_Hail_Mary','medal_Hang_Up','medal_Harpoon','medal_Hill_Guardian','medal_Hold_This','medal_Interlinked','medal_Killing_Frenzy','medal_Killing_Spree','medal_Killjoy','medal_Killtacular','medal_Killtrocity','medal_Last_Shot','medal_Marksman','medal_Mind_the_Gap','medal_Nade_Shot','medal_Ninja','medal_No_Scope','medal_Odin_s_Raven','medal_Off_the_Rack','medal_Overkill','medal_Pancake','medal_Perfect','medal_Pull','medal_Quick_Draw','medal_Quigley','medal_Remote_Detonation','medal_Return_to_Sender','medal_Reversal','medal_Rifleman','medal_Scattergunner','medal_Secure_Line','medal_Sharpshooter','medal_Shot_Caller','medal_Signal_Block','medal_Sneak_King','medal_Snipe','medal_Special_Delivery','medal_Spotter','medal_Steaktacular','medal_Stick','medal_Stopped_Short','medal_Straight_Balling','medal_Treasure_Hunter','medal_Triple_Kill','medal_Warrior','medal_Whiplash','medal_Wingman','medal_Yard_Sale','medal_id_1024030246','medal_id_1032565232','medal_id_1117301492','medal_id_1267013266','medal_id_152718958','medal_id_1552628741','medal_id_1825517751','medal_id_204144695','medal_id_22113181','medal_id_2387185397','medal_id_2408971842','medal_id_249491819','medal_id_3002710045','medal_id_316828380','medal_id_340198991','medal_id_3507884073','medal_id_4130011565','medal_id_4247243561','medal_id_454168309','medal_id_555570945','medal_id_601966503','medal_id_638246808','medal_id_709346128','medal_id_746397417','medal_id_911992497','all_time_max_csr_initial_measurement_matches','all_time_max_csr_measurement_matches_remaining','all_time_max_csr_next_sub_tier','all_time_max_csr_next_tier','all_time_max_csr_next_tier_start','all_time_max_csr_sub_tier','all_time_max_csr_tier','all_time_max_csr_tier_start','all_time_max_csr_value','current_csr_initial_measurement_matches','current_csr_measurement_matches_remaining','current_csr_next_sub_tier','current_csr_next_tier','current_csr_next_tier_start','current_csr_sub_tier','current_csr_tier','current_csr_tier_start','current_csr_value','season_max_csr_initial_measurement_matches','season_max_csr_measurement_matches_remaining','season_max_csr_next_sub_tier','season_max_csr_next_tier','season_max_csr_next_tier_start','season_max_csr_sub_tier','season_max_csr_tier','season_max_csr_tier_start','season_max_csr_value','average_life_duration','betrayals','callout_assists','capture_the_flag_stats_flag_capture_assists','capture_the_flag_stats_flag_captures','capture_the_flag_stats_flag_carriers_killed','capture_the_flag_stats_flag_grabs','capture_the_flag_stats_flag_returners_killed','capture_the_flag_stats_flag_returns','capture_the_flag_stats_flag_secures','capture_the_flag_stats_flag_steals','capture_the_flag_stats_kills_as_flag_carrier','capture_the_flag_stats_kills_as_flag_returner','capture_the_flag_stats_time_as_flag_carrier','damage_dealt','damage_taken','driver_assists','emp_assists','extraction_stats_extraction_conversions_completed','extraction_stats_extraction_conversions_denied','extraction_stats_extraction_initiations_completed','extraction_stats_extraction_initiations_denied','extraction_stats_successful_extractions','grenade_kills','headshot_kills','hijacks','max_killing_spree','melee_kills','objectives_completed','oddball_stats_kills_as_skull_carrier','oddball_stats_longest_time_as_skull_carrier','oddball_stats_skull_carriers_killed','oddball_stats_skull_grabs','oddball_stats_skull_scoring_ticks','oddball_stats_time_as_skull_carrier','personal_score','power_weapon_kills','pvp_stats_assists','pvp_stats_deaths','pvp_stats_kda','pvp_stats_kills','rounds_lost','rounds_tied','rounds_won','shots_fired','shots_hit','spawns','suicides','vehicle_destroys','zones_stats_stronghold_captures','zones_stats_stronghold_defensive_kills','zones_stats_stronghold_occupation_time','zones_stats_stronghold_offensive_kills','zones_stats_stronghold_scoring_ticks','zones_stats_stronghold_secures','team_accuracy','team_assists','team_average_life_duration','team_betrayals','team_callout_assists','team_capture_the_flag_stats_flag_capture_assists','team_capture_the_flag_stats_flag_captures','team_capture_the_flag_stats_flag_carriers_killed','team_capture_the_flag_stats_flag_grabs','team_capture_the_flag_stats_flag_returners_killed','team_capture_the_flag_stats_flag_returns','team_capture_the_flag_stats_flag_secures','team_capture_the_flag_stats_flag_steals','team_capture_the_flag_stats_kills_as_flag_carrier','team_capture_the_flag_stats_kills_as_flag_returner','team_capture_the_flag_stats_time_as_flag_carrier','team_damage_dealt','team_damage_taken','team_deaths','team_driver_assists','team_emp_assists','team_extraction_stats_extraction_conversions_completed','team_extraction_stats_extraction_conversions_denied','team_extraction_stats_extraction_initiations_completed','team_extraction_stats_extraction_initiations_denied','team_extraction_stats_successful_extractions','team_grenade_kills','team_headshot_kills','team_hijacks','team_id','team_kda','team_kills','team_max_killing_spree','team_medal_count','team_medals','team_melee_kills','team_objectives_completed','team_oddball_stats_kills_as_skull_carrier','team_oddball_stats_longest_time_as_skull_carrier','team_oddball_stats_skull_carriers_killed','team_oddball_stats_skull_grabs','team_oddball_stats_skull_scoring_ticks','team_oddball_stats_time_as_skull_carrier','team_personal_score','team_power_weapon_kills','team_pvp_stats_assists','team_pvp_stats_deaths','team_pvp_stats_kda','team_pvp_stats_kills','team_rank','team_rounds_lost','team_rounds_tied','team_rounds_won','team_score','team_shots_fired','team_shots_hit','team_spawns','team_suicides','team_vehicle_destroys','team_zones_stats_stronghold_captures','team_zones_stats_stronghold_defensive_kills','team_zones_stats_stronghold_occupation_time','team_zones_stats_stronghold_offensive_kills','team_zones_stats_stronghold_scoring_ticks','team_zones_stats_stronghold_secures','enemy_team_accuracy','enemy_team_assists','enemy_team_average_life_duration','enemy_team_betrayals','enemy_team_callout_assists','enemy_team_capture_the_flag_stats_flag_capture_assists','enemy_team_capture_the_flag_stats_flag_captures','enemy_team_capture_the_flag_stats_flag_carriers_killed','enemy_team_capture_the_flag_stats_flag_grabs','enemy_team_capture_the_flag_stats_flag_returners_killed','enemy_team_capture_the_flag_stats_flag_returns','enemy_team_capture_the_flag_stats_flag_secures','enemy_team_capture_the_flag_stats_flag_steals','enemy_team_capture_the_flag_stats_kills_as_flag_carrier','enemy_team_capture_the_flag_stats_kills_as_flag_returner','enemy_team_capture_the_flag_stats_time_as_flag_carrier','enemy_team_damage_dealt','enemy_team_damage_taken','enemy_team_deaths','enemy_team_driver_assists','enemy_team_emp_assists','enemy_team_extraction_stats_extraction_conversions_completed','enemy_team_extraction_stats_extraction_conversions_denied','enemy_team_extraction_stats_extraction_initiations_completed','enemy_team_extraction_stats_extraction_initiations_denied','enemy_team_extraction_stats_successful_extractions','enemy_team_grenade_kills','enemy_team_headshot_kills','enemy_team_hijacks','enemy_team_kda','enemy_team_kills','enemy_team_max_killing_spree','enemy_team_medal_count','enemy_team_medals','enemy_team_melee_kills','enemy_team_objectives_completed','enemy_team_oddball_stats_kills_as_skull_carrier','enemy_team_oddball_stats_longest_time_as_skull_carrier','enemy_team_oddball_stats_skull_carriers_killed','enemy_team_oddball_stats_skull_grabs','enemy_team_oddball_stats_skull_scoring_ticks','enemy_team_oddball_stats_time_as_skull_carrier','enemy_team_personal_score','enemy_team_power_weapon_kills','enemy_team_pvp_stats_assists','enemy_team_pvp_stats_deaths','enemy_team_pvp_stats_kda','enemy_team_pvp_stats_kills','enemy_team_rounds_lost','enemy_team_rounds_tied','enemy_team_rounds_won','enemy_team_score','enemy_team_shots_fired','enemy_team_shots_hit','enemy_team_spawns','enemy_team_suicides','enemy_team_vehicle_destroys','enemy_team_zones_stats_stronghold_captures','enemy_team_zones_stats_stronghold_defensive_kills','enemy_team_zones_stats_stronghold_occupation_time','enemy_team_zones_stats_stronghold_offensive_kills','enemy_team_zones_stats_stronghold_scoring_ticks','enemy_team_zones_stats_stronghold_secures'
]

TIME_COLUMNS = [
    'duration',
    'average_life_duration',
    'capture_the_flag_stats_time_as_flag_carrier',
    'oddball_stats_time_as_skull_carrier',
    'oddball_stats_longest_time_as_skull_carrier',
    'team_average_life_duration',
    'team_capture_the_flag_stats_time_as_flag_carrier',
    'team_oddball_stats_time_as_skull_carrier',
    'team_oddball_stats_longest_time_as_skull_carrier',
    'enemy_team_average_life_duration',
    'enemy_team_capture_the_flag_stats_time_as_flag_carrier',
    'enemy_team_oddball_stats_time_as_skull_carrier',
    'enemy_team_oddball_stats_longest_time_as_skull_carrier',
    'zones_stats_stronghold_occupation_time',
    'team_zones_stats_stronghold_occupation_time',
    'enemy_team_zones_stats_stronghold_occupation_time'
]

# Outcome mapping
OUTCOMES = {0: "Left", 1: "Tie", 2: "Win", 3: "Loss", 4: "Dnf"}

# Cache for medal metadata
medal_cache = {}
        
def clean_xuid(xuid):
    """Clean XUID format"""
    if isinstance(xuid, str) and "xuid(" in xuid:
        return xuid.replace("xuid(", "").replace(")", "")
    return str(xuid)

def load_tokens():
    """Load authentication tokens from file"""
    with open(TOKENS_PATH, 'r') as f:
        return json.load(f)

def get_or_default(obj, *attrs, default=None):
    """Safely navigate nested objects"""
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            if obj is None:
                return default
        else:
            return default
    return obj

async def get_rank_recap_csr_change(client, match_id, xuid):
    """Fetch pre-match and post-match CSR (if available) for a specific player and match."""
    try:
        response = await client.skill.get_match_skill(match_id=match_id, xuids=[xuid])
        data = await response.parse()

        for entry in data.value:
            if clean_xuid(entry.id) != xuid:
                continue

            result = getattr(entry, 'result', None)
            if not result or not hasattr(result, 'rank_recap'):
                # 🟡 No rank recap available (normal in many matches)
                return None, None

            recap = result.rank_recap
            pre = getattr(recap, 'pre_match_csr', None)
            post = getattr(recap, 'post_match_csr', None)

            pre_value = getattr(pre, 'value', None) if pre else None
            post_value = getattr(post, 'value', None) if post else None

            return pre_value, post_value

    except Exception as e:
        print(f"❌ Error fetching CSR rank recap for match {match_id}: {e}")

    return None, None

def normalize_row(row, all_fields):
    """
    Ensure all expected fields exist in the row with appropriate default values.
    Numeric fields get 0, text fields get ''.
    """
    normalized = {}
    
    # If row is None, create an empty dictionary
    if row is None:
        row = {}
    
    for field in all_fields:
        clean_field = field.strip()  # Remove any whitespace
        if clean_field in row:
            # Convert empty strings to appropriate defaults
            if row[clean_field] == '':
                if (
                    clean_field.startswith(('medal_', 'time_', 'damage_', 'team_', 'enemy_team_', 'capture_', 'extraction_', 'oddball_', 'zones_')) or
                    any(clean_field.endswith(suffix) for suffix in (
                        '_count', '_kills', '_score', '_value', '_ticks', '_deaths', '_assists',
                        '_starts', '_captures', '_grabs', '_returns', '_carrier', '_denied', '_completed', '_steals', '_spawns',
                        '_shots', '_suicides', '_rank', '_duration'
                    )) or
                    clean_field in ('kills', 'deaths', 'assists', 'kd', 'kda', 'accuracy', 'score', 'team_rank', 'betrayals')
                ):
                    normalized[clean_field] = 0
                else:
                    normalized[clean_field] = ''
            else:
                normalized[clean_field] = row[clean_field]
        else:
            # Determine if this should be a numeric field based on prefix/suffix
            if (
                clean_field.startswith(('medal_', 'time_', 'damage_', 'team_', 'enemy_team_', 'capture_', 'extraction_', 'oddball_', 'zones_')) or
                any(clean_field.endswith(suffix) for suffix in (
                    '_count', '_kills', '_score', '_value', '_ticks', '_deaths', '_assists',
                    '_starts', '_captures', '_grabs', '_returns', '_carrier', '_denied', '_completed', '_steals', '_spawns',
                    '_shots', '_suicides', '_rank', '_duration'
                )) or
                clean_field in ('kills', 'deaths', 'assists', 'kd', 'kda', 'accuracy', 'score', 'team_rank', 'betrayals')
            ):
                normalized[clean_field] = 0
            else:
                normalized[clean_field] = ''
    return normalized

def parse_duration_to_seconds(duration_str):
    try:
        if not isinstance(duration_str, str):
            duration_str = str(duration_str)

        if duration_str.count(':') == 2:
            h, m, s = duration_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif duration_str.count(':') == 1:
            m, s = duration_str.split(':')
            return int(m) * 60 + float(s)
    except Exception as e:
        print(f"Error parsing duration '{duration_str}': {e}")
    return 0

def get_engine():
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_url, pool_pre_ping=True)

def table_exists(engine) -> bool:
    return inspect(engine).has_table("halo_match_stats")

def dedupe_columns(cols):
    counter = {}
    new_cols = []
    for col in cols:
        count = counter.get(col, 0)
        new_col = f"{col}.{count}" if count > 0 else col
        new_cols.append(new_col)
        counter[col] = count + 1
    return new_cols

def normalize_columns_for_db(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = dedupe_columns(df.columns)
    return df

def prepare_results_dataframe(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Debug: Check what we have before conversion
    damage_cols = [col for col in df.columns if 'damage' in col.lower()]
    if damage_cols:
        print(f"🔍 Before conversion - damage columns: {damage_cols}")
        for col in damage_cols:
            sample_values = df[col].head(3).tolist()
            print(f"   {col}: {sample_values}")

    df = convert_time_columns_to_seconds(df)
    df = normalize_columns_for_db(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in df.columns:
        if col in TEXT_COLUMNS:
            df[col] = df[col].astype(str)
        elif col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Debug: Check what happened after conversion
    if damage_cols:
        print(f"🔍 After conversion - damage columns:")
        for col in damage_cols:
            if col in df.columns:
                sample_values = df[col].head(3).tolist()
                null_count = df[col].isna().sum()
                print(f"   {col}: {sample_values} (NULLs: {null_count})")

    return df

def delete_existing_matches(engine, df: pd.DataFrame) -> None:
    if df.empty or "player_xuid" not in df.columns or "match_id" not in df.columns:
        return

    pairs = df[["player_xuid", "match_id"]].dropna().drop_duplicates()
    if pairs.empty:
        return

    with engine.begin() as conn:
        for player_xuid, group in pairs.groupby("player_xuid"):
            match_ids = group["match_id"].tolist()
            conn.execute(
                text(
                    "DELETE FROM halo_match_stats "
                    "WHERE player_xuid = :xuid AND match_id = ANY(:match_ids)"
                ),
                {"xuid": player_xuid, "match_ids": match_ids},
            )

def ensure_indexes(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_halo_match_stats_player "
                "ON halo_match_stats (player_xuid)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_halo_match_stats_date "
                "ON halo_match_stats (date)"
            )
        )
        try:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_halo_match_stats_unique "
                    "ON halo_match_stats (player_xuid, match_id)"
                )
            )
        except Exception as exc:
            print(f"Warning: unique index not created: {exc}")


def write_single_result_to_db(match_row: dict, engine) -> bool:
    """Write a single normalized match row to the DB immediately.

    This keeps the website up-to-date during long scrape runs.
    We delete any existing (player_xuid, match_id) row first to avoid
    unique-index conflicts and allow refresh/re-scrapes.
    """

    if not match_row:
        return False

    df = prepare_results_dataframe([match_row])
    if df.empty:
        return False

    try:
        if not table_exists(engine):
            ensure_schema(engine)
        created = False
        # Table should already exist via ensure_schema; keep 'created' for compatibility.
        created = False

        delete_existing_matches(engine, df)
        df.to_sql(
            "halo_match_stats",
            engine,
            if_exists="append",
            index=False,
            chunksize=1,
            method="multi",
        )

        if created:
            ensure_indexes(engine)

        return True
    except Exception as exc:
        print(f"Warning: failed to write single match to DB: {exc}")
        return False

def write_results_to_db(results: list, engine) -> int:
    df = prepare_results_dataframe(results)
    if df.empty:
        return 0

    try:
        if not table_exists(engine):
            ensure_schema(engine)

        delete_existing_matches(engine, df)
        df.to_sql(
            "halo_match_stats",
            engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )
        ensure_indexes(engine)
        return len(df)
    except Exception as exc:
        print(f"Warning: failed to write results to DB: {exc}")
        return 0

def get_existing_match_ids(engine, player_xuid: str) -> set:
    try:
        if not table_exists(engine):
            # Make sure the table exists with the stable schema so future
            # calls don't repeatedly attempt to auto-create it.
            ensure_schema(engine)
        if not table_exists(engine):
            return set()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT match_id FROM halo_match_stats WHERE player_xuid = :xuid"
                ),
                {"xuid": player_xuid},
            ).fetchall()
        # Ensure match IDs are strings for comparison
        match_ids = {str(row[0]) for row in rows}
        print(f"📊 Found {len(match_ids)} existing matches for xuid {player_xuid}")
        return match_ids
    except Exception as exc:
        print(f"Warning: failed to load existing match ids for {player_xuid}: {exc}")
        return set()

def trim_player_history(engine, player_xuid: str, limit: int) -> None:
    """DISABLED - This function was incorrectly deleting historical data.
    
    match_limit should only control how many matches to FETCH per run,
    not how many to KEEP in the database. Keeping all historical data.
    """
    # DO NOT DELETE HISTORICAL DATA
    pass


def write_update_status(inserted_rows: int) -> None:
    update_status["new_rows_added"] = inserted_rows > 0
    update_status["new_row_count"] = inserted_rows
    update_status["last_update"] = datetime.now(dt_timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(UPDATE_STATUS_PATH, "w") as f:
            json.dump(update_status, f, indent=2)
            print("Wrote update_status.json")
    except Exception as e:
        print(f"Failed to write update_status.json: {e}")


async def get_medal_metadata(client):
    """Fetch medal metadata to get proper names"""
    global medal_cache
    if medal_cache:
        return medal_cache
    
    try:
        metadata_response = await client.gamecms_hacs.get_medal_metadata()
        metadata = await metadata_response.parse()
        
        if hasattr(metadata, 'medals'):
            for medal in metadata.medals:
                if hasattr(medal, 'name_id') and hasattr(medal, 'name'):
                    medal_id = medal.name_id
                    medal_name = medal.name.value if hasattr(medal.name, 'value') else str(medal.name)
                    medal_cache[medal_id] = medal_name
                    medal_cache[str(medal_id)] = medal_name
    except Exception as e:
        print(f"Error fetching medal metadata: {e}")
    
    return medal_cache

async def fetch_metadata(client, match_id, asset_id, version_id, cache, fetch_func):
    """Generic function to fetch and cache metadata"""
    key = f"{asset_id}:{version_id}"
    if key in cache:
        return cache[key]
    
    try:
        response = await fetch_func(asset_id, version_id)
        data = await response.parse()
        
        # Try common name attributes
        for attr in ['name', 'asset_name', 'internal_name', 'display_name', 'public_name', 'title']:
            if hasattr(data, attr):
                name = getattr(data, attr)
                cache[key] = name
                return name
                
        # If properties exist, try those too
        properties = get_or_default(data, 'properties')
        if properties:
            for attr in ['name', 'display_name', 'game_mode', 'variant_name']:
                if hasattr(properties, attr):
                    name = getattr(properties, attr)
                    cache[key] = name
                    return name
    except Exception as e:
        print(f"Error fetching metadata for {match_id}, {asset_id}: {e}")
    
    return f"ID: {asset_id}"

def process_medals(medals_data, match_data, medal_names):
    """Process medal data into the match record with defensive coding"""
    if not medals_data:
        return
    
    try:
        # Ensure medals_data is iterable
        if not hasattr(medals_data, '__iter__'):
            print(f"Warning: medals_data is not iterable: {type(medals_data)}")
            return
            
        for medal in medals_data:
            # Defensive check for medal object
            if medal is None:
                continue
                
            medal_id = get_or_default(medal, 'name_id')
            medal_count = get_or_default(medal, 'count', default=0)
            
            if not medal_id:
                continue
                
            # Use medal name if available, otherwise use ID
            if medal_names and str(medal_id) in medal_names:
                medal_name = medal_names[str(medal_id)]
                # Clean the name for use as a column name
                clean_name = ''.join(c if c.isalnum() else '_' for c in medal_name)
                column_name = f"medal_{clean_name}"
            else:
                column_name = f"medal_id_{medal_id}"
                
            match_data[column_name] = medal_count
    except Exception as e:
        print(f"Error processing medals: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace

async def process_match(
    client,
    player_info,
    match_id,
    match_number,
    results,
    medal_names,
    engine=None,
    inserted_counter: dict | None = None,
):
    """Process a single match for a player"""
    player_gamertag = player_info["gamertag"]
    player_xuid = clean_xuid(player_info["xuid"])

    print(f"Processing match ID: {match_id} for player {player_gamertag}")

    try:
        # Fetch match stats
        match_stats_response = await client.stats.get_match_stats(match_id)
        match_stats = await match_stats_response.parse()

        if not match_stats or not getattr(match_stats, "players", None):
            print(f"⚠️ Match stats not yet available for {match_id}, skipping")
            return False

        match_date = match_stats.match_info.start_time
        match_duration = match_stats.match_info.duration

        # Metadata
        game_variant = get_or_default(match_stats.match_info, 'ugc_game_variant')
        game_type = await fetch_metadata(client, match_id, game_variant.asset_id, game_variant.version_id, {}, lambda a, v: client.discovery_ugc.get_ugc_game_variant(a, v)) if game_variant else "Unknown"

        map_variant = get_or_default(match_stats.match_info, 'map_variant')
        map_name = await fetch_metadata(client, match_id, map_variant.asset_id, map_variant.version_id, {}, lambda a, v: client.discovery_ugc.get_map(a, v)) if map_variant else "Unknown"

        playlist_obj = get_or_default(match_stats.match_info, 'playlist')
        playlist_id = get_or_default(playlist_obj, 'asset_id')
        version_id = get_or_default(playlist_obj, 'version_id')
        playlist = await fetch_metadata(client, match_id, playlist_id, version_id, {}, lambda a, v: client.discovery_ugc.get_playlist(a, v)) if playlist_id and version_id else "Unknown"

        # Find player in match
        for player in match_stats.players:
            current_xuid = clean_xuid(get_or_default(player, 'player_id'))
            if current_xuid != player_xuid:
                continue

            player_team_id = get_or_default(player, 'last_team_id', default=0)
            outcome = get_or_default(player, 'outcome', default="Unknown")
            readable_outcome = OUTCOMES.get(outcome, f"Unknown ({outcome})") if str(outcome).isdigit() else str(outcome)

            team_rank = 0
            friendly_team_stats = {}
            enemy_team_stats = {}

            for team in get_or_default(match_stats, 'teams', default=[]):
                team_id = get_or_default(team, 'team_id')
                is_player_team = (team_id == player_team_id)
                team_stats_dict = friendly_team_stats if is_player_team else enemy_team_stats
                team_prefix = 'team_' if is_player_team else 'enemy_team_'

                if hasattr(team, 'stats') and team.stats:
                    if hasattr(team.stats, 'core_stats'):
                        for stat_name, stat_value in vars(team.stats.core_stats).items():
                            if not stat_name.startswith('_'):
                                if stat_name == 'medals' and stat_value:
                                    team_stats_dict[f'{team_prefix}medal_count'] = len(stat_value)
                                else:
                                    team_stats_dict[f'{team_prefix}{stat_name}'] = stat_value
                    for category_name in vars(team.stats):
                        if category_name != 'core_stats':
                            cat_stats = getattr(team.stats, category_name, None)
                            if cat_stats:
                                for stat_name, stat_value in vars(cat_stats).items():
                                    team_stats_dict[f'{team_prefix}{category_name}_{stat_name}'] = stat_value or 0

            eastern = timezone('US/Eastern')
            local_dt = match_date.astimezone(eastern)

            match_data = {
                'player_gamertag': player_gamertag,
                'player_xuid': player_xuid,
                'match_id': match_id,
                'date': local_dt.replace(microsecond=0).isoformat(),
                'duration': str(match_duration),
                'game_type': game_type,
                'map': map_name,
                'playlist': playlist,
                'playlist_id': playlist_id or 'Unknown',
                'outcome': readable_outcome,
                'team_id': player_team_id,
                'team_rank': team_rank,
            }

            match_data.update(friendly_team_stats)
            match_data.update(enemy_team_stats)

            # 🆕 Fetch pre/post match CSR from rank recap
            pre_csr, post_csr = await get_rank_recap_csr_change(client, match_id, player_xuid)
            match_data['pre_match_csr'] = pre_csr if pre_csr is not None else 0
            match_data['post_match_csr'] = post_csr if post_csr is not None else 0
            # 🖨️ Debug print:
            print(f"🔎 {player_gamertag} Match {match_id}: pre_match_csr={pre_csr}, post_match_csr={post_csr}")

            # Fetch full playlist CSR data (current/season_max/all_time_max)
            try:
                default_ranked_playlist = "edfef3ac-9cbe-4fa2-b949-8f29deafd483"  # Default Ranked Arena playlist
                playlist_to_check = playlist_id if playlist_id else default_ranked_playlist

                playlist_csr_response = await client.skill.get_playlist_csr(
                    playlist_id=playlist_to_check,
                    xuids=[player_xuid]
                )

                if playlist_csr_response:
                    playlist_csr_data = await playlist_csr_response.parse()
                    if hasattr(playlist_csr_data, 'value') and playlist_csr_data.value:
                        for player_data in playlist_csr_data.value:
                            if clean_xuid(get_or_default(player_data, 'id')) == player_xuid:
                                result = get_or_default(player_data, 'result')
                                if result:
                                    for csr_type in ['current', 'season_max', 'all_time_max']:
                                        csr_obj = get_or_default(result, csr_type)
                                        if csr_obj and hasattr(csr_obj, '__dict__'):
                                            for k, v in vars(csr_obj).items():
                                                if not k.startswith('_'):
                                                    match_data[f'{csr_type}_csr_{k}'] = v

            except Exception as e:
                print(f"⚠️ Failed to get playlist CSR for {player_gamertag}: {e}")

            # 🧮 Player Stats
            player_team_stats = get_or_default(player, 'player_team_stats', default=[])
            if player_team_stats:
                stats = get_or_default(player_team_stats[0], 'stats')
                if stats:
                    core = get_or_default(stats, 'core_stats')
                    if core:
                        core_attrs = [k for k in vars(core).keys() if not k.startswith('_')]
                        print(f"🔍 {player_gamertag} {match_id}: Core stats attributes = {core_attrs}")
                        
                        for stat_name, stat_value in vars(core).items():
                            if stat_name == 'medals':
                                match_data['medal_count'] = len(stat_value)
                                process_medals(stat_value, match_data, medal_names)

                            elif stat_name == 'accuracy':
                                match_data['accuracy'] = stat_value
                            else:
                                match_data[stat_name] = stat_value

                    for category in vars(stats):
                        if category != 'core_stats':
                            cat_stats = getattr(stats, category, None)
                            if cat_stats:
                                for stat_name, stat_value in vars(cat_stats).items():
                                    match_data[f"{category}_{stat_name}"] = stat_value or 0

            # Store the full API-derived payload so we never lose fields.
            # (Calculated fields are added after this snapshot.)
            raw_payload = match_data.copy()
            
            # Debug: Check if damage values are in match_data
            damage_keys = [k for k in match_data.keys() if 'damage' in k.lower()]
            if damage_keys:
                print(f"✅ Found damage keys: {damage_keys}")
                for key in damage_keys:
                    print(f"   {key} = {match_data[key]}")
            else:
                print(f"❌ NO damage keys found! Available keys: {list(match_data.keys())}")
            
            match_data["raw_json"] = json.dumps(raw_payload, default=str)
            match_data["scraped_at"] = datetime.now(dt_timezone.utc).isoformat()

            match_data = add_calculated_fields(match_data)
            normalized_match_data = normalize_row(match_data, all_db_columns())
            if results is not None:
                results.append(normalized_match_data)

            if engine is not None:
                if write_single_result_to_db(normalized_match_data, engine):
                    try:
                        write_extra_stats_to_kv(
                            engine,
                            player_xuid=player_xuid,
                            match_id=match_id,
                            payload=raw_payload,
                            scraped_at_iso=match_data.get("scraped_at"),
                        )
                    except Exception as exc:
                        print(f"Warning: failed to persist extra stats for {player_gamertag} {match_id}: {exc}")
                    if inserted_counter is not None:
                        inserted_counter["count"] = int(inserted_counter.get("count", 0)) + 1
                        # Keep update_status.json fresh during long runs
                        write_update_status(int(inserted_counter["count"]))
            return True

        print(f"❌ Player {player_gamertag} not found in match {match_id}")
        return False

    except Exception as e:
        print(f"❌ Error processing match {match_id}: {e}")
        return False

async def process_player(
    client,
    player_info,
    results,
    medal_names,
    max_matches=None,
    force_refresh=False,
    existing_match_ids=None,
    engine=None,
    inserted_counter: dict | None = None,
):
    player_gamertag = player_info["gamertag"]
    player_xuid = clean_xuid(player_info["xuid"])
    existing_match_ids = existing_match_ids or set()

    total_seen = 0
    start = 0
    page_size = 25 

    print(f"Fetching up to {max_matches or 'all'} matches for {player_gamertag}")

    force_refresh_latched = bool(force_refresh)

    while True:
        # --- RETRY LOGIC FOR MATCH HISTORY ---
        retries = 0
        max_retries = 5
        match_history = None

        while retries < max_retries:
            try:
                history_response = await client.stats.get_match_history(
                    player=player_xuid,
                    start=start,
                    count=page_size,
                    match_type='all'
                )
                match_history = await history_response.parse()
                break # Success, exit retry loop
            except ClientResponseError as e:
                if e.status == 429:
                    wait_time = (2 ** retries) + random.uniform(0.5, 2.0)
                    print(f"⚠️ 429 Too Many Requests fetching history. Sleeping {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    print(f"❌ Error fetching history for {player_gamertag}: {e}")
                    raise e # Raise non-429 errors
        
        if match_history is None:
            print(f"❌ Failed to fetch match history for {player_gamertag} after {max_retries} retries. Skipping player.")
            return
        # -------------------------------------

        results_batch = match_history.results
        print(f"→ Fetched {len(results_batch)} matches starting at {start} for {player_gamertag}")

        if not results_batch:
            break

        skipped_count = 0
        processed_count = 0
        
        for match_result in results_batch:
            effective_limit = max_matches if max_matches is not None else get_match_limit()
            if effective_limit is not None and total_seen >= effective_limit:
                print(f"Reached limit of {effective_limit} total matches for {player_gamertag}")
                return

            match_id = match_result.match_id
            total_seen += 1
            
            # Ensure match_id is a string for comparison
            match_id_str = str(match_id)

            # Allow toggling True Refresh mid-run; once enabled, keep it on
            # for the remainder of this player's scrape.
            if not force_refresh_latched:
                force_refresh_latched = consume_force_refresh_setting() or get_force_refresh()

            if not force_refresh_latched and match_id_str in existing_match_ids:
                skipped_count += 1
                continue

            # PACING: Sleep slightly to prevent hammering the API during individual match processing
            await asyncio.sleep(0.5)

            try:
                await process_match(
                    client, player_info, match_id_str, total_seen,
                    results, medal_names, engine=engine, inserted_counter=inserted_counter
                )
                processed_count += 1
                # Avoid re-processing duplicates within this run.
                existing_match_ids.add(match_id_str)
            except ClientResponseError as e:
                if e.status == 429:
                    print(f"⚠️ 429 encountered inside process_match. Pausing 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print(f"Error in process_match: {e}")
        
        if skipped_count > 0:
            print(f"⏩ Skipped {skipped_count} existing matches, processed {processed_count} new matches")

        if len(results_batch) < page_size:
            break 

        start += page_size


def reorder_columns(columns):
    # Core identifiers
    core_fields = [
        'player_gamertag', 'player_xuid', 'match_id', 'date', 'duration',
        'game_type', 'map', 'playlist', 'playlist_id',
        'outcome', 'team_id', 'team_rank'
    ]

    # Performance
    perf_fields = ['kills', 'deaths', 'assists', 'kda', 'accuracy', 'score', 'medal_count']

    # Custom calculated columns — always include them
    calc_fields = ['dmg/ka', 'dmg/death', 'dmg/min', 'dmg_difference']

    # Medals
    named_medals = sorted([c for c in columns if c.startswith('medal_') and not c.startswith('medal_id_') and c != 'medal_count'])
    id_medals = sorted([c for c in columns if c.startswith('medal_id_')])

    # CSR fields
    csr_fields = sorted([c for c in columns if c.startswith(('current_csr_', 'season_max_csr_', 'all_time_max_csr_'))])

    # Team stats
    team_fields = sorted([c for c in columns if c.startswith('team_')])
    enemy_team_fields = sorted([c for c in columns if c.startswith('enemy_team_')])

    # Remaining = player stats and misc
    used = set(core_fields + perf_fields + calc_fields + named_medals + id_medals + csr_fields + team_fields + enemy_team_fields)
    misc_fields = sorted([c for c in columns if c not in used])

    # Final order
    ordered = (
        core_fields +
        perf_fields +
        calc_fields +  # Always include them here
        named_medals +
        id_medals +
        csr_fields +
        misc_fields +
        team_fields +
        enemy_team_fields
    )
    return ordered

def add_calculated_fields(row):
    """Add calculated fields to a match data row with robust error handling"""
    # Initialize calculated fields with default values
    calculated_fields = {
        "dmg/ka": 0,
        "dmg/death": 0,
        "dmg/min": 0,
        "dmg_difference": 0
    }
    
    # If row is None, return a dictionary with default calculated fields
    if row is None:
        return calculated_fields
        
    # Create a copy of the row to avoid modifying the original
    result = row.copy() if isinstance(row, dict) else {}
    
    try:
        # Use defensive conversion with fallbacks
        kills = float(row.get("kills", 0) or 0)
        assists = float(row.get("assists", 0) or 0)
        deaths = float(row.get("deaths", 0) or 0)
        damage_dealt = float(row.get("damage_dealt", 0) or 0)
        damage_taken = float(row.get("damage_taken", 0) or 0)
        duration_str = row.get("duration", "0:00.0")

        duration_seconds = parse_duration_to_seconds(duration_str)

        # Calculate
        ka = kills + assists
        if ka > 0:
            result["dmg/ka"] = round(damage_dealt / ka, 2)
        else:
            result["dmg/ka"] = 0
            
        if deaths > 0:
            result["dmg/death"] = round(damage_dealt / deaths, 2)
        else:
            result["dmg/death"] = 0
            
        if duration_seconds > 0:
            result["dmg/min"] = round(damage_dealt / (duration_seconds / 60), 2)
        else:
            result["dmg/min"] = 0
            
        result["dmg_difference"] = round(damage_dealt - damage_taken, 2)

    except Exception as e:
        print(f"Error calculating derived fields: {e}")
        # Add the default calculated fields if there was an error
        result.update(calculated_fields)

    # Ensure all calculated fields exist in the result
    for field in calculated_fields:
        if field not in result:
            result[field] = calculated_fields[field]
            
    return result

async def run_stats(max_matches=None, force_refresh=False):
    tokens = load_tokens()
    # We still keep a small in-memory list for debugging, but DB writes happen per match.
    results = []
    engine = get_engine()
    ensure_schema(engine)

    # If caller passes explicit overrides, they stay fixed for the whole run.
    # Otherwise, per-player/per-page logic will pick up Settings changes mid-run.
    force_refresh_override = bool(force_refresh)
    match_limit_override = max_matches if max_matches is not None else None

    # One-shot settings behavior: consume True Refresh once per run and
    # latch it so it applies for the whole run.
    force_refresh_once = consume_force_refresh_setting() if not force_refresh_override else False
    force_refresh_latched = bool(force_refresh_override or force_refresh_once)

    effective_limit_for_log = match_limit_override if match_limit_override is not None else get_match_limit()
    effective_force_refresh_for_log = force_refresh_latched or get_force_refresh()
    print(
        f"▶️ Starting stats run with match_limit={effective_limit_for_log} "
        f"force_refresh={effective_force_refresh_for_log}"
    )

    inserted_counter = {"count": 0}

    async with ClientSession() as session:
        client = HaloInfiniteClient(
            session=session,
            spartan_token=tokens["spartan_token"],
            clearance_token=tokens["clearance_token"]
        )

        medal_names = await get_medal_metadata(client)

        for player in PLAYERS:
            xuid = clean_xuid(player["xuid"])
            existing_match_ids = set()
            existing_match_ids = get_existing_match_ids(engine, xuid)
            await process_player(
                client,
                player,
                results,
                medal_names,
                max_matches=match_limit_override,
                force_refresh=force_refresh_latched,
                existing_match_ids=existing_match_ids,
                engine=engine,
                inserted_counter=inserted_counter,
            )

    inserted_rows = int(inserted_counter.get("count", 0))
    # Final index ensure (cheap if already exists)
    try:
        if table_exists(engine):
            ensure_indexes(engine)
    except Exception:
        pass
    for player in PLAYERS:
        effective_trim_limit = match_limit_override if match_limit_override is not None else get_match_limit()
        trim_player_history(engine, clean_xuid(player["xuid"]), effective_trim_limit)
    write_update_status(inserted_rows)

def convert_time_columns_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all time-based columns in TIME_COLUMNS to seconds.
    Handles formats like:
    - MM:SS.s
    - H:MM:SS
    - 0 days 00:00:25.400000 (pandas timedelta string)

    Returns the updated DataFrame.
    """

    def parse_time_to_seconds(time_str):
        if not isinstance(time_str, str):
            return 0.0
        try:
            time_str = time_str.strip()

            # Handle '0 days HH:MM:SS.microseconds'
            if "days" in time_str:
                time_part = time_str.split("days")[-1].strip()
                h, m, s = time_part.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)

            # Handle HH:MM:SS
            if time_str.count(':') == 2:
                h, m, s = time_str.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)

            # Handle MM:SS
            elif time_str.count(':') == 1:
                m, s = time_str.split(':')
                return float(m) * 60 + float(s)

            # Handle pure seconds
            elif time_str.replace('.', '', 1).isdigit():
                return float(time_str)

        except Exception as e:
            print(f"⚠️ Failed to convert time '{time_str}': {e}")
        return 0.0

    for col in TIME_COLUMNS:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).apply(parse_time_to_seconds)
                print(f"✅ Converted '{col}' to seconds.")
            except Exception as e:
                print(f"❌ Error converting column '{col}': {e}")
        else:
            print(f"⚠️ Column '{col}' not found. Skipping.")

    return df

if __name__ == "__main__":
    # Run with no max_matches limit to get all matches
    # asyncio.run(run_stats(max_matches=None))
    
    # Or run with a specific limit (e.g., 10000 matches per player)
    asyncio.run(run_stats(max_matches=None))
    
