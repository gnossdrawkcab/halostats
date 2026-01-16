# NOTE: This file was recovered from an earlier backup but appears to be incomplete.
# Many build_* functions are truncated or missing implementations.
# The file compiles but may have runtime issues until all functions are restored.

import html
import json
import os
import re
import time
from itertools import combinations
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, Response
from sqlalchemy import create_engine, inspect
from sqlalchemy import text
from halo_paths import data_path

# Setup Flask with correct template and static paths for Docker
APP_ROOT = Path(__file__).parent.parent  # Go up from /app/src to /app
TEMPLATE_DIR = APP_ROOT / 'templates'
STATIC_DIR = APP_ROOT / 'static'

APP_TITLE = os.getenv('HALO_SITE_TITLE', '👑 Scrim Kings')
TIMEZONE = os.getenv('HALO_TZ', 'US/Eastern')
SESSION_LIMIT_DEFAULT = int(os.getenv('HALO_SESSION_LIMIT', '50'))
LIFETIME_LIMIT_DEFAULT = int(os.getenv('HALO_LIFETIME_LIMIT_DEFAULT', '200'))
LIFETIME_LIMIT_MAX = int(os.getenv('HALO_LIFETIME_LIMIT_MAX', '2000'))
STATUS_PATH = data_path(os.getenv('HALO_STATUS_NAME', 'update_status.json'))
SETTINGS_PATH = data_path('settings.json')
INSIGHTS_CACHE_PATH = data_path(os.getenv('HALO_INSIGHTS_CACHE_NAME', 'insights_cache.json'))
STATIC_VERSION_OVERRIDE = os.getenv('HALO_STATIC_VERSION')
CACHE_TTL = int(os.getenv('HALO_CACHE_TTL', '120'))
DB_COUNT_TTL = int(os.getenv('HALO_DB_COUNT_TTL', '60'))
INSIGHTS_CACHE_TTL = int(os.getenv('HALO_INSIGHTS_CACHE_TTL', '300'))
INSIGHTS_CACHE_DISK_TTL = int(os.getenv('HALO_INSIGHTS_CACHE_DISK_TTL', '21600'))
LINEUP_MATCH_LIMIT = int(os.getenv('HALO_LINEUP_MATCH_LIMIT', '0'))
MAP_VETO_MIN_GAMES = int(os.getenv('HALO_MAP_VETO_MIN_GAMES', '50'))
NOTABLE_GAMES_LIMIT = int(os.getenv('HALO_NOTABLE_GAMES_LIMIT', '100'))
PLAYER_HOVER_CACHE_TTL = int(os.getenv('HALO_PLAYER_HOVER_TTL', '300'))
DB_NAME = os.getenv('HALO_DB_NAME', 'halostatsapi')
DB_USER = os.getenv('HALO_DB_USER', 'postgres')
DB_PASSWORD = os.getenv('HALO_DB_PASSWORD')
DB_HOST = os.getenv('HALO_DB_HOST', 'halostatsapi')
DB_PORT = os.getenv('HALO_DB_PORT', '5432')

NUMERIC_COLUMNS = ['kills', 'deaths', 'assists', 'kda', 'accuracy', 'score', 'dmg/ka', 'dmg/death', 'dmg/min', 'dmg_difference']
MATCH_COLUMNS = [
    # Core match info
    'match_id', 'date', 'player_gamertag', 'playlist', 'game_type', 'map', 'outcome',
    # Player core stats
    'kills', 'deaths', 'assists', 'kda', 'accuracy', 'score', 'personal_score',
    'duration', 'medal_count', 'average_life_duration',
    # Weapon/Damage stats
    'damage_dealt', 'damage_taken', 'shots_fired', 'shots_hit',
    'headshot_kills', 'melee_kills', 'grenade_kills', 'power_weapon_kills',
    'vehicle_destroys', 'hijacks',
    # Objective stats
    'objectives_completed', 'callout_assists', 'betrayals', 'suicides',
    'rounds_won', 'rounds_lost', 'rounds_tied',
    # Calculated stats
    'dmg/ka', 'dmg/death', 'dmg/min', 'dmg_difference',
    # CSR tracking
    'pre_match_csr', 'post_match_csr',
    # Game type specific
    'capture_the_flag_stats_flag_captures', 'capture_the_flag_stats_flag_returns',
    'oddball_stats_time_as_skull_carrier', 'zones_stats_stronghold_captures',
    'extraction_stats_successful_extractions',
    # Team stats
    'team_id', 'team_rank', 'team_damage_dealt', 'team_score', 'team_personal_score',
    'enemy_team_damage_dealt', 'enemy_team_score'
]
MAJOR_STAT_COLUMNS = [
    ('kills', 'Kills'), ('deaths', 'Deaths'), ('assists', 'Assists'), 
    ('kda', 'KDA'), ('accuracy', 'Accuracy'),
    ('damage_dealt', 'Damage Dealt'), ('damage_taken', 'Damage Taken'),
    ('dmg/ka', 'DMG/KA'), ('dmg/death', 'DMG/Death'), ('dmg/min', 'DMG/Min'),
    ('dmg_difference', 'Damage Diff'),
    ('shots_fired', 'Shots Fired'), ('shots_hit', 'Shots Hit'),
    ('medal_count', 'Medals'), ('personal_score', 'Personal Score'),
    ('callout_assists', 'Callouts'), 
    ('headshot_kills', 'Headshots'), ('melee_kills', 'Melee'), ('grenade_kills', 'Grenades'),
    ('power_weapon_kills', 'Power Weapons'),
    ('average_life_duration', 'Avg Life'), ('objectives_completed', 'Objectives'),
    ('betrayals', 'Betrayals'), ('suicides', 'Suicides'),
    ('pre_match_csr', 'Pre-CSR'), ('post_match_csr', 'Post-CSR'),
    ('vehicle_destroys', 'Vehicle Kills'), ('hijacks', 'Hijacks'),
    ('rounds_won', 'Rounds Won'), ('rounds_lost', 'Rounds Lost')
]
INDEX_DEFINITIONS = [('idx_halo_match_stats_playlist', 'playlist'), ('idx_halo_match_stats_outcome', 'outcome'), ('idx_halo_match_stats_date', 'date'), ('idx_halo_match_stats_player', 'player_gamertag'), ('idx_halo_match_stats_match', 'match_id')]
OBJECTIVE_PREFIXES = ('capture_the_flag_stats_', 'oddball_stats_', 'zones_stats_', 'extraction_stats_')
EXTRA_MATCH_COLUMNS = ['objectives_completed', 'betrayals', 'suicides']

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

STATIC_VERSION_PATHS = [
    STATIC_DIR / 'styles.css',
    STATIC_DIR / 'app.js'
]

PLAYER_COLORS = {
    'zaidster': 'player-zaidster',
    '0cty': 'player-octy',
    'octy': 'player-octy',
    'viper': 'player-viper',
    'jordo': 'player-jordo',
    'p1n1': 'player-pini',
    'pini': 'player-pini'
}


def get_player_class(player_name: str) -> str:
    """Return CSS class for player-specific coloring."""
    if not player_name:
        return ''
    name_lower = str(player_name).strip().lower()
    for key, css_class in PLAYER_COLORS.items():
        if key in name_lower:
            return css_class
    return ''


def get_static_version() -> str:
    if STATIC_VERSION_OVERRIDE:
        return STATIC_VERSION_OVERRIDE
    try:
        mtimes = []
        for path in STATIC_VERSION_PATHS:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        if mtimes:
            return str(int(max(mtimes)))
    except Exception:
        return '1'
    return '1'


@app.template_filter('player_class')
def player_class_filter(player_name):
    return get_player_class(player_name)


@app.context_processor
def utility_processor():
    return dict(get_player_class=get_player_class, static_version=get_static_version())


@app.context_processor
def hover_data_processor():
    df = cache.get()
    return dict(player_hover_data=build_player_hover_data(df))


def get_engine():
    db_url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    return create_engine(db_url, pool_pre_ping=True)


def quote_identifier(name: str) -> str:
    safe_name = str(name).replace('"', '""')
    return f'"{safe_name}"'


def select_match_columns(available: set[str]) -> list[str]:
    medal_cols = [col for col in available if str(col).startswith('medal_') or str(col).startswith('medal_id_')]
    objective_cols = [col for col in available if any(str(col).startswith(prefix) for prefix in OBJECTIVE_PREFIXES)]
    extra_cols = [col for col in EXTRA_MATCH_COLUMNS if col in available]
    ordered = []
    seen = set()
    for col in MATCH_COLUMNS + sorted(objective_cols) + sorted(medal_cols) + extra_cols:
        if col in available and col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered or sorted(available)


def ensure_indexes(engine) -> None:
    try:
        inspector = inspect(engine)
        if not inspector.has_table('halo_match_stats'):
            return
        columns = {c.get('name') for c in inspector.get_columns('halo_match_stats')}
        columns.discard(None)
        if not columns:
            return
        with engine.begin() as conn:
            for index_name, column_name in INDEX_DEFINITIONS:
                if column_name not in columns:
                    continue
                col_sql = quote_identifier(column_name)
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS {index_name} ON halo_match_stats ({col_sql})'))
    except Exception:
        return None


def load_dataframe(engine) -> pd.DataFrame:
    try:
        inspector = inspect(engine)
        if not inspector.has_table('halo_match_stats'):
            return pd.DataFrame()
        
        columns = {c.get('name') for c in inspector.get_columns('halo_match_stats')}
        columns.discard(None)
        
        # Exclude raw_json to save memory (it's huge and we have the parsed columns)
        exclude_cols = {'raw_json'}
        select_cols = [c for c in columns if c not in exclude_cols]
        select_sql = ', '.join(quote_identifier(col) for col in select_cols) if select_cols else '*'
        
        where = []
        if 'playlist' in columns:
            where.append("TRIM(playlist) = 'Ranked Arena'")
        if 'outcome' in columns:
            where.append("LOWER(outcome) <> 'dnf'")
        
        tie_conditions = []
        if 'kills' in columns:
            tie_conditions.append('COALESCE(kills, 0) <= 1')
        if 'duration' in columns:
            tie_conditions.append('COALESCE(duration, 0) < 120')
        if tie_conditions:
            tie_clause = ' OR '.join(tie_conditions)
            where.append(f"NOT (LOWER(outcome) = 'tie' AND ({tie_clause}))")
        
        query = f'SELECT {select_sql} FROM halo_match_stats'
        if where:
            query = f"{query} WHERE {' AND '.join(where)}"
        
        df = pd.read_sql_query(query, engine)
        return df
    except Exception:
        return pd.DataFrame()


def load_db_row_count(engine) -> int:
    try:
        if not inspect(engine).has_table('halo_match_stats'):
            return 0
        
        columns = {c.get('name') for c in inspect(engine).get_columns('halo_match_stats')}
        where = ["TRIM(playlist) = 'Ranked Arena'"] if 'playlist' in columns else []
        
        if 'outcome' in columns:
            where.append("LOWER(outcome) <> 'dnf'")
        
        tie_conditions = []
        if 'kills' in columns:
            tie_conditions.append('COALESCE(kills, 0) <= 1')
        if 'duration' in columns:
            tie_conditions.append('COALESCE(duration, 0) < 120')
        if tie_conditions:
            tie_clause = ' OR '.join(tie_conditions)
            where.append(f"NOT (LOWER(outcome) = 'tie' AND ({tie_clause}))")
        
        where_sql = ' AND '.join(where)
        
        with engine.connect() as conn:
            if where_sql:
                query = f'SELECT COUNT(*) FROM halo_match_stats WHERE {where_sql}'
            else:
                query = 'SELECT COUNT(*) FROM halo_match_stats'
            return int(conn.execute(text(query)).scalar() or 0)
    except Exception:
        return 0


class DataCache:
    """Smart cache that only reloads when new matches are added."""
    
    def __init__(self, engine) -> None:
        self.engine = engine
        self.df = pd.DataFrame()
        self.last_count = 0
        self.last_check = 0.0
    
    def get(self) -> pd.DataFrame:
        now = time.time()
        if now - self.last_check >= 30:
            current_count = load_db_row_count(self.engine)
            self.last_check = now
            if current_count != self.last_count or self.df.empty:
                self.df = normalize_df(load_dataframe(self.engine))
                self.last_count = current_count
        return self.df
    
    def force_reload(self) -> pd.DataFrame:
        """Force a reload of the data."""
        self.df = normalize_df(load_dataframe(self.engine))
        self.last_count = load_db_row_count(self.engine)
        self.last_check = time.time()
        return self.df


class DbCountCache:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.count = 0
        self.last_load = 0.0
    
    def get(self) -> int:
        now = time.time()
        if now - self.last_load >= DB_COUNT_TTL:
            self.count = load_db_row_count(self.engine)
            self.last_load = now
        return self.count


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        try:
            df['date_local'] = df['date'].dt.tz_convert(TIMEZONE)
        except Exception:
            df['date_local'] = df['date']
    else:
        df['date_local'] = pd.NaT
    
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0
    
    for col in ['player_gamertag', 'playlist', 'game_type', 'map', 'outcome']:
        if col not in df.columns:
            df[col] = ''
    
    return df


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        with open(STATUS_PATH, 'r') as file:
            status = json.load(file)
        if isinstance(status, dict) and status.get('last_update'):
            status['last_update'] = format_last_update(status.get('last_update'))
        return status
    except Exception:
        return {}


def load_settings() -> dict:
    defaults = {
        'match_limit': int(os.getenv('HALO_MATCH_LIMIT', '100')),
        'update_interval': int(os.getenv('HALO_UPDATE_INTERVAL', '60')),
        'force_refresh': os.getenv('HALO_FORCE_REFRESH', 'false').strip().lower() in ['1', 'true', 'yes', 'on']
    }
    
    if not SETTINGS_PATH.exists():
        return defaults
    
    try:
        with open(SETTINGS_PATH, 'r') as file:
            settings = json.load(file)
        return {**defaults, **settings}
    except Exception:
        return defaults


def ensure_suggestions_table(engine) -> None:
    ddl = '''
    CREATE TABLE IF NOT EXISTS halo_suggestions (
        id BIGSERIAL PRIMARY KEY,
        submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        name TEXT,
        gamertag TEXT,
        contact TEXT,
        summary TEXT NOT NULL,
        details TEXT,
        follow_up TEXT
    )
    '''
    with engine.begin() as conn:
        conn.execute(text(ddl))


def fetch_suggestions(engine, limit: int = 50) -> list[dict]:
    ensure_suggestions_table(engine)
    query = text('''
    SELECT id, submitted_at, name, gamertag, contact, summary, details, follow_up
    FROM halo_suggestions
    ORDER BY submitted_at DESC
    LIMIT :limit
    ''')
    
    with engine.begin() as conn:
        rows = conn.execute(query, {'limit': limit}).fetchall()
    
    suggestions = []
    for row in rows:
        suggestions.append({
            'id': row[0],
            'submitted_at': format_date(row[1]),
            'name': row[2] or '',
            'gamertag': row[3] or '',
            'contact': row[4] or '',
            'summary': row[5] or '',
            'details': row[6] or '',
            'follow_up': row[7] or ''
        })
    return suggestions


def save_suggestion(engine, payload: dict) -> None:
    ensure_suggestions_table(engine)
    query = text('''
    INSERT INTO halo_suggestions (name, gamertag, contact, summary, details, follow_up)
    VALUES (:name, :gamertag, :contact, :summary, :details, :follow_up)
    ''')
    with engine.begin() as conn:
        conn.execute(query, payload)


def load_presence() -> dict:
    """Best-effort load of online/presence information."""
    for name in ['online_status.json', 'player_presence.json', 'player_status.json']:
        path = data_path(name)
        if not path.exists():
            continue
        
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            
            if isinstance(data, dict):
                return data
            
            if isinstance(data, list):
                return {'players': {str(item): {'online': True} for item in data}}
        except Exception as e:
            print(f'⚠️ Failed to load presence file {path}: {e}')
    
    return {}


def is_player_online(presence: dict, gamertag: str) -> bool:
    if not presence or not gamertag:
        return False
    
    players = presence.get('players') if isinstance(presence, dict) else None
    
    if isinstance(players, dict):
        for key, val in players.items():
            if str(key).strip().lower() == str(gamertag).strip().lower():
                if isinstance(val, dict):
                    return bool(val.get('online') or val.get('is_online'))
                return bool(val)
    
    if isinstance(presence, dict):
        for key, val in presence.items():
            if str(key).strip().lower() == str(gamertag).strip().lower():
                if isinstance(val, dict):
                    return bool(val.get('online') or val.get('is_online'))
                return bool(val)
    
    return False


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, 'w') as file:
            json.dump(settings, file, indent=2)
    except Exception as e:
        print(f'Failed to save settings: {e}')


def safe_int(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def format_int(value) -> str:
    return f'{safe_int(value):,}' if safe_int(value) else '0'


def format_float(value, digits=2) -> str:
    return f'{safe_float(value):.{digits}f}' if safe_float(value) else '0'


def format_pct(value) -> str:
    pct = safe_float(value)
    if pct <= 1.0:
        pct *= 100
    return f'{pct:.1f}%' if pct else '0%'


def format_optional_int(value) -> str:
    if value is None or pd.isna(value):
        return '-'
    int_val = safe_int(value)
    return '-' if int_val == 0 else f'{int_val:,}'


def format_optional_float(value, digits=2) -> str:
    if value is None or pd.isna(value):
        return '-'
    float_val = safe_float(value)
    return '-' if float_val == 0 else f'{float_val:.{digits}f}'


def format_optional_pct(value) -> str:
    if value is None or pd.isna(value):
        return '-'
    pct_val = safe_float(value)
    if pct_val <= 1.0:
        pct_val *= 100
    return '-' if pct_val == 0 else f'{pct_val:.1f}%'


def normalize_map_name(map_name: str) -> str:
    if map_name is None or pd.isna(map_name):
        return ''
    text = str(map_name).strip()
    if not text:
        return ''
    text = re.sub(r'\s-\sranked(?:\s+arena)?$', '', text, flags=re.IGNORECASE).strip()
    return text


def add_normalized_map_column(df: pd.DataFrame, source_col: str = 'map') -> pd.DataFrame:
    if df.empty or source_col not in df.columns:
        return df
    working = df.copy()
    working['_map_normalized'] = working[source_col].map(normalize_map_name)
    return working


def series_max(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors='coerce')
    if series.dropna().empty:
        return None
    return series.max()


def series_min(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors='coerce')
    if series.dropna().empty:
        return None
    return series.min()


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(0)
    return pd.Series([0] * len(df), index=df.index)


def score_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if 'personal_score' in df.columns:
        return numeric_series(df, 'personal_score')
    if 'score' in df.columns:
        return numeric_series(df, 'score')
    return pd.Series(dtype=float)


def objective_score_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    base_score = score_series(df)
    if base_score.empty:
        return pd.Series(dtype=float)
    kills = numeric_series(df, 'kills')
    assists = numeric_series(df, 'assists')
    callouts = numeric_series(df, 'callout_assists')
    return base_score - (kills * 100) - (assists * 50) - (callouts * 10)


def to_number(value) -> float | None:
    """Best-effort conversion of formatted strings to float for heatmaps."""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    text = str(value).strip()
    if not text or text == '-':
        return None
    
    try:
        text = text.replace(',', '')
        if text.endswith('%'):
            text = text[:-1]
        return float(text)
    except Exception:
        return None


def add_heatmap_classes(rows: list, stat_columns: dict) -> None:
    """Mutates rows in-place adding <col>_heat CSS class fields."""
    if not rows:
        return
    
    for col, higher_better in stat_columns.items():
        values = [to_number(r.get(col)) for r in rows]
        for r in rows:
            r[f'{col}_heat'] = get_heatmap_class(r.get(col), values, higher_better)


def add_outlier_classes(rows: list, stat_columns: list[str], iqr_mult: float = 1.5) -> None:
    if not rows or not stat_columns:
        return
    
    for col in stat_columns:
        values = [to_number(r.get(col)) for r in rows]
        numeric = [v for v in values if v is not None]
        
        if len(numeric) < 4:
            continue
        
        series = pd.Series(numeric)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                continue
            low = mean - 2 * std
            high = mean + 2 * std
        else:
            low = q1 - iqr_mult * iqr
            high = q3 + iqr_mult * iqr
        
        for r in rows:
            value = to_number(r.get(col))
            if value is None:
                continue
            
            cls = None
            if value <= low:
                cls = 'outlier-low'
            elif value >= high:
                cls = 'outlier-high'
            
            if not cls:
                continue
            
            existing = r.get(f'{col}_heat', '')
            r[f'{col}_heat'] = f'{existing} {cls}'.strip()


def get_heatmap_class(value, values_list, higher_is_better=True):
    """Return CSS class based on value's percentile within values_list."""
    try:
        if not values_list or value is None:
            return ''
        
        val = to_number(value)
        if val is None:
            return ''
        
        vals = [to_number(v) for v in values_list]
        vals = [v for v in vals if v is not None]
        
        if not vals or len(vals) < 2:
            return ''
        
        below_count = sum(1 for v in vals if v < val)
        percentile = below_count / len(vals)
        
        if not higher_is_better:
            percentile = 1 - percentile
        
        if percentile >= 0.8:
            return 'heat-excellent'
        if percentile >= 0.6:
            return 'heat-good'
        if percentile >= 0.4:
            return 'heat-average'
        if percentile >= 0.2:
            return 'heat-below'
        return 'heat-poor'
    except (ValueError, TypeError):
        return ''


def format_last_update(value) -> str | None:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors='coerce')
        if pd.isna(ts):
            return str(value)
        try:
            ts = ts.tz_convert(TIMEZONE)
        except Exception:
            pass
        return ts.strftime('%Y-%m-%d %I:%M %p')
    except Exception:
        return str(value)


def format_date(value) -> str:
    if value is None or pd.isna(value):
        return '-'
    try:
        ts = pd.to_datetime(value, utc=True, errors='coerce')
        if pd.isna(ts):
            return str(value)
        try:
            ts = ts.tz_convert(TIMEZONE)
        except Exception:
            pass
        return ts.strftime('%Y-%m-%d %I:%M %p')
    except Exception:
        return str(value)


def format_iso(value) -> str:
    try:
        if value is None or pd.isna(value):
            return ''
        ts = pd.to_datetime(value, utc=True, errors='coerce')
        if pd.isna(ts):
            return ''
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        return ts.isoformat()
    except Exception:
        return ''


def format_signed(value, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return '-'
    sign = '+' if value > 0 else ''
    if digits <= 0:
        return f'{sign}{value:.0f}'
    return f'{sign}{value:.{digits}f}'


def outcome_class(value: str) -> str:
    text = str(value or '').strip().lower()
    if text in ('win', 'won'):
        return 'outcome-win'
    if text in ('loss', 'lose', 'lost'):
        return 'outcome-loss'
    if text == 'tie':
        return 'outcome-tie'
    if text == 'dnf':
        return 'outcome-dnf'
    if text == 'left':
        return 'outcome-left'
    return 'outcome-unknown'


def safe_kda(kills, assists, deaths) -> float:
    kills = safe_float(kills)
    assists = safe_float(assists)
    deaths = safe_float(deaths)
    return kills + assists / 3 - deaths


def compute_streaks(player_df: pd.DataFrame) -> tuple[int, int]:
    if player_df.empty or 'outcome' not in player_df.columns or 'date' not in player_df.columns:
        return 0, 0
    ordered = player_df.copy()
    ordered['date'] = pd.to_datetime(ordered['date'], errors='coerce', utc=True)
    ordered = ordered.dropna(subset=['date']).sort_values('date')
    if ordered.empty:
        return 0, 0
    max_win = max_loss = 0
    current_win = current_loss = 0
    for outcome in ordered['outcome'].astype(str).str.lower():
        if outcome == 'win':
            current_win += 1
            current_loss = 0
        elif outcome == 'loss':
            current_loss += 1
            current_win = 0
        else:
            current_win = 0
            current_loss = 0
        if current_win > max_win:
            max_win = current_win
        if current_loss > max_loss:
            max_loss = current_loss
    return max_win, max_loss


def compute_current_streak(player_df: pd.DataFrame) -> int:
    if player_df.empty or 'outcome' not in player_df.columns or 'date' not in player_df.columns:
        return 0
    ordered = player_df.copy()
    ordered['date'] = pd.to_datetime(ordered['date'], errors='coerce', utc=True)
    ordered = ordered.dropna(subset=['date']).sort_values('date', ascending=False)
    if ordered.empty:
        return 0
    streak = 0
    for outcome in ordered['outcome'].astype(str).str.lower():
        if outcome not in ('win', 'loss'):
            if streak == 0:
                continue
            break
        if streak == 0:
            streak = 1 if outcome == 'win' else -1
            continue
        if outcome == 'win' and streak > 0:
            streak += 1
        elif outcome == 'loss' and streak < 0:
            streak -= 1
        else:
            break
    return streak


def unique_sorted(series: pd.Series) -> list:
    values = [str(v).strip() for v in series.dropna().unique().tolist() if str(v).strip()]
    return sorted(set(values))


def apply_filters(df: pd.DataFrame, player: str, playlist: str, mode: str) -> pd.DataFrame:
    filtered = df
    if player and player != 'all' and 'player_gamertag' in filtered.columns:
        filtered = filtered[filtered['player_gamertag'] == player]
    if playlist and playlist != 'all' and 'playlist' in filtered.columns:
        filtered = filtered[filtered['playlist'] == playlist]
    if mode and mode != 'all' and 'game_type' in filtered.columns:
        filtered = filtered[filtered['game_type'] == mode]
    return filtered


def extract_csr_values(df: pd.DataFrame) -> pd.Series:
    """Extract CSR values, preferring post_match_csr over pre_match_csr."""
    if df.empty:
        return pd.Series(dtype=float)
    
    post_vals = pd.to_numeric(df.get('post_match_csr', pd.Series()), errors='coerce') if 'post_match_csr' in df.columns else pd.Series()
    pre_vals = pd.to_numeric(df.get('pre_match_csr', pd.Series()), errors='coerce') if 'pre_match_csr' in df.columns else pd.Series()
    
    csr_vals = post_vals.where(post_vals > 0).combine_first(pre_vals.where(pre_vals > 0))
    return csr_vals.dropna()


def compute_csr_window_delta(player_df: pd.DataFrame, days: int) -> float | None:
    """Compute CSR change over the last N days."""
    if player_df.empty or 'date' not in player_df.columns:
        return None
    
    now = pd.Timestamp.now(tz='UTC')
    cutoff = now - pd.Timedelta(days=days)
    window_df = player_df[player_df['date'] >= cutoff].sort_values('date', ascending=True)
    
    if window_df.empty:
        return None
    
    csr_vals = extract_csr_values(window_df)
    if csr_vals.empty:
        return None
    
    return float(csr_vals.iloc[-1] - csr_vals.iloc[0])


def build_csr_overview(df: pd.DataFrame) -> list:
    """Build CSR overview showing current CSR, session change, and deltas."""
    if df.empty or 'playlist' not in df.columns or 'date' not in df.columns or 'player_gamertag' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy()
    if ranked_df.empty:
        return []
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], errors='coerce', utc=True)
    ranked_df = ranked_df.dropna(subset=['date'])
    if ranked_df.empty:
        return []
    
    presence = load_presence()
    rows = []
    
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            continue
        
        latest_row = player_df.iloc[0]
        current_csr_val = latest_row.get('post_match_csr')
        if pd.isna(current_csr_val) or current_csr_val is None:
            current_csr_val = latest_row.get('pre_match_csr')
        
        # Find session matches (30 min gap)
        session_rows = [latest_row]
        prev_ts = pd.Timestamp(latest_row['date'])
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.tz_localize('UTC')
        
        for _, r in player_df.iloc[1:].iterrows():
            ts = pd.Timestamp(r['date'])
            if pd.isna(ts):
                continue
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            
            if prev_ts - ts <= pd.Timedelta(minutes=30):
                session_rows.append(r)
                prev_ts = ts
            else:
                break
        
        session_df = pd.DataFrame(session_rows).sort_values('date', ascending=True)
        session_start_csr_val = None
        if not session_df.empty:
            first_row = session_df.iloc[0]
            session_start_csr_val = first_row.get('pre_match_csr')
            if pd.isna(session_start_csr_val) or session_start_csr_val == 0:
                session_start_csr_val = first_row.get('post_match_csr')
        
        session_csr_change = None
        if current_csr_val and session_start_csr_val and current_csr_val > 0 and session_start_csr_val > 0:
            session_csr_change = current_csr_val - session_start_csr_val
        
        # Max CSR
        max_csr_val = None
        max_csr_date = None
        if 'post_match_csr' in player_df.columns:
            post_vals = pd.to_numeric(player_df['post_match_csr'], errors='coerce')
            if post_vals.notna().any():
                max_csr_val = float(post_vals.max())
                max_idx = post_vals.idxmax()
                max_csr_date = player_df.loc[max_idx, 'date']
        
        delta_7 = compute_csr_window_delta(player_df, 7)
        delta_30 = compute_csr_window_delta(player_df, 30)
        delta_90 = compute_csr_window_delta(player_df, 90)
        
        target_delta_val = None
        if current_csr_val and not pd.isna(current_csr_val) and current_csr_val > 0:
            target_delta_val = float(current_csr_val) - 1700.0
        
        rows.append({
            'player': player,
            'is_online': is_player_online(presence, player),
            'last_match_iso': format_iso(latest_row.get('date')),
            'current_csr': format_float(current_csr_val, 1) if current_csr_val and not pd.isna(current_csr_val) else '-',
            'session_start_csr': format_float(session_start_csr_val, 1) if session_start_csr_val and not pd.isna(session_start_csr_val) else '-',
            'session_csr_change': format_signed(session_csr_change, 1) if session_csr_change is not None else '-',
            'delta_7': format_signed(delta_7, 0) if delta_7 is not None else '-',
            'delta_30': format_signed(delta_30, 0) if delta_30 is not None else '-',
            'delta_90': format_signed(delta_90, 0) if delta_90 is not None else '-',
            'target_delta': format_signed(target_delta_val, 0) if target_delta_val is not None else '-',
            'max_csr': format_float(max_csr_val, 1) if max_csr_val else '-',
            'max_csr_date': format_date(max_csr_date) if max_csr_date else '-'
        })
    
    add_heatmap_classes(rows, {
        'current_csr': True, 'session_csr_change': True, 
        'delta_7': True, 'delta_30': True, 'delta_90': True,
        'target_delta': True, 'max_csr': True
    })
    
    rows.sort(key=lambda x: to_number(x['current_csr']) or -999, reverse=True)
    return rows


def build_csr_trends(df: pd.DataFrame) -> dict:
    """Build CSR trend data for all players."""
    if df.empty or 'player_gamertag' not in df.columns:
        return {}
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    if ranked_df.empty or 'date' not in ranked_df.columns:
        return {}
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], errors='coerce', utc=True)
    ranked_df = ranked_df.dropna(subset=['date'])
    
    try:
        ranked_df['date_local'] = ranked_df['date'].dt.tz_convert(TIMEZONE)
    except:
        ranked_df['date_local'] = ranked_df['date']
    
    ranked_df['date_str'] = ranked_df['date_local'].dt.strftime('%Y-%m-%d')
    
    trends = {}
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player].sort_values('date')
        if player_df.empty:
            continue
        
        player_df['csr_value'] = extract_csr_values(player_df)
        player_df = player_df.dropna(subset=['csr_value'])
        
        if player_df.empty:
            continue
        
        daily = player_df.groupby('date_str')['csr_value'].last().reset_index()
        daily['date_key'] = pd.to_datetime(daily['date_str'], errors='coerce')
        daily = daily.sort_values('date_key')
        
        trends[player] = [
            {'date': row['date_str'], 'csr': float(row['csr_value'])}
            for _, row in daily.iterrows() if pd.notna(row['csr_value'])
        ]
    
    return trends


def build_outlier_spotlight(df: pd.DataFrame, range_key: str | None = None) -> list[dict]:
    """Build spotlight highlighting player outliers vs the group."""
    if df.empty or 'player_gamertag' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    range_key = (range_key or 'all').lower()
    if range_key == 'lifetime':
        range_key = 'all'
    if ranked_df.empty:
        return []

    if range_key != 'all':
        if 'date' not in ranked_df.columns:
            return []
        ranked_df = ranked_df.copy()
        ranked_df['date'] = pd.to_datetime(ranked_df['date'], errors='coerce', utc=True)
        ranked_df = ranked_df.dropna(subset=['date'])
        ranked_df = apply_trend_range(ranked_df, range_key)
        if ranked_df.empty:
            return []
    
    player_stats = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        if games == 0:
            continue
        
        total_kills = numeric_series(player_df, 'kills').sum()
        total_deaths = numeric_series(player_df, 'deaths').sum()
        total_assists = numeric_series(player_df, 'assists').sum()
        
        kills_pg = total_kills / games
        deaths_pg = total_deaths / games
        assists_pg = total_assists / games
        kda = safe_kda(kills_pg, assists_pg, deaths_pg)
        
        fired = numeric_series(player_df, 'shots_fired').sum()
        hit = numeric_series(player_df, 'shots_hit').sum()
        if fired > 0:
            accuracy = hit / fired * 100
        else:
            accuracy = numeric_series(player_df, 'accuracy').mean()
            if accuracy <= 1:
                accuracy *= 100
        
        total_dmg_dealt = numeric_series(player_df, 'damage_dealt').sum()
        total_dmg_taken = numeric_series(player_df, 'damage_taken').sum()
        dmg_diff_pg = (total_dmg_dealt - total_dmg_taken) / games
        
        total_duration = numeric_series(player_df, 'duration').sum()
        dmg_per_min = total_dmg_dealt / (total_duration / 60) if total_duration > 0 else 0
        
        score_total = score_series(player_df).sum()
        score_pg = score_total / games
        
        obj_scores = objective_score_series(player_df)
        obj_score_pg = obj_scores.sum() / games if not obj_scores.empty else 0
        
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        win_rate = wins / games * 100
        
        player_stats.append({
            'player': player,
            'games': games,
            'win_rate': win_rate,
            'kda': kda,
            'kills_pg': kills_pg,
            'deaths_pg': deaths_pg,
            'assists_pg': assists_pg,
            'accuracy': accuracy,
            'dmg_per_min': dmg_per_min,
            'dmg_diff_pg': dmg_diff_pg,
            'score_pg': score_pg,
            'obj_score_pg': obj_score_pg
        })
    
    if not player_stats:
        return []
    
    stats_info = [
        {'key': 'win_rate', 'label': 'Win Rate', 'higher_better': True, 'format': lambda v: f'{v:.1f}%'},
        {'key': 'kda', 'label': 'KDA', 'higher_better': True, 'format': lambda v: f'{v:.2f}'},
        {'key': 'kills_pg', 'label': 'Kills/Game', 'higher_better': True, 'format': lambda v: f'{v:.1f}'},
        {'key': 'deaths_pg', 'label': 'Deaths/Game', 'higher_better': False, 'format': lambda v: f'{v:.1f}'},
        {'key': 'assists_pg', 'label': 'Assists/Game', 'higher_better': True, 'format': lambda v: f'{v:.1f}'},
        {'key': 'accuracy', 'label': 'Accuracy', 'higher_better': True, 'format': lambda v: f'{v:.1f}%'},
        {'key': 'dmg_per_min', 'label': 'Damage/Min', 'higher_better': True, 'format': lambda v: f'{v:.0f}'},
        {'key': 'dmg_diff_pg', 'label': 'Damage Diff/Game', 'higher_better': True, 'format': lambda v: format_signed(v, 0)},
        {'key': 'score_pg', 'label': 'Score/Game', 'higher_better': True, 'format': lambda v: f'{v:.0f}'},
        {'key': 'obj_score_pg', 'label': 'Obj Score/Game', 'higher_better': True, 'format': lambda v: f'{v:.1f}'}
    ]
    
    stat_means = {}
    stat_stds = {}
    stat_values = {}
    for stat in stats_info:
        values = [row[stat['key']] for row in player_stats]
        series = pd.Series(values, dtype=float)
        stat_means[stat['key']] = series.mean()
        stat_stds[stat['key']] = series.std(ddof=0)
        stat_values[stat['key']] = values
    
    positive_vibes = ['On Fire', 'Heat Check', 'Hot Hand', 'Glow Up', 'Pop Off']
    negative_vibes = ['Cold Snap', 'Slump', 'Ice Bath', 'Rough Patch', 'Frost Bite']
    
    rows = []
    for row in player_stats:
        candidates = []
        for stat in stats_info:
            value = row[stat['key']]
            std = stat_stds.get(stat['key'], 0)
            if std == 0 or pd.isna(std):
                continue
            mean = stat_means.get(stat['key'], 0)
            z = (value - mean) / std
            adj = z if stat['higher_better'] else -z
            values = stat_values.get(stat['key'], [])
            if len(values) > 1:
                others_mean = (sum(values) - value) / (len(values) - 1)
            else:
                others_mean = mean
            if abs(others_mean) < 1e-6:
                diff_pct = 0.0 if abs(value) < 1e-6 else 100.0
            else:
                diff_pct = (value - others_mean) / abs(others_mean) * 100
            candidates.append({
                'stat': stat,
                'value': value,
                'adj': adj,
                'diff_pct': diff_pct
            })
        
        good = [c for c in candidates if c['adj'] > 0]
        bad = [c for c in candidates if c['adj'] < 0]
        good.sort(key=lambda c: c['adj'], reverse=True)
        bad.sort(key=lambda c: c['adj'])
        
        picks = []
        used = set()
        for entry in good[:5]:
            if entry['stat']['key'] not in used:
                picks.append(entry)
                used.add(entry['stat']['key'])
        for entry in bad[:5]:
            if entry['stat']['key'] not in used:
                picks.append(entry)
                used.add(entry['stat']['key'])
        if len(picks) < 10:
            remaining = [c for c in candidates if c['stat']['key'] not in used]
            remaining.sort(key=lambda c: abs(c['adj']), reverse=True)
            for entry in remaining:
                picks.append(entry)
                used.add(entry['stat']['key'])
                if len(picks) >= 10:
                    break
        
        highlights = []
        for entry in picks[:10]:
            stat = entry['stat']
            diff_pct = entry['diff_pct']
            sign = '+' if diff_pct >= 0 else ''
            value = stat['format'](entry['value'])
            advantage = entry['adj'] > 0
            emoji = '🔥' if advantage else '🥶'
            vibe_list = positive_vibes if advantage else negative_vibes
            vibe = vibe_list[abs(hash(stat['key'])) % len(vibe_list)]
            highlights.append(
                f"{emoji} {vibe}: {stat['label']} {value} ({sign}{diff_pct:.0f}% vs pack)"
            )
        
        while len(highlights) < 10:
            highlights.append('-')
        
        rows.append({'player': row['player'], 'highlights': highlights[:10]})
    
    return rows


def build_ranked_arena_summary(df: pd.DataFrame) -> list:
    """Build summary for each player's last ranked session."""
    if df.empty:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    if 'date' not in ranked_df.columns:
        return []
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], errors='coerce', utc=True)
    ranked_df = ranked_df.dropna(subset=['date']).sort_values('date', ascending=False)
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            continue
        
        # Find session (30 min gaps)
        session_matches = []
        prev_time = None
        
        for idx, row_data in player_df.iterrows():
            match_time = row_data['date']
            if prev_time is None:
                session_matches.append(idx)
                prev_time = match_time
                continue
            
            time_diff = (prev_time - match_time).total_seconds() / 60
            if time_diff <= 30:
                session_matches.append(idx)
                prev_time = match_time
            else:
                break
        
        session_df = player_df.loc[session_matches]
        if session_df.empty:
            continue
        
        games = len(session_df)
        outcomes = session_df['outcome'].astype(str).str.lower() if 'outcome' in session_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        win_pct = wins / games * 100 if games > 0 else 0
        
        # Basic stats
        total_kills = pd.to_numeric(session_df.get('kills', 0), errors='coerce').fillna(0).sum()
        total_deaths = pd.to_numeric(session_df.get('deaths', 0), errors='coerce').fillna(0).sum()
        total_assists = pd.to_numeric(session_df.get('assists', 0), errors='coerce').fillna(0).sum()
        kills = total_kills / games if games else 0
        deaths = total_deaths / games if games else 0
        assists = total_assists / games if games else 0
        kda = safe_kda(kills, assists, deaths)
        kd1 = kills / deaths if deaths > 0 else kills
        kd2 = (kills + assists) / deaths if deaths > 0 else kills + assists
        
        # Damage stats
        total_dmg_dealt = pd.to_numeric(session_df.get('damage_dealt', 0), errors='coerce').fillna(0).sum()
        total_dmg_taken = pd.to_numeric(session_df.get('damage_taken', 0), errors='coerce').fillna(0).sum()
        dmg_plus = total_dmg_dealt / games if games else 0
        dmg_minus = total_dmg_taken / games if games else 0
        dmg_diff = total_dmg_dealt - total_dmg_taken
        dmg_per_ka = total_dmg_dealt / (total_kills + total_assists) if (total_kills + total_assists) > 0 else 0
        dmg_per_death = total_dmg_dealt / total_deaths if total_deaths > 0 else total_dmg_dealt
        
        # Duration and dmg/min
        total_duration = pd.to_numeric(session_df.get('duration', 0), errors='coerce').fillna(0).sum()
        dmg_per_min = total_dmg_dealt / (total_duration / 60) if total_duration > 0 else 0
        
        # Team damage percentage
        team_dmg = pd.to_numeric(session_df.get('team_damage_dealt', 0), errors='coerce').fillna(0).sum()
        enemy_dmg = pd.to_numeric(session_df.get('enemy_team_damage_dealt', 0), errors='coerce').fillna(0).sum()
        dmg_pct_plus = (total_dmg_dealt / team_dmg * 100) if team_dmg > 0 else 0
        dmg_pct_minus = (total_dmg_dealt / enemy_dmg * 100) if enemy_dmg > 0 else 0
        
        # Accuracy
        fired = pd.to_numeric(session_df.get('shots_fired', 0), errors='coerce').fillna(0).sum()
        hit = pd.to_numeric(session_df.get('shots_hit', 0), errors='coerce').fillna(0).sum()
        accuracy = hit / fired * 100 if fired > 0 else 0
        
        # Score
        total_score = score_series(session_df).sum()
        score = total_score / games if games else 0
        team_score = pd.to_numeric(session_df.get('team_personal_score', 0), errors='coerce').fillna(0).sum()
        score_pct = (total_score / team_score * 100) if team_score > 0 else 0
        obj_scores = objective_score_series(session_df)
        obj_score = obj_scores.sum() / games if games else 0
        
        # Medals and misc
        total_medals = pd.to_numeric(session_df.get('medal_count', 0), errors='coerce').fillna(0).sum()
        medals = total_medals / games if games else 0
        avg_life = pd.to_numeric(session_df.get('average_life_duration', 0), errors='coerce').fillna(0).mean()
        callouts = pd.to_numeric(session_df.get('callout_assists', 0), errors='coerce').fillna(0).sum() / games if games else 0
        
        # CSR
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in session_df.columns:
            post_vals = pd.to_numeric(session_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[-1]
                post_csr = post_vals.iloc[-1]
        
        if 'pre_match_csr' in session_df.columns:
            pre_vals = pd.to_numeric(session_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = 0
        if pre_csr and post_csr:
            csr_delta = post_csr - pre_csr
        
        rows.append({
            'player': player,
            'session_date': format_date(session_df['date'].max()),
            'csr': format_float(latest_csr, 1) if latest_csr else '-',
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kills': format_float(kills, 1),
            'deaths': format_float(deaths, 1),
            'assists': format_float(assists, 1),
            'kd1': format_float(kd1, 2),
            'kd2': format_float(kd2, 2),
            'kda': format_float(kda, 2),
            'dmg_plus': format_float(dmg_plus, 0),
            'dmg_minus': format_float(dmg_minus, 0),
            'dmg_diff': format_signed(dmg_diff, 0),
            'dmg_per_ka': format_float(dmg_per_ka, 0),
            'dmg_per_death': format_float(dmg_per_death, 0),
            'dmg_per_min': format_float(dmg_per_min, 0),
            'dmg_pct_plus': format_float(dmg_pct_plus, 1),
            'dmg_pct_minus': format_float(dmg_pct_minus, 1),
            'fired': format_int(fired),
            'landed': format_int(hit),
            'accuracy': format_float(accuracy, 1),
            'score': format_float(score, 0),
            'obj_score': format_float(obj_score, 1),
            'score_pct': format_float(score_pct, 1),
            'medals': format_float(medals, 1),
            'avg_life': format_float(avg_life, 1),
            'callouts': format_float(callouts, 1),
            'pre_csr': format_int(pre_csr) if pre_csr else '-',
            'post_csr': format_int(post_csr) if post_csr else '-',
            'csr_delta': format_signed(csr_delta, 0)
        })
    
    add_heatmap_classes(rows, {
        'csr': True, 'games': True, 'win_pct': True, 'kda': True, 'kd1': True, 'kd2': True,
        'kills': True, 'deaths': False, 'assists': True,
        'dmg_plus': True, 'dmg_minus': False, 'dmg_diff': True,
        'dmg_per_ka': True, 'dmg_per_death': True, 'dmg_per_min': True,
        'dmg_pct_plus': True, 'dmg_pct_minus': True,
        'fired': True, 'landed': True, 'accuracy': True,
        'score': True, 'obj_score': True, 'score_pct': True,
        'medals': True, 'avg_life': True, 'callouts': True, 'csr_delta': True
    })
    
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_ranked_arena_30day(df: pd.DataFrame) -> list:
    """Build 30-day summary for ranked matches per player."""
    if df.empty or 'date' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []
    
    max_date = ranked_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=30)
    ranked_df = ranked_df[ranked_df['date'] >= cutoff_date]
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
                post_csr = post_vals.iloc[0]
        
        if 'pre_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = (post_csr - pre_csr) if pre_csr and post_csr else 0
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        row['pre_csr'] = format_int(pre_csr) if pre_csr else '-'
        row['post_csr'] = format_int(post_csr) if post_csr else '-'
        row['csr_delta'] = format_signed(csr_delta, 0)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_ranked_arena_90day(df: pd.DataFrame) -> list:
    """Build 90-day summary for ranked matches per player."""
    if df.empty or 'date' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []
    
    max_date = ranked_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    ranked_df = ranked_df[ranked_df['date'] >= cutoff_date]
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
                post_csr = post_vals.iloc[0]
        
        if 'pre_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = (post_csr - pre_csr) if pre_csr and post_csr else 0
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        row['pre_csr'] = format_int(pre_csr) if pre_csr else '-'
        row['post_csr'] = format_int(post_csr) if post_csr else '-'
        row['csr_delta'] = format_signed(csr_delta, 0)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_ranked_arena_180day(df: pd.DataFrame) -> list:
    """Build 180-day summary for ranked matches per player."""
    if df.empty or 'date' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []
    
    max_date = ranked_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=180)
    ranked_df = ranked_df[ranked_df['date'] >= cutoff_date]
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
                post_csr = post_vals.iloc[0]
        
        if 'pre_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = (post_csr - pre_csr) if pre_csr and post_csr else 0
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        row['pre_csr'] = format_int(pre_csr) if pre_csr else '-'
        row['post_csr'] = format_int(post_csr) if post_csr else '-'
        row['csr_delta'] = format_signed(csr_delta, 0)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_ranked_arena_1y(df: pd.DataFrame) -> list:
    """Build 1-year summary for ranked matches per player."""
    if df.empty or 'date' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []
    
    max_date = ranked_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=365)
    ranked_df = ranked_df[ranked_df['date'] >= cutoff_date]
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
                post_csr = post_vals.iloc[0]
        
        if 'pre_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = (post_csr - pre_csr) if pre_csr and post_csr else 0
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        row['pre_csr'] = format_int(pre_csr) if pre_csr else '-'
        row['post_csr'] = format_int(post_csr) if post_csr else '-'
        row['csr_delta'] = format_signed(csr_delta, 0)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_ranked_arena_2y(df: pd.DataFrame) -> list:
    """Build 2-year summary for ranked matches per player."""
    if df.empty or 'date' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []
    
    max_date = ranked_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=730)
    ranked_df = ranked_df[ranked_df['date'] >= cutoff_date]
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        pre_csr = None
        post_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
                post_csr = post_vals.iloc[0]
        
        if 'pre_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_df['pre_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            if not pre_vals.empty:
                pre_csr = pre_vals.iloc[0]
        
        csr_delta = (post_csr - pre_csr) if pre_csr and post_csr else 0
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        row['pre_csr'] = format_int(pre_csr) if pre_csr else '-'
        row['post_csr'] = format_int(post_csr) if post_csr else '-'
        row['csr_delta'] = format_signed(csr_delta, 0)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def calculate_player_stats(player_df: pd.DataFrame, games: int) -> dict:
    """Calculate all stats for a player's matches. Returns a dict of stat values."""
    if games == 0:
        return {}
    
    # Basic stats
    total_kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum()
    total_deaths = pd.to_numeric(player_df.get('deaths', 0), errors='coerce').fillna(0).sum()
    total_assists = pd.to_numeric(player_df.get('assists', 0), errors='coerce').fillna(0).sum()
    kills = total_kills / games
    deaths = total_deaths / games
    assists = total_assists / games
    kda = safe_kda(kills, assists, deaths)
    kd1 = kills / deaths if deaths > 0 else kills
    kd2 = (kills + assists) / deaths if deaths > 0 else kills + assists
    
    # Damage stats
    total_dmg_dealt = pd.to_numeric(player_df.get('damage_dealt', 0), errors='coerce').fillna(0).sum()
    total_dmg_taken = pd.to_numeric(player_df.get('damage_taken', 0), errors='coerce').fillna(0).sum()
    dmg_plus = total_dmg_dealt / games
    dmg_minus = total_dmg_taken / games
    dmg_diff = total_dmg_dealt - total_dmg_taken
    dmg_per_ka = total_dmg_dealt / (total_kills + total_assists) if (total_kills + total_assists) > 0 else 0
    dmg_per_death = total_dmg_dealt / total_deaths if total_deaths > 0 else total_dmg_dealt
    
    # Duration and dmg/min
    total_duration = pd.to_numeric(player_df.get('duration', 0), errors='coerce').fillna(0).sum()
    dmg_per_min = total_dmg_dealt / (total_duration / 60) if total_duration > 0 else 0
    
    # Team damage percentage
    team_dmg = pd.to_numeric(player_df.get('team_damage_dealt', 0), errors='coerce').fillna(0).sum()
    enemy_dmg = pd.to_numeric(player_df.get('enemy_team_damage_dealt', 0), errors='coerce').fillna(0).sum()
    dmg_pct_plus = (total_dmg_dealt / team_dmg * 100) if team_dmg > 0 else 0
    dmg_pct_minus = (total_dmg_dealt / enemy_dmg * 100) if enemy_dmg > 0 else 0
    
    # Accuracy
    fired = pd.to_numeric(player_df.get('shots_fired', 0), errors='coerce').fillna(0).sum()
    hit = pd.to_numeric(player_df.get('shots_hit', 0), errors='coerce').fillna(0).sum()
    accuracy = hit / fired * 100 if fired > 0 else 0
    
    # Score
    total_score = score_series(player_df).sum()
    score = total_score / games
    team_score = pd.to_numeric(player_df.get('team_personal_score', 0), errors='coerce').fillna(0).sum()
    score_pct = (total_score / team_score * 100) if team_score > 0 else 0
    obj_scores = objective_score_series(player_df)
    obj_score = obj_scores.sum() / games if games else 0
    
    # Medals and misc
    total_medals = pd.to_numeric(player_df.get('medal_count', 0), errors='coerce').fillna(0).sum()
    medals = total_medals / games
    avg_life = pd.to_numeric(player_df.get('average_life_duration', 0), errors='coerce').fillna(0).mean()
    callouts = pd.to_numeric(player_df.get('callout_assists', 0), errors='coerce').fillna(0).sum() / games
    
    return {
        'kills': kills, 'deaths': deaths, 'assists': assists,
        'kd1': kd1, 'kd2': kd2, 'kda': kda,
        'dmg_plus': dmg_plus, 'dmg_minus': dmg_minus, 'dmg_diff': dmg_diff,
        'dmg_per_ka': dmg_per_ka, 'dmg_per_death': dmg_per_death, 'dmg_per_min': dmg_per_min,
        'dmg_pct_plus': dmg_pct_plus, 'dmg_pct_minus': dmg_pct_minus,
        'fired': fired, 'hit': hit, 'accuracy': accuracy,
        'score': score, 'obj_score': obj_score, 'score_pct': score_pct,
        'medals': medals, 'avg_life': avg_life, 'callouts': callouts
    }


def format_player_stats_row(player: str, games: int, wins: int, stats: dict, csr: float = None) -> dict:
    """Format stats dict into a row dict with proper formatting."""
    win_pct = wins / games * 100 if games > 0 else 0
    return {
        'player': player,
        'csr': format_float(csr, 1) if csr else '-',
        'games': format_int(games),
        'win_pct': format_float(win_pct, 1),
        'kills': format_float(stats.get('kills', 0), 1),
        'deaths': format_float(stats.get('deaths', 0), 1),
        'assists': format_float(stats.get('assists', 0), 1),
        'kd1': format_float(stats.get('kd1', 0), 2),
        'kd2': format_float(stats.get('kd2', 0), 2),
        'kda': format_float(stats.get('kda', 0), 2),
        'dmg_plus': format_float(stats.get('dmg_plus', 0), 0),
        'dmg_minus': format_float(stats.get('dmg_minus', 0), 0),
        'dmg_diff': format_signed(stats.get('dmg_diff', 0), 0),
        'dmg_per_ka': format_float(stats.get('dmg_per_ka', 0), 0),
        'dmg_per_death': format_float(stats.get('dmg_per_death', 0), 0),
        'dmg_per_min': format_float(stats.get('dmg_per_min', 0), 0),
        'dmg_pct_plus': format_float(stats.get('dmg_pct_plus', 0), 1),
        'dmg_pct_minus': format_float(stats.get('dmg_pct_minus', 0), 1),
        'fired': format_int(stats.get('fired', 0)),
        'landed': format_int(stats.get('hit', 0)),
        'accuracy': format_float(stats.get('accuracy', 0), 1),
        'score': format_float(stats.get('score', 0), 0),
        'obj_score': format_float(stats.get('obj_score', 0), 1),
        'score_pct': format_float(stats.get('score_pct', 0), 1),
        'medals': format_float(stats.get('medals', 0), 1),
        'avg_life': format_float(stats.get('avg_life', 0), 1),
        'callouts': format_float(stats.get('callouts', 0), 1)
    }


FULL_HEATMAP_CONFIG = {
    'csr': True, 'games': True, 'win_pct': True, 'kda': True, 'kd1': True, 'kd2': True,
    'kills': True, 'deaths': False, 'assists': True,
    'dmg_plus': True, 'dmg_minus': False, 'dmg_diff': True,
    'dmg_per_ka': True, 'dmg_per_death': True, 'dmg_per_min': True,
    'dmg_pct_plus': True, 'dmg_pct_minus': True,
    'fired': True, 'landed': True, 'accuracy': True,
    'score': True, 'obj_score': True, 'score_pct': True,
    'medals': True, 'avg_life': True, 'callouts': True, 'csr_delta': True
}


def build_ranked_arena_lifetime(df: pd.DataFrame) -> list:
    """Build lifetime summary for all ranked matches per player."""
    if df.empty:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        stats = calculate_player_stats(player_df, games)
        
        latest_csr = None
        if 'post_match_csr' in player_df.columns and 'date' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
        
        row = format_player_stats_row(player, games, wins, stats, latest_csr)
        rows.append(row)
    
    add_heatmap_classes(rows, FULL_HEATMAP_CONFIG)
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


def build_breakdown(df: pd.DataFrame, column: str, limit: int = 100) -> list:
    """Build breakdown stats for maps/playlists/modes."""
    if df.empty or column not in df.columns:
        return []
    
    working = df.copy()
    group_col = column
    
    if column == 'map':
        working = add_normalized_map_column(df, column)
        group_col = '_map_normalized'
    
    rows = []
    grouped = working.groupby(group_col)
    
    for name, group in grouped:
        if not str(name).strip():
            continue
        
        matches = len(group)
        outcomes = group['outcome'].astype(str).str.lower() if 'outcome' in group.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        win_rate = wins / matches * 100 if matches else 0
        
        rows.append({
            'name': name,
            'matches': matches,
            'win_rate': win_rate
        })
    
    rows.sort(key=lambda item: item['matches'], reverse=True)
    trimmed = rows[:limit] if limit else rows
    
    out = [
        {
            'name': row['name'],
            'matches': format_int(row['matches']),
            'win_rate': f"{row['win_rate']:.1f}%" if row['matches'] else '0%'
        }
        for row in trimmed
    ]
    
    add_heatmap_classes(out, {'matches': True, 'win_rate': True})
    return out


def build_cards(df: pd.DataFrame) -> list:
    """Build summary cards for dashboard."""
    if df.empty:
        return []
    
    matches = len(df)
    outcomes = df['outcome'].astype(str).str.lower() if 'outcome' in df.columns else pd.Series()
    wins = (outcomes == 'win').sum() if not outcomes.empty else 0
    losses = (outcomes == 'loss').sum() if not outcomes.empty else 0
    
    kills = pd.to_numeric(df.get('kills', 0), errors='coerce').fillna(0).sum() if matches else 0
    deaths = pd.to_numeric(df.get('deaths', 0), errors='coerce').fillna(0).sum() if matches else 0
    assists = pd.to_numeric(df.get('assists', 0), errors='coerce').fillna(0).sum() if matches else 0
    
    avg_kda = safe_kda(kills / matches if matches else 0, 
                       assists / matches if matches else 0,
                       deaths / matches if matches else 0)
    
    accuracy = 0
    if 'shots_fired' in df.columns and 'shots_hit' in df.columns:
        fired = pd.to_numeric(df['shots_fired'], errors='coerce').fillna(0).sum()
        hit = pd.to_numeric(df['shots_hit'], errors='coerce').fillna(0).sum()
        accuracy = hit / fired * 100 if fired > 0 else 0
    
    win_rate = wins / matches * 100 if matches else 0
    
    return [
        {
            'label': 'Matches',
            'value': format_int(matches),
            'detail': 'Total matches'
        },
        {
            'label': 'Win Rate',
            'value': f'{win_rate:.1f}%',
            'detail': f'{wins}W - {losses}L'
        },
        {
            'label': 'Avg KDA',
            'value': format_float(avg_kda, 2),
            'detail': 'Kills + Assists/3 - Deaths'
        },
        {
            'label': 'Accuracy',
            'value': format_pct(accuracy / 100),
            'detail': 'Shot accuracy'
        }
    ]


def normalize_trend_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe for trend analysis."""
    if df.empty:
        return df
    
    working = df.copy()
    
    if 'playlist' in working.columns:
        working = working[working['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy()
    
    if 'outcome' in working.columns:
        outcome_lower = working['outcome'].astype(str).str.lower()
        working = working[outcome_lower != 'dnf'].copy()
    
    if 'date' in working.columns:
        working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
        working = working.dropna(subset=['date'])
        
        try:
            working['date_local'] = working['date'].dt.tz_convert(TIMEZONE)
        except:
            working['date_local'] = working['date']
        
        working['date_str'] = working['date_local'].dt.strftime('%Y-%m-%d')
    
    return working


def apply_trend_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    """Filter trends to a specific date range."""
    if df.empty or 'date' not in df.columns:
        return df
    
    days_map = {
        '7': 7, '30': 30, '90': 90, '180': 180,
        '365': 365, '730': 730, '1095': 1095
    }
    
    days = days_map.get(range_key)
    if days is None:  # 'all' or unknown
        return df
    
    working = df.copy()
    date_series = pd.to_datetime(working['date'], errors='coerce', utc=True)
    
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    working = working[date_series >= cutoff]
    
    return working


def apply_leaderboard_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns:
        return df
    
    period_days = {
        'week': 7,
        'month': 30
    }
    days = period_days.get(period)
    if not days:
        return df
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return working
    
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    return working[working['date'] >= cutoff]


def build_lifetime_stats(df: pd.DataFrame) -> list:
    """Build lifetime statistics per player."""
    if df.empty:
        return []
    
    rows = []
    for player in unique_sorted(df['player_gamertag']):
        player_df = df[df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        losses = (outcomes == 'loss').sum() if not outcomes.empty else 0
        ties = (outcomes == 'tie').sum() if not outcomes.empty else 0
        
        total_kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum()
        total_deaths = pd.to_numeric(player_df.get('deaths', 0), errors='coerce').fillna(0).sum()
        total_assists = pd.to_numeric(player_df.get('assists', 0), errors='coerce').fillna(0).sum()
        
        total_damage_dealt = pd.to_numeric(player_df.get('damage_dealt', 0), errors='coerce').fillna(0).sum()
        total_damage_taken = pd.to_numeric(player_df.get('damage_taken', 0), errors='coerce').fillna(0).sum()
        
        kills_pg = total_kills / games if games else 0
        deaths_pg = total_deaths / games if games else 0
        assists_pg = total_assists / games if games else 0
        damage_pg = total_damage_dealt / games if games else 0
        
        kda = safe_kda(kills_pg, assists_pg, deaths_pg)
        kd_ratio = kills_pg / deaths_pg if deaths_pg > 0 else kills_pg
        
        accuracy = 0
        if 'shots_fired' in player_df.columns and 'shots_hit' in player_df.columns:
            fired = pd.to_numeric(player_df['shots_fired'], errors='coerce').fillna(0).sum()
            hit = pd.to_numeric(player_df['shots_hit'], errors='coerce').fillna(0).sum()
            accuracy = hit / fired if fired > 0 else 0
        
        total_score = score_series(player_df).sum()
        avg_score = total_score / games if games else 0
        obj_scores = objective_score_series(player_df)
        avg_obj_score = obj_scores.sum() / games if games else 0
        
        rows.append({
            'player': player,
            'matches': format_int(games),
            'wins': format_int(wins),
            'losses': format_int(losses),
            'win_rate': format_float(wins / games * 100 if games else 0, 1),
            'kills': format_float(kills_pg, 1),
            'deaths': format_float(deaths_pg, 1),
            'assists': format_float(assists_pg, 1),
            'kda': format_float(kda, 2),
            'accuracy': format_pct(accuracy),
            'avg_score': format_float(avg_score, 0),
            'avg_obj_score': format_float(avg_obj_score, 1)
        })
    
    add_heatmap_classes(rows, {
        'matches': True, 'wins': True, 'losses': False, 'win_rate': True,
        'kills': True, 'deaths': False, 'assists': True,
        'kda': True, 'accuracy': True, 'avg_score': True, 'avg_obj_score': True
    })
    
    return rows


def build_session_history(df: pd.DataFrame, limit: int | None = 20) -> list:
    """Build recent match history across all players."""
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    
    if working.empty:
        return []
    
    if isinstance(limit, int) and limit > 0:
        working = working.nlargest(limit, 'date')
    else:
        working = working.sort_values('date', ascending=False)
    
    score_values = score_series(working)
    if score_values.empty:
        score_values = pd.Series(0.0, index=working.index)
    obj_scores = objective_score_series(working)
    if obj_scores.empty:
        obj_scores = pd.Series(0.0, index=working.index)
    
    rows = []
    for idx, row in working.iterrows():
        kills = safe_float(row.get('kills', 0))
        deaths = safe_float(row.get('deaths', 0))
        assists = safe_float(row.get('assists', 0))
        kda = safe_kda(kills, assists, deaths)
        kd = kills / deaths if deaths > 0 else kills
        
        damage_dealt = safe_float(row.get('damage_dealt', 0))
        damage_taken = safe_float(row.get('damage_taken', 0))
        damage_diff = damage_dealt - damage_taken
        
        fired = safe_float(row.get('shots_fired', 0))
        hit = safe_float(row.get('shots_hit', 0))
        accuracy = hit / fired * 100 if fired > 0 else safe_float(row.get('accuracy', 0))
        
        score = safe_float(score_values.loc[idx]) if idx in score_values.index else 0
        obj_score = safe_float(obj_scores.loc[idx]) if idx in obj_scores.index else 0
        
        rows.append({
            'date': format_date(row.get('date')),
            'player': row.get('player_gamertag', ''),
            'game_type': row.get('game_type', ''),
            'map': row.get('map', ''),
            'playlist': row.get('playlist', ''),
            'outcome': str(row.get('outcome', '')).title(),
            'outcome_class': outcome_class(row.get('outcome', '')),
            'kills': format_int(kills),
            'deaths': format_int(deaths),
            'assists': format_int(assists),
            'kda': format_float(kda, 2),
            'kd': format_float(kd, 2),
            'damage_dealt': format_int(damage_dealt),
            'damage_taken': format_int(damage_taken),
            'damage_diff': format_signed(damage_diff, 0),
            'shots_fired': format_int(fired),
            'shots_landed': format_int(hit),
            'accuracy': format_pct(accuracy),
            'score': format_int(score),
            'obj_score': format_float(obj_score, 1),
            'medals': format_int(row.get('medal_count', 0)),
            'avg_life': format_float(row.get('average_life_duration', 0), 1),
            'headshots': format_int(row.get('headshot_kills', 0)),
            'melee': format_int(row.get('melee_kills', 0)),
            'grenade': format_int(row.get('grenade_kills', 0)),
            'power': format_int(row.get('power_weapon_kills', 0)),
            'callouts': format_int(row.get('callout_assists', 0))
        })
    
    add_heatmap_classes(rows, {
        'kills': True, 'deaths': False, 'assists': True,
        'kda': True, 'kd': True,
        'damage_dealt': True, 'damage_taken': False, 'damage_diff': True,
        'accuracy': True, 'score': True, 'obj_score': True,
        'medals': True, 'avg_life': True,
        'headshots': True, 'melee': True, 'grenade': True, 'power': True,
        'callouts': True
    })
    
    return rows


def extract_objective_score(df: pd.DataFrame) -> pd.Series:
    """Calculate objective score from personal score and combat/callout bonuses."""
    return objective_score_series(df)


def safe_col_sum(df: pd.DataFrame, col_name: str) -> float:
    """Safely get sum of a column, returning 0 if column doesn't exist."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').fillna(0).sum()
    return 0


def build_objective_stats(df: pd.DataFrame, period: str = 'all') -> list:
    """Build objective statistics (CTF, Oddball, Stronghold, KOTH, Extraction)."""
    if df.empty:
        return []
    
    # Filter by period if needed
    working = df.copy()
    if period == 'session':
        if 'date' in working.columns:
            working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
            working = working.sort_values('date', ascending=False).head(50)
    elif period == '30day' and 'date' in working.columns:
        working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
        working = working[working['date'] >= cutoff]
    
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        
        # CTF stats
        ctf_caps = safe_col_sum(player_df, 'capture_the_flag_stats_flag_captures')
        ctf_grabs = safe_col_sum(player_df, 'capture_the_flag_stats_flag_grabs')
        ctf_returns = safe_col_sum(player_df, 'capture_the_flag_stats_flag_returns')
        ctf_steals = safe_col_sum(player_df, 'capture_the_flag_stats_flag_steals')
        ctf_secures = safe_col_sum(player_df, 'capture_the_flag_stats_flag_secures')
        ctf_carrier_kills = safe_col_sum(player_df, 'capture_the_flag_stats_flag_carriers_killed')
        ctf_returner_kills = safe_col_sum(player_df, 'capture_the_flag_stats_flag_returners_killed')
        ctf_kills_as_carrier = safe_col_sum(player_df, 'capture_the_flag_stats_kills_as_flag_carrier')
        ctf_kills_as_returner = safe_col_sum(player_df, 'capture_the_flag_stats_kills_as_flag_returner')
        ctf_time = safe_col_sum(player_df, 'capture_the_flag_stats_time_as_flag_carrier')
        
        # Oddball stats
        oddball_time = safe_col_sum(player_df, 'oddball_stats_time_as_skull_carrier')
        oddball_longest = safe_col_sum(player_df, 'oddball_stats_longest_time_as_skull_carrier')
        oddball_grabs = safe_col_sum(player_df, 'oddball_stats_skull_grabs')
        oddball_ticks = safe_col_sum(player_df, 'oddball_stats_skull_scoring_ticks')
        oddball_carrier_kills = safe_col_sum(player_df, 'oddball_stats_kills_as_skull_carrier')
        oddball_carriers_killed = safe_col_sum(player_df, 'oddball_stats_skull_carriers_killed')
        
        # Stronghold/Zone stats
        sh_caps = safe_col_sum(player_df, 'zones_stats_stronghold_captures')
        sh_secures = safe_col_sum(player_df, 'zones_stats_stronghold_secures')
        sh_ticks = safe_col_sum(player_df, 'zones_stats_stronghold_scoring_ticks')
        sh_off_kills = safe_col_sum(player_df, 'zones_stats_stronghold_offensive_kills')
        sh_def_kills = safe_col_sum(player_df, 'zones_stats_stronghold_defensive_kills')
        sh_time = safe_col_sum(player_df, 'zones_stats_stronghold_occupation_time')
        
        # KOTH is same as stronghold in Halo Infinite
        koth_time = sh_time
        koth_ticks = sh_ticks
        
        # Extraction stats
        extract_success = safe_col_sum(player_df, 'extraction_stats_successful_extractions')
        extract_conv_complete = safe_col_sum(player_df, 'extraction_stats_extraction_conversions_completed')
        extract_conv_denied = safe_col_sum(player_df, 'extraction_stats_extraction_conversions_denied')
        extract_init_complete = safe_col_sum(player_df, 'extraction_stats_extraction_initiations_completed')
        extract_init_denied = safe_col_sum(player_df, 'extraction_stats_extraction_initiations_denied')
        
        # Average life
        avg_life = 0
        if 'average_life_duration' in player_df.columns:
            avg_life = pd.to_numeric(player_df['average_life_duration'], errors='coerce').fillna(0).mean()
        
        rows.append({
            'player': player,
            'games': format_int(games),
            # CTF
            'ctf_caps': format_int(ctf_caps),
            'ctf_grabs': format_int(ctf_grabs),
            'ctf_returns': format_int(ctf_returns),
            'ctf_steals': format_int(ctf_steals),
            'ctf_secures': format_int(ctf_secures),
            'ctf_carrier_kills': format_int(ctf_carrier_kills),
            'ctf_returner_kills': format_int(ctf_returner_kills),
            'ctf_kills_as_carrier': format_int(ctf_kills_as_carrier),
            'ctf_kills_as_returner': format_int(ctf_kills_as_returner),
            'ctf_time': format_int(ctf_time),
            # Oddball
            'oddball_time': format_int(oddball_time),
            'oddball_longest': format_int(oddball_longest),
            'oddball_grabs': format_int(oddball_grabs),
            'oddball_ticks': format_int(oddball_ticks),
            'oddball_carrier_kills': format_int(oddball_carrier_kills),
            'oddball_carriers_killed': format_int(oddball_carriers_killed),
            # Stronghold
            'sh_caps': format_int(sh_caps),
            'sh_secures': format_int(sh_secures),
            'sh_ticks': format_int(sh_ticks),
            'sh_off_kills': format_int(sh_off_kills),
            'sh_def_kills': format_int(sh_def_kills),
            'sh_time': format_int(sh_time),
            # KOTH
            'koth_time': format_int(koth_time),
            'koth_ticks': format_int(koth_ticks),
            # Extraction
            'extract_success': format_int(extract_success),
            'extract_conv_complete': format_int(extract_conv_complete),
            'extract_conv_denied': format_int(extract_conv_denied),
            'extract_init_complete': format_int(extract_init_complete),
            'extract_init_denied': format_int(extract_init_denied),
            # Misc
            'avg_life': format_float(avg_life, 1)
        })
    
    add_heatmap_classes(rows, {
        'games': True,
        'ctf_caps': True, 'ctf_grabs': True, 'ctf_returns': True, 'ctf_steals': True,
        'ctf_secures': True, 'ctf_carrier_kills': True, 'ctf_kills_as_carrier': True, 'ctf_time': True,
        'oddball_time': True, 'oddball_longest': True, 'oddball_grabs': True, 
        'oddball_ticks': True, 'oddball_carrier_kills': True,
        'sh_caps': True, 'sh_secures': True, 'sh_ticks': True,
        'sh_off_kills': True, 'sh_def_kills': True, 'sh_time': True,
        'koth_time': True, 'koth_ticks': True,
        'extract_success': True, 'extract_conv_complete': True, 'extract_init_complete': True,
        'avg_life': True
    })
    
    return rows


def build_medal_matrix(
    df: pd.DataFrame,
    players: list[str],
    medal_cols: list[str],
    per_game: bool
) -> list[dict]:
    rows = []
    for col in medal_cols:
        medal_name = col.replace('medal_', '').replace('_', ' ').title()
        row = {'medal': medal_name}
        values = []
        for player in players:
            player_df = df[df['player_gamertag'] == player]
            total = pd.to_numeric(player_df.get(col, 0), errors='coerce').fillna(0).sum()
            if per_game:
                games = len(player_df)
                value = total / games if games else 0
                row[player] = format_float(value, 2)
            else:
                value = total
                row[player] = format_int(value)
            values.append(value)
        for player, value in zip(players, values):
            row[f'{player}_heat'] = get_heatmap_class(value, values, True)
        rows.append(row)
    return rows


def build_medal_stats(df: pd.DataFrame) -> tuple[list[str], list[dict], list[dict]]:
    """Build medal statistics - returns (players, per_game_rows, total_rows)."""
    if df.empty:
        return [], [], []
    
    players = unique_sorted(df['player_gamertag'])
    if not players:
        return [], [], []
    
    medal_cols = [
        col for col in df.columns
        if col.startswith('medal_') and col != 'medal_count'
    ]
    
    if not medal_cols:
        return players, [], []
    
    medal_totals = []
    for col in medal_cols:
        total = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
        if total > 0:
            medal_totals.append((col, total))
    
    medal_totals.sort(key=lambda item: item[1], reverse=True)
    top_cols = [col for col, _ in medal_totals[:50]]
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    per_game_rows = build_medal_matrix(ranked_df, players, top_cols, per_game=True)
    total_rows = build_medal_matrix(df, players, top_cols, per_game=False)
    
    return players, per_game_rows, total_rows


def build_highlight_games(df: pd.DataFrame, limit: int = 20) -> list:
    """Build list of highlight/best games."""
    if df.empty:
        return []
    
    highlights = []
    
    # Add KDA column for sorting
    if 'kills' in df.columns and 'deaths' in df.columns and 'assists' in df.columns:
        kills = pd.to_numeric(df['kills'], errors='coerce').fillna(0)
        deaths = pd.to_numeric(df['deaths'], errors='coerce').fillna(0).replace(0, 1)
        assists = pd.to_numeric(df['assists'], errors='coerce').fillna(0)
        df['_kda_score'] = kills + assists / 3 - deaths
    
    # Top KDA games
    if '_kda_score' in df.columns:
        top_kda = df.nlargest(limit, '_kda_score')
        
        for _, row in top_kda.iterrows():
            highlights.append({
                'player': row.get('player_gamertag', 'Unknown'),
                'date': format_date(row.get('date')),
                'playlist': row.get('playlist', ''),
                'game_type': row.get('game_type', ''),
                'map': normalize_map_name(row.get('map', '')),
                'outcome': str(row.get('outcome', '')).title(),
                'outlier_count': 0,
                'outlier_count_heat': '',
                'outlier_score': 0,
                'outlier_score_heat': '',
                'kills': format_int(row.get('kills', 0)),
                'kills_heat': '',
                'deaths': format_int(row.get('deaths', 0)),
                'deaths_heat': '',
                'assists': format_int(row.get('assists', 0)),
                'assists_heat': '',
                'kda': format_float(row.get('_kda_score', 0), 2),
                'kda_heat': '',
                'accuracy': format_float(row.get('accuracy', 0), 2),
                'accuracy_heat': '',
                'score': format_int(row.get('personal_score', 0)),
                'score_heat': '',
                'dmg_min': format_float(row.get('dmg/min', 0), 1),
                'dmg_min_heat': '',
                'dmg_diff': format_int(row.get('dmg_difference', 0)),
                'dmg_diff_heat': '',
                'medals': format_int(row.get('medal_count', 0)),
                'medals_heat': ''
            })
    
    return highlights[:limit]


def build_hall_fame_shame(df: pd.DataFrame) -> tuple[list, list]:
    """Build hall of fame and hall of shame."""
    if df.empty:
        return [], []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    if ranked_df.empty:
        return [], []
    
    fame_rows = []
    shame_rows = []
    
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        win_streak, loss_streak = compute_streaks(player_df)
        
        kills = numeric_series(player_df, 'kills')
        deaths = numeric_series(player_df, 'deaths')
        assists = numeric_series(player_df, 'assists')
        kd_series = pd.Series(0.0, index=player_df.index)
        nonzero = deaths > 0
        kd_series.loc[nonzero] = kills[nonzero] / deaths[nonzero]
        kd_series.loc[~nonzero] = kills[~nonzero]
        kda_series = kills + assists / 3 - deaths
        
        if 'shots_fired' in player_df.columns and 'shots_hit' in player_df.columns:
            fired = numeric_series(player_df, 'shots_fired')
            hit = numeric_series(player_df, 'shots_hit')
            accuracy = pd.Series(0.0, index=player_df.index)
            nonzero = fired > 0
            accuracy.loc[nonzero] = hit[nonzero] / fired[nonzero] * 100
        else:
            accuracy = numeric_series(player_df, 'accuracy')
        damage_dealt = numeric_series(player_df, 'damage_dealt')
        damage_taken = numeric_series(player_df, 'damage_taken')
        damage_diff = damage_dealt - damage_taken
        score_vals = score_series(player_df)
        obj_score = objective_score_series(player_df)
        
        medals = numeric_series(player_df, 'medal_count')
        headshots = numeric_series(player_df, 'headshot_kills')
        grenades = numeric_series(player_df, 'grenade_kills')
        melee = numeric_series(player_df, 'melee_kills')
        power = numeric_series(player_df, 'power_weapon_kills')
        callouts = numeric_series(player_df, 'callout_assists')
        fired = numeric_series(player_df, 'shots_fired')
        landed = numeric_series(player_df, 'shots_hit')
        objectives = numeric_series(player_df, 'objectives_completed')
        spree = numeric_series(player_df, 'max_killing_spree')
        avg_life = numeric_series(player_df, 'average_life_duration')
        betrayals = numeric_series(player_df, 'betrayals')
        suicides = numeric_series(player_df, 'suicides')
        
        csr_delta = pd.Series(dtype=float)
        if 'pre_match_csr' in player_df.columns and 'post_match_csr' in player_df.columns:
            pre = pd.to_numeric(player_df['pre_match_csr'], errors='coerce').fillna(0)
            post = pd.to_numeric(player_df['post_match_csr'], errors='coerce').fillna(0)
            csr_delta = post - pre
        
        fame_rows.append({
            'player': player,
            'win_streak': format_int(win_streak),
            'max_kills': format_int(kills.max() if not kills.empty else 0),
            'max_assists': format_int(assists.max() if not assists.empty else 0),
            'max_kda': format_float(kda_series.max() if not kda_series.empty else 0, 2),
            'max_kd': format_float(kd_series.max() if not kd_series.empty else 0, 2),
            'max_accuracy': format_float(accuracy.max() if not accuracy.empty else 0, 1),
            'max_damage_dealt': format_int(damage_dealt.max() if not damage_dealt.empty else 0),
            'max_damage_diff': format_signed(damage_diff.max() if not damage_diff.empty else 0, 0),
            'max_score': format_int(score_vals.max() if not score_vals.empty else 0),
            'max_obj_score': format_float(obj_score.max() if not obj_score.empty else 0, 1),
            'max_medals': format_int(medals.max() if not medals.empty else 0),
            'max_headshots': format_int(headshots.max() if not headshots.empty else 0),
            'max_grenades': format_int(grenades.max() if not grenades.empty else 0),
            'max_melee': format_int(melee.max() if not melee.empty else 0),
            'max_power': format_int(power.max() if not power.empty else 0),
            'max_callouts': format_int(callouts.max() if not callouts.empty else 0),
            'max_fired': format_int(fired.max() if not fired.empty else 0),
            'max_landed': format_int(landed.max() if not landed.empty else 0),
            'max_objectives': format_int(objectives.max() if not objectives.empty else 0),
            'max_spree': format_int(spree.max() if not spree.empty else 0),
            'max_avg_life': format_float(avg_life.max() if not avg_life.empty else 0, 1),
            'max_csr_gain': format_signed(csr_delta.max() if not csr_delta.empty else 0, 0)
        })
        
        shame_rows.append({
            'player': player,
            'loss_streak': format_int(loss_streak),
            'max_deaths': format_int(deaths.max() if not deaths.empty else 0),
            'min_kda': format_float(kda_series.min() if not kda_series.empty else 0, 2),
            'min_kd': format_float(kd_series.min() if not kd_series.empty else 0, 2),
            'min_accuracy': format_float(accuracy.min() if not accuracy.empty else 0, 1),
            'max_damage_taken': format_int(damage_taken.max() if not damage_taken.empty else 0),
            'min_damage_diff': format_signed(damage_diff.min() if not damage_diff.empty else 0, 0),
            'min_score': format_int(score_vals.min() if not score_vals.empty else 0),
            'min_obj_score': format_float(obj_score.min() if not obj_score.empty else 0, 1),
            'min_medals': format_int(medals.min() if not medals.empty else 0),
            'min_avg_life': format_float(avg_life.min() if not avg_life.empty else 0, 1),
            'max_csr_loss': format_signed(csr_delta.min() if not csr_delta.empty else 0, 0),
            'max_betrayals': format_int(betrayals.max() if not betrayals.empty else 0),
            'max_suicides': format_int(suicides.max() if not suicides.empty else 0)
        })
    
    add_heatmap_classes(fame_rows, {
        'win_streak': True, 'max_kills': True, 'max_assists': True,
        'max_kda': True, 'max_kd': True, 'max_accuracy': True,
        'max_damage_dealt': True, 'max_damage_diff': True, 'max_score': True,
        'max_obj_score': True, 'max_medals': True, 'max_headshots': True,
        'max_grenades': True, 'max_melee': True, 'max_power': True,
        'max_callouts': True, 'max_fired': True, 'max_landed': True,
        'max_objectives': True, 'max_spree': True, 'max_avg_life': True,
        'max_csr_gain': True
    })
    
    add_heatmap_classes(shame_rows, {
        'loss_streak': True, 'max_deaths': True, 'min_kda': False,
        'min_kd': False, 'min_accuracy': False, 'max_damage_taken': True,
        'min_damage_diff': False, 'min_score': False, 'min_obj_score': False,
        'min_medals': False, 'min_avg_life': False, 'max_csr_loss': False,
        'max_betrayals': True, 'max_suicides': True
    })
    
    return fame_rows, shame_rows


def build_map_stats(df: pd.DataFrame) -> list:
    """Build detailed map statistics."""
    if df.empty or 'map' not in df.columns:
        return []
    
    working = add_normalized_map_column(df)
    
    rows = []
    for map_name in unique_sorted(working['_map_normalized']):
        if not map_name:
            continue
        
        map_df = working[working['_map_normalized'] == map_name]
        if map_df.empty:
            continue
        
        games = len(map_df)
        outcomes = map_df['outcome'].astype(str).str.lower() if 'outcome' in map_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        kills = numeric_series(map_df, 'kills')
        deaths = numeric_series(map_df, 'deaths')
        assists = numeric_series(map_df, 'assists')
        kda_series = kills + assists / 3 - deaths
        
        avg_kills = kills.mean() if games else 0
        avg_deaths = deaths.mean() if games else 0
        avg_kda = kda_series.mean() if games else 0
        
        rows.append({
            'map': map_name,
            'games': format_int(games),
            'wins': format_int(wins),
            'win_pct': format_float(wins / games * 100 if games else 0, 1),
            'avg_kills': format_float(avg_kills, 1),
            'avg_deaths': format_float(avg_deaths, 1),
            'avg_kda': format_float(avg_kda, 2)
        })
    
    add_heatmap_classes(rows, {
        'games': True, 'win_pct': True, 'avg_kills': True,
        'avg_deaths': False, 'avg_kda': True
    })
    rows.sort(key=lambda x: to_number(x['games']) or 0, reverse=True)
    
    return rows


def build_mode_stats(df: pd.DataFrame) -> list:
    """Build detailed mode statistics."""
    if df.empty or 'game_type' not in df.columns:
        return []
    
    rows = []
    for mode_name in unique_sorted(df['game_type']):
        if not mode_name:
            continue
        
        mode_df = df[df['game_type'] == mode_name]
        games = len(mode_df)
        if games == 0:
            continue
        
        outcomes = mode_df['outcome'].astype(str).str.lower() if 'outcome' in mode_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        kills = numeric_series(mode_df, 'kills')
        deaths = numeric_series(mode_df, 'deaths')
        assists = numeric_series(mode_df, 'assists')
        kda_series = kills + assists / 3 - deaths
        avg_kda = kda_series.mean() if games else 0
        
        score_vals = score_series(mode_df)
        avg_score = score_vals.mean() if not score_vals.empty else 0
        
        rows.append({
            'mode': mode_name,
            'games': format_int(games),
            'win_pct': format_float(wins / games * 100 if games else 0, 1),
            'avg_kda': format_float(avg_kda, 2),
            'avg_score': format_float(avg_score, 0)
        })
    
    add_heatmap_classes(rows, {
        'games': True, 'win_pct': True, 'avg_kda': True, 'avg_score': True
    })
    
    rows.sort(key=lambda x: to_number(x['games']) or 0, reverse=True)
    return rows


def build_player_map_stats(df: pd.DataFrame) -> list:
    """Build per-player map performance."""
    if df.empty or 'map' not in df.columns:
        return []
    
    working = add_normalized_map_column(df)
    
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        
        for map_name in unique_sorted(player_df['_map_normalized']):
            if not map_name:
                continue
            
            map_df = player_df[player_df['_map_normalized'] == map_name]
            games = len(map_df)
            
            if games < 3:  # Skip maps with few games
                continue
            
            outcomes = map_df['outcome'].astype(str).str.lower() if 'outcome' in map_df.columns else pd.Series()
            wins = (outcomes == 'win').sum() if not outcomes.empty else 0
            
            kills = numeric_series(map_df, 'kills')
            deaths = numeric_series(map_df, 'deaths')
            assists = numeric_series(map_df, 'assists')
            kda_series = kills + assists / 3 - deaths
            avg_kda = kda_series.mean() if games else 0
            
            rows.append({
                'player': player,
                'map': map_name,
                'games': format_int(games),
                'win_pct': format_float(wins / games * 100 if games else 0, 1),
                'avg_kda': format_float(avg_kda, 2)
            })
    
    add_heatmap_classes(rows, {'win_pct': True, 'avg_kda': True})
    rows.sort(key=lambda x: (x['player'], -to_number(x['win_pct']) or 0))
    
    return rows


def build_trend_data(df: pd.DataFrame, stat_col: str, stat_name: str) -> dict:
    """Build generic trend data for a statistic."""
    if df.empty or 'date' not in df.columns or stat_col not in df.columns:
        return {}
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    
    try:
        working['date_local'] = working['date'].dt.tz_convert(TIMEZONE)
    except:
        working['date_local'] = working['date']
    
    working['date_str'] = working['date_local'].dt.strftime('%Y-%m-%d')
    working[stat_col] = pd.to_numeric(working[stat_col], errors='coerce').fillna(0)
    
    trends = {}
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player].sort_values('date')
        
        daily = player_df.groupby('date_str')[stat_col].mean().reset_index()
        
        trends[player] = [
            {'date': row['date_str'], stat_name: float(row[stat_col])}
            for _, row in daily.iterrows()
        ]
    
    return trends


def build_win_rate_trends(df: pd.DataFrame) -> dict:
    """Build cumulative win rate trends per player."""
    if df.empty or 'date' not in df.columns:
        return {}
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return {}
    
    try:
        working['date_local'] = working['date'].dt.tz_convert(TIMEZONE)
    except:
        working['date_local'] = working['date']
    
    working['date_key'] = working['date_local'].dt.normalize()
    
    if 'outcome' in working.columns:
        outcome_lower = working['outcome'].astype(str).str.lower()
        working['win_flag'] = (outcome_lower == 'win').astype(int)
    else:
        working['win_flag'] = 0
    working['game_flag'] = 1
    
    trends = {}
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        daily = player_df.groupby('date_key')[['win_flag', 'game_flag']].sum().sort_index()
        if daily.empty:
            trends[player] = []
            continue
        
        cumulative = daily[['win_flag', 'game_flag']].cumsum()
        win_rate = (cumulative['win_flag'] / cumulative['game_flag'] * 100).fillna(0)
        
        trends[player] = [
            {'date': idx.strftime('%Y-%m-%d'), 'win_rate': float(value)}
            for idx, value in win_rate.items()
        ]
    
    return trends


def build_activity_heatmap(df: pd.DataFrame) -> list[dict]:
    """Build activity heatmap data by weekday and hour."""
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return []
    
    try:
        working['date_local'] = working['date'].dt.tz_convert(TIMEZONE)
    except Exception:
        working['date_local'] = working['date']
    
    working['day_idx'] = working['date_local'].dt.dayofweek
    working['hour'] = working['date_local'].dt.hour
    
    counts = working.groupby(['day_idx', 'hour']).size().to_dict()
    max_count = max(counts.values()) if counts else 0
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    rows = []
    for day_idx, day_name in enumerate(day_names):
        hours = []
        for hour in range(24):
            count = int(counts.get((day_idx, hour), 0))
            if max_count > 0 and count > 0:
                intensity = 0.1 + 0.9 * (count / max_count)
            else:
                intensity = 0
            hours.append({
                'hour': hour,
                'count': count,
                'intensity': round(intensity, 3)
            })
        rows.append({'day': day_name, 'hours': hours})
    
    return rows


def build_win_corr(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    """Build win correlation rows for stats vs win flag."""
    if df.empty or 'outcome' not in df.columns:
        return []
    
    working = df.copy()
    outcome_lower = working['outcome'].astype(str).str.lower()
    mask = outcome_lower.isin(['win', 'loss'])
    working = working[mask]
    outcome_lower = outcome_lower[mask]
    
    if working.empty:
        return []
    
    win_flag = (outcome_lower == 'win').astype(int)
    if win_flag.nunique() < 2:
        return []
    
    rows = []
    
    def add_corr(label: str, series: pd.Series) -> None:
        values = pd.to_numeric(series, errors='coerce')
        if values.dropna().empty or values.nunique(dropna=True) < 2:
            return
        corr = values.corr(win_flag)
        if pd.isna(corr):
            return
        rows.append({'stat': label, 'corr': float(corr)})
    
    if 'kills' in working.columns:
        add_corr('Kills', working['kills'])
    if 'deaths' in working.columns:
        add_corr('Deaths', working['deaths'])
    if 'assists' in working.columns:
        add_corr('Assists', working['assists'])
    if 'kda' in working.columns:
        add_corr('KDA', working['kda'])
    
    if 'shots_fired' in working.columns and 'shots_hit' in working.columns:
        fired = pd.to_numeric(working['shots_fired'], errors='coerce')
        hit = pd.to_numeric(working['shots_hit'], errors='coerce')
        acc = pd.Series(0.0, index=working.index)
        nonzero = fired > 0
        acc.loc[nonzero] = hit[nonzero] / fired[nonzero] * 100
        add_corr('Accuracy', acc)
    elif 'accuracy' in working.columns:
        acc = pd.to_numeric(working['accuracy'], errors='coerce')
        if acc.dropna().max() <= 1:
            acc = acc * 100
        add_corr('Accuracy', acc)
    
    if 'damage_dealt' in working.columns:
        add_corr('Damage Dealt', working['damage_dealt'])
    if 'damage_taken' in working.columns:
        add_corr('Damage Taken', working['damage_taken'])
    
    if 'dmg/min' in working.columns:
        add_corr('DMG/Min', working['dmg/min'])
    elif 'damage_dealt' in working.columns and 'duration' in working.columns:
        damage_dealt = pd.to_numeric(working['damage_dealt'], errors='coerce')
        duration = pd.to_numeric(working['duration'], errors='coerce')
        dmg_per_min = pd.Series(0.0, index=working.index)
        nonzero = duration > 0
        dmg_per_min.loc[nonzero] = damage_dealt[nonzero] / (duration[nonzero] / 60.0)
        add_corr('DMG/Min', dmg_per_min)
    
    if 'dmg_difference' in working.columns:
        add_corr('Damage Diff', working['dmg_difference'])
    elif 'damage_dealt' in working.columns and 'damage_taken' in working.columns:
        damage_dealt = pd.to_numeric(working['damage_dealt'], errors='coerce')
        damage_taken = pd.to_numeric(working['damage_taken'], errors='coerce')
        add_corr('Damage Diff', damage_dealt - damage_taken)
    
    score_vals = score_series(working)
    if not score_vals.empty:
        add_corr('Personal Score', score_vals)
    
    obj_scores = objective_score_series(working)
    if not obj_scores.empty:
        add_corr('Objective Score', obj_scores)
    
    if 'medal_count' in working.columns:
        add_corr('Medals', working['medal_count'])
    if 'headshot_kills' in working.columns:
        add_corr('Headshots', working['headshot_kills'])
    if 'melee_kills' in working.columns:
        add_corr('Melee Kills', working['melee_kills'])
    if 'grenade_kills' in working.columns:
        add_corr('Grenade Kills', working['grenade_kills'])
    if 'power_weapon_kills' in working.columns:
        add_corr('Power Weapon Kills', working['power_weapon_kills'])
    if 'callout_assists' in working.columns:
        add_corr('Callouts', working['callout_assists'])
    if 'average_life_duration' in working.columns:
        add_corr('Avg Life', working['average_life_duration'])
    
    rows.sort(key=lambda item: abs(item['corr']), reverse=True)
    return rows[:limit]


def build_player_moments(df: pd.DataFrame) -> dict:
    if df.empty or 'player_gamertag' not in df.columns or 'date' not in df.columns:
        return {}
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return {}
    
    if 'date_local' not in working.columns:
        try:
            working['date_local'] = working['date'].dt.tz_convert(TIMEZONE)
        except Exception:
            working['date_local'] = working['date']
    if 'date_str' not in working.columns:
        working['date_str'] = working['date_local'].dt.strftime('%Y-%m-%d')
    
    if 'kda' not in working.columns:
        kills = numeric_series(working, 'kills')
        deaths = numeric_series(working, 'deaths')
        assists = numeric_series(working, 'assists')
        working['kda'] = kills + assists / 3 - deaths
    if 'win_rate' not in working.columns and 'outcome' in working.columns:
        outcome_lower = working['outcome'].astype(str).str.lower()
        working['win_rate'] = (outcome_lower == 'win').astype(float) * 100
    if 'obj_score' not in working.columns:
        working['obj_score'] = objective_score_series(working)
    if 'dmg_diff' not in working.columns:
        working['dmg_diff'] = numeric_series(working, 'damage_dealt') - numeric_series(working, 'damage_taken')
    if 'dmg_min' not in working.columns:
        damage_dealt = numeric_series(working, 'damage_dealt')
        duration = numeric_series(working, 'duration')
        duration_min = duration / 60.0
        working['dmg_min'] = 0.0
        nonzero = duration_min > 0
        working.loc[nonzero, 'dmg_min'] = damage_dealt[nonzero] / duration_min[nonzero]
    
    daily_player = (
        working.groupby(['date_str', 'player_gamertag'])
        .agg(
            win_rate=('win_rate', 'mean'),
            kda=('kda', 'mean'),
            obj_score=('obj_score', 'mean'),
            dmg_min=('dmg_min', 'mean'),
            dmg_diff=('dmg_diff', 'mean')
        )
        .reset_index()
    )
    
    heroics_by_player = {}
    for date_str, group in daily_player.groupby('date_str'):
        if group.empty:
            continue
        temp = group.copy()
        temp['kda_rank'] = temp['kda'].rank(method='min', ascending=False)
        temp['dmg_rank'] = temp['dmg_diff'].rank(method='min', ascending=False)
        heroics = temp[(temp['kda_rank'] <= 3) & (temp['dmg_rank'] <= 3)]
        for _, row in heroics.iterrows():
            heroics_by_player.setdefault(row['player_gamertag'], []).append((date_str, row['kda']))
    
    moments = {}
    limits = {
        'tilt': 3,
        'tilt_window': 10,
        'clutch': 4,
        'carry': 5,
        'heroic': 5,
        'objective': 5,
        'silent': 5,
        'momentum': 3,
        'rivalry': 6
    }
    
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player].sort_values('date')
        if player_df.empty:
            continue
        
        daily_stats = daily_player[daily_player['player_gamertag'] == player].set_index('date_str')
        events = []
        counts = {key: 0 for key in limits}
        seen = set()
        
        def lookup_value(date_key: str, stat: str, fallback: float | None) -> float | None:
            if date_key in daily_stats.index and stat in daily_stats.columns:
                val = daily_stats.at[date_key, stat]
                if pd.notna(val):
                    return float(val)
            if fallback is None or pd.isna(fallback):
                return None
            return float(fallback)
        
        def add_event(date_key: str, stat: str, label: str, event_type: str, value: float | None) -> None:
            if not date_key or event_type not in limits:
                return
            if counts[event_type] >= limits[event_type]:
                return
            if value is None or pd.isna(value):
                return
            key = (date_key, stat, label, event_type)
            if key in seen:
                return
            seen.add(key)
            events.append({
                'date': date_key,
                'stat': stat,
                'label': label,
                'type': event_type,
                'value': float(value)
            })
            counts[event_type] += 1
        
        outcomes = player_df['outcome'].astype(str).str.lower().tolist() if 'outcome' in player_df.columns else []
        dates = player_df['date_str'].tolist()
        for i in range(len(outcomes) - 2):
            if counts['tilt'] >= limits['tilt']:
                break
            if outcomes[i] == outcomes[i + 1] == outcomes[i + 2] == 'loss' and (i == 0 or outcomes[i - 1] != 'loss'):
                date_key = dates[i]
                value = lookup_value(date_key, 'win_rate', None)
                add_event(date_key, 'win_rate', 'Tilt start (3L)', 'tilt', value)
                for j in range(1, 6):
                    if i + j >= len(dates):
                        break
                    date_window = dates[i + j]
                    value_window = lookup_value(date_window, 'win_rate', None)
                    add_event(date_window, 'win_rate', 'Tilt window', 'tilt_window', value_window)
        
        if 'team_score' in player_df.columns and 'enemy_team_score' in player_df.columns and 'outcome' in player_df.columns:
            team_score = pd.to_numeric(player_df['team_score'], errors='coerce').fillna(0)
            enemy_score = pd.to_numeric(player_df['enemy_team_score'], errors='coerce').fillna(0)
            close_mask = (team_score - enemy_score).abs() <= 5
            close_df = player_df[close_mask].copy()
            if not close_df.empty:
                close_outcome = close_df['outcome'].astype(str).str.lower()
                close_df = close_df[close_outcome.isin(['win', 'loss'])]
                if not close_df.empty:
                    close_df['win_flag'] = (close_df['outcome'].astype(str).str.lower() == 'win').astype(int)
                    daily_close = close_df.groupby('date_str')['win_flag'].agg(['mean', 'count']).reset_index()
                    daily_close['win_rate'] = daily_close['mean'] * 100
                    daily_close['date_dt'] = pd.to_datetime(daily_close['date_str'], errors='coerce')
                    daily_close = daily_close.dropna(subset=['date_dt']).sort_values('date_dt')
                    daily_close = daily_close.set_index('date_dt')
                    daily_close['rolling'] = daily_close['win_rate'].rolling('30D', min_periods=1).mean()
                    daily_close = daily_close.reset_index()
                    for _, row in daily_close.iterrows():
                        if counts['clutch'] >= limits['clutch']:
                            break
                        if row['count'] < 2:
                            continue
                        diff = row['win_rate'] - row['rolling']
                        if diff >= 20:
                            label = f'Clutch spike (+{diff:.0f}%)'
                            value = lookup_value(row['date_str'], 'win_rate', row['win_rate'])
                            add_event(row['date_str'], 'win_rate', label, 'clutch', value)
        
        if 'team_damage_dealt' in player_df.columns and 'damage_dealt' in player_df.columns:
            team_damage = pd.to_numeric(player_df['team_damage_dealt'], errors='coerce').fillna(0)
            damage = pd.to_numeric(player_df['damage_dealt'], errors='coerce').fillna(0)
            share = pd.Series(0.0, index=player_df.index)
            nonzero = team_damage > 0
            share.loc[nonzero] = damage[nonzero] / team_damage[nonzero]
            carry_df = player_df[share >= 0.45].copy()
            if not carry_df.empty:
                carry_df['share'] = share.loc[carry_df.index]
                carry_df = carry_df.sort_values('share', ascending=False).head(limits['carry'])
                for _, row in carry_df.iterrows():
                    label = f'Carry day ({row["share"] * 100:.0f}% team dmg)'
                    dmg_min = row.get('dmg_min')
                    value = lookup_value(row['date_str'], 'dmg_min', dmg_min)
                    add_event(row['date_str'], 'dmg_min', label, 'carry', value)
        
        for date_key, kda_val in heroics_by_player.get(player, [])[:limits['heroic']]:
            value = lookup_value(date_key, 'kda', kda_val)
            add_event(date_key, 'kda', 'Heroics day (top 3 KDA + dmg diff)', 'heroic', value)
        
        obj_scores = pd.to_numeric(player_df.get('obj_score', 0), errors='coerce').fillna(0)
        obj_median = obj_scores[obj_scores > 0].median() if not obj_scores.empty else 0
        if obj_median and obj_median > 0:
            anchor_df = player_df[obj_scores >= (2 * obj_median)].copy()
            if not anchor_df.empty:
                anchor_df['obj_score'] = obj_scores.loc[anchor_df.index]
                anchor_df = anchor_df.sort_values('obj_score', ascending=False).head(limits['objective'])
                for _, row in anchor_df.iterrows():
                    label = f'Objective anchor ({row["obj_score"]:.0f})'
                    value = lookup_value(row['date_str'], 'obj_score', row['obj_score'])
                    add_event(row['date_str'], 'obj_score', label, 'objective', value)
        
        if 'callout_assists' in player_df.columns:
            kda_vals = pd.to_numeric(player_df.get('kda', 0), errors='coerce').fillna(0)
            threshold = kda_vals.quantile(0.9) if not kda_vals.empty else None
            if threshold is not None:
                silent_df = player_df[(pd.to_numeric(player_df['callout_assists'], errors='coerce').fillna(0) <= 0) & (kda_vals >= threshold)].copy()
                if not silent_df.empty:
                    silent_df['kda'] = kda_vals.loc[silent_df.index]
                    silent_df = silent_df.sort_values('kda', ascending=False).head(limits['silent'])
                    for _, row in silent_df.iterrows():
                        label = f'Silent assassin (KDA {row["kda"]:.2f})'
                        value = lookup_value(row['date_str'], 'kda', row['kda'])
                        add_event(row['date_str'], 'kda', label, 'silent', value)
        
        if outcomes:
            session_ids = []
            session = 0
            last_ts = None
            for ts in player_df['date']:
                if last_ts is not None and ts - last_ts > pd.Timedelta(minutes=30):
                    session += 1
                session_ids.append(session)
                last_ts = ts
            session_df_all = player_df.copy()
            session_df_all['session_id'] = session_ids
            for _, session_df in session_df_all.groupby('session_id'):
                if counts['momentum'] >= limits['momentum']:
                    break
                if len(session_df) < 6:
                    continue
                session_df = session_df.sort_values('date')
                mid = len(session_df) // 2
                first_half = session_df.iloc[:mid]
                second_half = session_df.iloc[mid:]
                first_wins = (first_half['outcome'].astype(str).str.lower() == 'win').sum()
                second_wins = (second_half['outcome'].astype(str).str.lower() == 'win').sum()
                first_rate = first_wins / len(first_half) * 100 if len(first_half) else 0
                second_rate = second_wins / len(second_half) * 100 if len(second_half) else 0
                if first_rate < 40 and second_rate > 60:
                    date_key = session_df.iloc[0]['date_str']
                    label = f'Momentum flip ({first_rate:.0f}% -> {second_rate:.0f}%)'
                    value = lookup_value(date_key, 'win_rate', None)
                    add_event(date_key, 'win_rate', label, 'momentum', value)
        
        if 'map' in player_df.columns and 'outcome' in player_df.columns:
            overall_games = len(player_df)
            if overall_games:
                overall_wins = (player_df['outcome'].astype(str).str.lower() == 'win').sum()
                overall_win_pct = overall_wins / overall_games * 100
                map_df = player_df.copy()
                map_df['_map_name'] = map_df['map'].map(normalize_map_name)
                map_stats = []
                for map_name, group in map_df.groupby('_map_name'):
                    if not map_name:
                        continue
                    games = len(group)
                    if games < 5:
                        continue
                    wins = (group['outcome'].astype(str).str.lower() == 'win').sum()
                    win_pct = wins / games * 100 if games else 0
                    diff = win_pct - overall_win_pct
                    if abs(diff) >= 20:
                        map_stats.append((map_name, diff))
                map_stats.sort(key=lambda item: abs(item[1]), reverse=True)
                for map_name, diff in map_stats[:2]:
                    map_rows = map_df[map_df['_map_name'] == map_name].sort_values('date').head(3)
                    for _, row in map_rows.iterrows():
                        label = f'Rivalry map {map_name} ({diff:+.0f}%)'
                        value = lookup_value(row['date_str'], 'win_rate', None)
                        add_event(row['date_str'], 'win_rate', label, 'rivalry', value)
        
        if events:
            moments[player] = events
    
    return moments


def build_lineup_stats(df: pd.DataFrame, stack_size: int, min_games: int = 5, limit: int = 15) -> list[dict]:
    if df.empty or 'match_id' not in df.columns or 'player_gamertag' not in df.columns:
        return []
    if 'team_id' not in df.columns:
        return []
    if stack_size < 2 or stack_size > 4:
        return []
    
    working = df.copy()
    lineup_totals = {}
    
    grouped = working.groupby(['match_id', 'team_id'])
    for _, group in grouped:
        players = unique_sorted(group['player_gamertag'])
        if len(players) < stack_size:
            continue
        outcome = str(group['outcome'].iloc[0]).strip().lower() if 'outcome' in group.columns else ''
        win_flag = 1 if outcome == 'win' else 0
        
        per_player = {}
        for player in players:
            player_rows = group[group['player_gamertag'] == player]
            if player_rows.empty:
                continue
            kills = numeric_series(player_rows, 'kills').sum()
            deaths = numeric_series(player_rows, 'deaths').sum()
            assists = numeric_series(player_rows, 'assists').sum()
            kda = safe_kda(kills, assists, deaths)
            
            fired = numeric_series(player_rows, 'shots_fired').sum()
            hit = numeric_series(player_rows, 'shots_hit').sum()
            if fired > 0:
                accuracy = hit / fired * 100
            else:
                accuracy = pd.to_numeric(player_rows.get('accuracy', 0), errors='coerce').fillna(0).mean()
                if accuracy <= 1:
                    accuracy *= 100
            
            obj_score = objective_score_series(player_rows).sum()
            dmg_diff = numeric_series(player_rows, 'damage_dealt').sum() - numeric_series(player_rows, 'damage_taken').sum()
            score = score_series(player_rows).sum()
            
            per_player[player] = {
                'kda': kda,
                'accuracy': accuracy,
                'obj_score': obj_score,
                'dmg_diff': dmg_diff,
                'score': score
            }
        
        if len(per_player) < stack_size:
            continue
        
        for lineup in combinations(sorted(per_player.keys()), stack_size):
            metrics = [per_player[player] for player in lineup]
            entry = lineup_totals.setdefault(lineup, {
                'games': 0,
                'wins': 0,
                'kda': 0.0,
                'accuracy': 0.0,
                'obj_score': 0.0,
                'dmg_diff': 0.0,
                'score': 0.0
            })
            entry['games'] += 1
            entry['wins'] += win_flag
            entry['kda'] += sum(item['kda'] for item in metrics) / stack_size
            entry['accuracy'] += sum(item['accuracy'] for item in metrics) / stack_size
            entry['obj_score'] += sum(item['obj_score'] for item in metrics) / stack_size
            entry['dmg_diff'] += sum(item['dmg_diff'] for item in metrics) / stack_size
            entry['score'] += sum(item['score'] for item in metrics) / stack_size
    
    rows = []
    for lineup, totals in lineup_totals.items():
        games = totals['games']
        if games < min_games:
            continue
        win_pct = totals['wins'] / games * 100 if games else 0
        rows.append({
            'players': list(lineup),
            'lineup': ' + '.join(lineup),
            'games': format_int(games),
            'wins': format_int(totals['wins']),
            'win_pct': format_float(win_pct, 1),
            'kda': format_float(totals['kda'] / games if games else 0, 2),
            'accuracy': format_float(totals['accuracy'] / games if games else 0, 1),
            'obj_score': format_float(totals['obj_score'] / games if games else 0, 1),
            'dmg_diff': format_signed(totals['dmg_diff'] / games if games else 0, 0),
            'score': format_float(totals['score'] / games if games else 0, 0)
        })
    
    add_heatmap_classes(rows, {
        'win_pct': True,
        'kda': True,
        'accuracy': True,
        'obj_score': True,
        'dmg_diff': True,
        'score': True
    })
    
    rows.sort(key=lambda r: (
        to_number(r.get('win_pct')) or 0,
        to_number(r.get('kda')) or 0,
        to_number(r.get('games')) or 0
    ), reverse=True)
    
    return rows[:limit]


def build_player_hover_data(df: pd.DataFrame) -> dict:
    if df.empty or 'player_gamertag' not in df.columns:
        return {}
    
    now = time.time()
    cached = PLAYER_HOVER_CACHE.get('payload')
    if cached and now - PLAYER_HOVER_CACHE['last_ts'] < PLAYER_HOVER_CACHE_TTL:
        return cached
    
    working = df.copy()
    if 'date' in working.columns:
        working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
        working = working.dropna(subset=['date'])
    if working.empty:
        return {}
    
    payload = {}
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        if player_df.empty:
            continue
        games = len(player_df)
        outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        win_pct = wins / games * 100 if games else 0
        
        kills = numeric_series(player_df, 'kills').sum()
        deaths = numeric_series(player_df, 'deaths').sum()
        assists = numeric_series(player_df, 'assists').sum()
        kda = safe_kda(kills / games if games else 0, assists / games if games else 0, deaths / games if games else 0)
        
        player_df = player_df.sort_values('date', ascending=True)
        csr_vals = extract_csr_values(player_df)
        current_csr = csr_vals.iloc[-1] if not csr_vals.empty else None
        last_match = player_df['date'].max() if 'date' in player_df.columns else None
        
        payload[player.lower()] = {
            'player': player,
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kda': format_float(kda, 2),
            'csr': format_float(current_csr, 1) if current_csr is not None and not pd.isna(current_csr) else '-',
            'last_match': format_date(last_match)
        }
    
    PLAYER_HOVER_CACHE['payload'] = payload
    PLAYER_HOVER_CACHE['last_ts'] = now
    return payload


def load_insights_cache() -> dict | None:
    try:
        if not INSIGHTS_CACHE_PATH.exists():
            return None
        with open(INSIGHTS_CACHE_PATH, 'r') as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return None
        created_ts = data.get('created_ts')
        payload = data.get('payload')
        if not isinstance(created_ts, (int, float)) or not isinstance(payload, dict):
            return None
        if INSIGHTS_CACHE_DISK_TTL > 0:
            age = time.time() - float(created_ts)
            if age > INSIGHTS_CACHE_DISK_TTL:
                return None
        return payload
    except Exception:
        return None


def save_insights_cache(payload: dict) -> None:
    try:
        data = {
            'created_ts': time.time(),
            'payload': payload
        }
        with open(INSIGHTS_CACHE_PATH, 'w') as file:
            json.dump(data, file)
    except Exception:
        return None


def get_insights_payload(ranked_df: pd.DataFrame) -> dict:
    if ranked_df.empty:
        return {
            'clutch_rows': [],
            'role_rows': [],
            'momentum_rows': [],
            'veto_rows': [],
            'consistency_rows': [],
            'notable_rows': [],
            'change_rows': [],
            'lineup2_rows': [],
            'lineup3_rows': [],
            'lineup4_rows': []
        }
    
    now = time.time()
    cached = INSIGHTS_CACHE.get('payload')
    if cached and now - INSIGHTS_CACHE['last_ts'] < INSIGHTS_CACHE_TTL:
        return cached
    
    disk_payload = load_insights_cache()
    if disk_payload:
        INSIGHTS_CACHE['payload'] = disk_payload
        INSIGHTS_CACHE['last_ts'] = now
        return disk_payload
    
    payload = {
        'clutch_rows': build_clutch_index(ranked_df),
        'role_rows': build_role_heatmap(ranked_df),
        'momentum_rows': build_momentum_rows(ranked_df),
        'veto_rows': build_map_veto_hints(ranked_df, MAP_VETO_MIN_GAMES),
        'consistency_rows': build_consistency_rows(ranked_df),
        'notable_rows': build_notable_games(ranked_df),
        'change_rows': build_change_summary(ranked_df),
        'lineup2_rows': build_lineup_stats(ranked_df, 2),
        'lineup3_rows': build_lineup_stats(ranked_df, 3),
        'lineup4_rows': build_lineup_stats(ranked_df, 4)
    }
    
    INSIGHTS_CACHE['payload'] = payload
    INSIGHTS_CACHE['last_ts'] = now
    save_insights_cache(payload)
    return payload


def parse_date_bound(value: str, is_end: bool) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(TIMEZONE)
        except Exception:
            ts = ts.tz_localize('UTC')
    if is_end:
        ts = ts + pd.Timedelta(days=1)
    try:
        return ts.tz_convert('UTC')
    except Exception:
        return ts


def apply_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns:
        return df
    start_ts = parse_date_bound(start, False)
    end_ts = parse_date_bound(end, True)
    working = df.copy()
    date_series = pd.to_datetime(working['date'], errors='coerce', utc=True)
    if start_ts is not None:
        working = working[date_series >= start_ts]
    if end_ts is not None:
        working = working[date_series < end_ts]
    return working


def summarize_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    games = len(df)
    outcomes = df['outcome'].astype(str).str.lower() if 'outcome' in df.columns else pd.Series()
    wins = (outcomes == 'win').sum() if not outcomes.empty else 0
    win_rate = wins / games * 100 if games else 0
    
    total_kills = numeric_series(df, 'kills').sum()
    total_deaths = numeric_series(df, 'deaths').sum()
    total_assists = numeric_series(df, 'assists').sum()
    kills_pg = total_kills / games if games else 0
    deaths_pg = total_deaths / games if games else 0
    assists_pg = total_assists / games if games else 0
    kda = safe_kda(kills_pg, assists_pg, deaths_pg)
    
    fired = numeric_series(df, 'shots_fired').sum()
    hit = numeric_series(df, 'shots_hit').sum()
    if fired > 0:
        accuracy = hit / fired * 100
    else:
        accuracy = numeric_series(df, 'accuracy').mean()
        if accuracy <= 1:
            accuracy *= 100
    
    damage_dealt = numeric_series(df, 'damage_dealt').sum()
    damage_taken = numeric_series(df, 'damage_taken').sum()
    dmg_diff_pg = (damage_dealt - damage_taken) / games if games else 0
    duration = numeric_series(df, 'duration').sum()
    dmg_per_min = damage_dealt / (duration / 60) if duration > 0 else 0
    
    score_total = score_series(df).sum()
    score_pg = score_total / games if games else 0
    
    obj_scores = objective_score_series(df)
    obj_score_pg = obj_scores.sum() / games if not obj_scores.empty and games else 0
    
    return {
        'games': games,
        'win_rate': win_rate,
        'kda': kda,
        'kills_pg': kills_pg,
        'deaths_pg': deaths_pg,
        'assists_pg': assists_pg,
        'accuracy': accuracy,
        'dmg_per_min': dmg_per_min,
        'dmg_diff_pg': dmg_diff_pg,
        'score_pg': score_pg,
        'obj_score_pg': obj_score_pg
    }


def build_session_compare(
    df: pd.DataFrame,
    player: str,
    start_a: str,
    end_a: str,
    start_b: str,
    end_b: str
) -> list[dict]:
    if df.empty:
        return []
    working = df
    if player and player != 'all' and 'player_gamertag' in working.columns:
        working = working[working['player_gamertag'] == player]
    if working.empty:
        return []
    
    range_a = apply_date_range(working, start_a, end_a)
    range_b = apply_date_range(working, start_b, end_b)
    
    stats_a = summarize_stats(range_a)
    stats_b = summarize_stats(range_b)
    
    def safe(value):
        return value if value is not None else 0
    
    rows = []
    for key, label, fmt, delta_fmt in [
        ('games', 'Games', lambda v: format_int(v), lambda v: format_signed(v, 0)),
        ('win_rate', 'Win %', lambda v: format_float(v, 1), lambda v: format_signed(v, 1)),
        ('kda', 'KDA', lambda v: format_float(v, 2), lambda v: format_signed(v, 2)),
        ('kills_pg', 'Kills/Game', lambda v: format_float(v, 1), lambda v: format_signed(v, 1)),
        ('deaths_pg', 'Deaths/Game', lambda v: format_float(v, 1), lambda v: format_signed(v, 1)),
        ('assists_pg', 'Assists/Game', lambda v: format_float(v, 1), lambda v: format_signed(v, 1)),
        ('accuracy', 'Accuracy %', lambda v: format_float(v, 1), lambda v: format_signed(v, 1)),
        ('dmg_per_min', 'Damage/Min', lambda v: format_float(v, 0), lambda v: format_signed(v, 0)),
        ('dmg_diff_pg', 'Damage Diff/Game', lambda v: format_signed(v, 0), lambda v: format_signed(v, 0)),
        ('score_pg', 'Score/Game', lambda v: format_float(v, 0), lambda v: format_signed(v, 0)),
        ('obj_score_pg', 'Obj Score/Game', lambda v: format_float(v, 1), lambda v: format_signed(v, 1))
    ]:
        a_val = safe(stats_a.get(key)) if stats_a else 0
        b_val = safe(stats_b.get(key)) if stats_b else 0
        delta = b_val - a_val
        rows.append({
            'stat': label,
            'a': fmt(a_val),
            'b': fmt(b_val),
            'delta': delta_fmt(delta)
        })
    
    return rows


def build_clutch_index(df: pd.DataFrame) -> list[dict]:
    if df.empty or 'player_gamertag' not in df.columns:
        return []
    
    if 'team_score' not in df.columns or 'enemy_team_score' not in df.columns:
        return []
    
    working = df.copy()
    team_score = pd.to_numeric(working['team_score'], errors='coerce').fillna(0)
    enemy_score = pd.to_numeric(working['enemy_team_score'], errors='coerce').fillna(0)
    working['score_diff'] = (team_score - enemy_score).abs()
    close_games = working[working['score_diff'] <= 5]
    
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        if player_df.empty:
            continue
        close_df = close_games[close_games['player_gamertag'] == player]
        close_count = len(close_df)
        if close_count == 0:
            continue
        
        close_outcomes = close_df['outcome'].astype(str).str.lower() if 'outcome' in close_df.columns else pd.Series()
        close_wins = (close_outcomes == 'win').sum() if not close_outcomes.empty else 0
        close_win_pct = close_wins / close_count * 100
        
        close_stats = summarize_stats(close_df)
        overall_stats = summarize_stats(player_df)
        clutch_index = close_win_pct - overall_stats.get('win_rate', 0)
        
        rows.append({
            'player': player,
            'close_games': format_int(close_count),
            'close_win_pct': format_float(close_win_pct, 1),
            'close_kda': format_float(close_stats.get('kda', 0), 2),
            'clutch_index': format_signed(clutch_index, 1)
        })
    
    add_heatmap_classes(rows, {
        'close_games': True,
        'close_win_pct': True,
        'close_kda': True,
        'clutch_index': True
    })
    rows.sort(key=lambda x: to_number(x.get('clutch_index')) or 0, reverse=True)
    return rows


def build_role_heatmap(df: pd.DataFrame) -> list[dict]:
    if df.empty or 'player_gamertag' not in df.columns:
        return []
    
    rows = []
    raw_scores = []
    for player in unique_sorted(df['player_gamertag']):
        player_df = df[df['player_gamertag'] == player]
        stats = summarize_stats(player_df)
        if not stats:
            continue
        
        games = stats.get('games', 1)
        kills_pg = stats.get('kills_pg', 0)
        dmg_pg = numeric_series(player_df, 'damage_dealt').sum() / games
        slayer_score = kills_pg + (dmg_pg / 1000)
        
        obj_score = stats.get('obj_score_pg', 0)
        assists_pg = stats.get('assists_pg', 0)
        callouts_pg = numeric_series(player_df, 'callout_assists').sum() / games
        support_score = assists_pg + callouts_pg
        
        raw_scores.append((player, slayer_score, obj_score, support_score))
    
    if not raw_scores:
        return []
    
    slayer_vals = [score[1] for score in raw_scores]
    obj_vals = [score[2] for score in raw_scores]
    support_vals = [score[3] for score in raw_scores]
    
    def normalize(value, values):
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    for player, slayer_score, obj_score, support_score in raw_scores:
        role_map = {
            'Slayer': normalize(slayer_score, slayer_vals),
            'Objective': normalize(obj_score, obj_vals),
            'Support': normalize(support_score, support_vals)
        }
        role = max(role_map, key=role_map.get)
        
        rows.append({
            'player': player,
            'slayer_score': format_float(slayer_score, 2),
            'obj_score': format_float(obj_score, 2),
            'support_score': format_float(support_score, 2),
            'role': role
        })
    
    add_heatmap_classes(rows, {
        'slayer_score': True,
        'obj_score': True,
        'support_score': True
    })
    return rows


def build_momentum_rows(df: pd.DataFrame, limit: int = 10) -> list[dict]:
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            continue
        
        recent = player_df.head(limit)
        results = []
        for _, row in recent.iterrows():
            outcome = str(row.get('outcome', '')).lower()
            if outcome == 'win':
                results.append({'label': 'W', 'class': 'streak-win'})
            elif outcome == 'loss':
                results.append({'label': 'L', 'class': 'streak-loss'})
            else:
                results.append({'label': '-', 'class': 'streak-tie'})
        
        pre = pd.to_numeric(recent.get('pre_match_csr', 0), errors='coerce')
        post = pd.to_numeric(recent.get('post_match_csr', 0), errors='coerce')
        pre = pre[pre > 0]
        post = post[post > 0]
        csr_delta = 0
        if not pre.empty and not post.empty:
            csr_delta = post.iloc[0] - pre.iloc[-1]
        
        rows.append({
            'player': player,
            'recent_results': results,
            'csr_delta': format_signed(csr_delta, 0),
            'last_played': format_date(recent['date'].max())
        })
    
    return rows


def build_map_veto_hints(df: pd.DataFrame, min_games: int = MAP_VETO_MIN_GAMES) -> list[dict]:
    if df.empty or 'map' not in df.columns:
        return []
    
    working = add_normalized_map_column(df)
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        entries = []
        for map_name in unique_sorted(player_df['_map_normalized']):
            if not map_name:
                continue
            map_df = player_df[player_df['_map_normalized'] == map_name]
            games = len(map_df)
            if games < min_games:
                continue
            outcomes = map_df['outcome'].astype(str).str.lower() if 'outcome' in map_df.columns else pd.Series()
            wins = (outcomes == 'win').sum() if not outcomes.empty else 0
            win_pct = wins / games * 100 if games else 0
            entries.append({'map': map_name, 'games': games, 'win_pct': win_pct})
        
        if not entries:
            continue
        
        entries.sort(key=lambda x: x['win_pct'], reverse=True)
        best = entries[0]
        worst = entries[-1]
        
        rows.append({
            'player': player,
            'best_map': best['map'],
            'best_win_pct': format_float(best['win_pct'], 1),
            'best_games': format_int(best['games']),
            'worst_map': worst['map'],
            'worst_win_pct': format_float(worst['win_pct'], 1),
            'worst_games': format_int(worst['games'])
        })
    
    add_heatmap_classes(rows, {'best_win_pct': True, 'worst_win_pct': False})
    return rows


def build_consistency_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty or 'player_gamertag' not in df.columns or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return []
    
    now = working['date'].max()
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        stats = {}
        for days, label in [(30, '30d'), (90, '90d')]:
            cutoff = now - pd.Timedelta(days=days)
            window_df = player_df[player_df['date'] >= cutoff]
            if window_df.empty:
                stats[f'csr_std_{label}'] = 0
                stats[f'kda_std_{label}'] = 0
                continue
            csr_vals = extract_csr_values(window_df)
            if csr_vals.empty:
                csr_vals = pd.to_numeric(window_df.get('pre_match_csr', 0), errors='coerce').dropna()
            kda_vals = pd.to_numeric(window_df.get('kda', pd.Series()), errors='coerce')
            if kda_vals.dropna().empty:
                kills = numeric_series(window_df, 'kills')
                deaths = numeric_series(window_df, 'deaths')
                assists = numeric_series(window_df, 'assists')
                kda_vals = kills + assists / 3 - deaths
            kda_vals = kda_vals.dropna()
            stats[f'csr_std_{label}'] = float(csr_vals.std(ddof=0)) if not csr_vals.empty else 0
            stats[f'kda_std_{label}'] = float(kda_vals.std(ddof=0)) if not kda_vals.empty else 0
        
        total_std = stats['csr_std_30d'] + stats['kda_std_30d'] + stats['csr_std_90d'] + stats['kda_std_90d']
        consistency = 100 / (1 + total_std) if total_std >= 0 else 0
        
        rows.append({
            'player': player,
            'csr_std_30d': format_float(stats['csr_std_30d'], 2),
            'kda_std_30d': format_float(stats['kda_std_30d'], 2),
            'csr_std_90d': format_float(stats['csr_std_90d'], 2),
            'kda_std_90d': format_float(stats['kda_std_90d'], 2),
            'consistency': format_float(consistency, 1)
        })
    
    add_heatmap_classes(rows, {
        'csr_std_30d': False,
        'kda_std_30d': False,
        'csr_std_90d': False,
        'kda_std_90d': False,
        'consistency': True
    })
    rows.sort(key=lambda x: to_number(x.get('consistency')) or 0, reverse=True)
    return rows


def build_notable_games(df: pd.DataFrame, limit: int = NOTABLE_GAMES_LIMIT) -> list[dict]:
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return []
    
    kills = numeric_series(working, 'kills')
    deaths = numeric_series(working, 'deaths')
    assists = numeric_series(working, 'assists')
    kda_series = kills + assists / 3 - deaths
    
    medals = numeric_series(working, 'medal_count')
    damage_diff = numeric_series(working, 'damage_dealt') - numeric_series(working, 'damage_taken')
    
    picks = []
    seen = set()
    
    def add_top(series: pd.Series, label: str, formatter, top_n: int = 5) -> None:
        if series.empty:
            return
        sorted_series = series.sort_values(ascending=False).head(top_n)
        for idx, value in sorted_series.items():
            row = working.loc[idx]
            match_id = row.get('match_id')
            key = match_id or (row.get('date'), row.get('player_gamertag'), label)
            if key in seen:
                continue
            seen.add(key)
            picks.append({
                'date': format_date(row.get('date')),
                'player': row.get('player_gamertag', ''),
                'map': row.get('map', ''),
                'mode': row.get('game_type', ''),
                'reason': f'{label} {formatter(value)}'
            })
    
    add_top(kda_series, 'KDA', lambda v: format_float(v, 2), top_n=5)
    add_top(medals, 'Medals', lambda v: format_int(v), top_n=5)
    add_top(damage_diff, 'Damage Swing', lambda v: format_signed(v, 0), top_n=5)
    
    picks.sort(key=lambda row: row.get('date', ''), reverse=True)
    return picks[:limit]


def build_change_summary(df: pd.DataFrame, days: int = 7) -> list[dict]:
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return []
    
    now = working['date'].max()
    recent_start = now - pd.Timedelta(days=days)
    prev_start = now - pd.Timedelta(days=days * 2)
    
    rows = []
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        recent_df = player_df[player_df['date'] >= recent_start]
        prev_df = player_df[(player_df['date'] < recent_start) & (player_df['date'] >= prev_start)]
        
        recent_stats = summarize_stats(recent_df)
        prev_stats = summarize_stats(prev_df)
        
        if not recent_stats and not prev_stats:
            continue
        
        win_delta = recent_stats.get('win_rate', 0) - prev_stats.get('win_rate', 0)
        kda_delta = recent_stats.get('kda', 0) - prev_stats.get('kda', 0)
        
        rows.append({
            'player': player,
            'recent_games': format_int(recent_stats.get('games', 0)),
            'prev_games': format_int(prev_stats.get('games', 0)),
            'win_delta': format_signed(win_delta, 1),
            'kda_delta': format_signed(kda_delta, 2),
            'win_delta_heat': 'heat-good' if win_delta > 0 else 'heat-poor' if win_delta < 0 else '',
            'kda_delta_heat': 'heat-good' if kda_delta > 0 else 'heat-poor' if kda_delta < 0 else ''
        })
    
    rows.sort(key=lambda r: abs(to_number(r.get('win_delta')) or 0) + abs(to_number(r.get('kda_delta')) or 0), reverse=True)
    return rows


def build_player_match_history(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date']).sort_values('date', ascending=False)
    if limit:
        working = working.head(limit)
    
    score_vals = score_series(working)
    if score_vals.empty:
        score_vals = pd.Series(0.0, index=working.index)
    
    rows = []
    for idx, row in working.iterrows():
        kills = safe_float(row.get('kills', 0))
        deaths = safe_float(row.get('deaths', 0))
        assists = safe_float(row.get('assists', 0))
        kda = safe_kda(kills, assists, deaths)
        
        fired = safe_float(row.get('shots_fired', 0))
        hit = safe_float(row.get('shots_hit', 0))
        accuracy = hit / fired * 100 if fired > 0 else safe_float(row.get('accuracy', 0))
        if accuracy <= 1:
            accuracy *= 100
        
        pre_csr = row.get('pre_match_csr')
        post_csr = row.get('post_match_csr')
        pre_val = safe_float(pre_csr)
        post_val = safe_float(post_csr)
        csr_delta = post_val - pre_val if pre_val and post_val else 0
        
        rows.append({
            'match_id': row.get('match_id', ''),
            'date': format_date(row.get('date')),
            'game_type': row.get('game_type', ''),
            'map': row.get('map', ''),
            'outcome': str(row.get('outcome', '')).title(),
            'outcome_class': outcome_class(row.get('outcome', '')),
            'kills': format_int(kills),
            'deaths': format_int(deaths),
            'assists': format_int(assists),
            'kda': format_float(kda, 2),
            'accuracy': format_float(accuracy, 1),
            'pre_csr': format_optional_int(pre_csr),
            'post_csr': format_optional_int(post_csr),
            'csr_delta': format_signed(csr_delta, 0),
            'damage_dealt': format_int(row.get('damage_dealt', 0)),
            'damage_taken': format_int(row.get('damage_taken', 0)),
            'dmg_diff': format_signed(
                safe_float(row.get('damage_dealt', 0)) - safe_float(row.get('damage_taken', 0)), 0
            ),
            'shots_fired': format_int(fired),
            'shots_hit': format_int(hit),
            'headshots': format_int(row.get('headshot_kills', 0)),
            'score': format_int(score_vals.loc[idx] if idx in score_vals.index else 0),
            'medals': format_int(row.get('medal_count', 0)),
            'avg_life': format_float(row.get('average_life_duration', 0), 1)
        })
    
    add_heatmap_classes(rows, {
        'kills': True, 'deaths': False, 'assists': True,
        'kda': True, 'accuracy': True,
        'damage_dealt': True, 'damage_taken': False, 'dmg_diff': True,
        'score': True, 'medals': True, 'avg_life': True,
        'csr_delta': True
    })
    
    return rows


def build_player_map_summary(df: pd.DataFrame) -> list[dict]:
    if df.empty or 'map' not in df.columns:
        return []
    
    working = add_normalized_map_column(df)
    rows = []
    for map_name in unique_sorted(working['_map_normalized']):
        if not map_name:
            continue
        map_df = working[working['_map_normalized'] == map_name]
        games = len(map_df)
        if games == 0:
            continue
        outcomes = map_df['outcome'].astype(str).str.lower() if 'outcome' in map_df.columns else pd.Series()
        wins = (outcomes == 'win').sum() if not outcomes.empty else 0
        
        kills = numeric_series(map_df, 'kills')
        deaths = numeric_series(map_df, 'deaths')
        assists = numeric_series(map_df, 'assists')
        kda_series = kills + assists / 3 - deaths
        
        rows.append({
            'map': map_name,
            'games': format_int(games),
            'win_pct': format_float(wins / games * 100 if games else 0, 1),
            'kda': format_float(kda_series.mean() if games else 0, 2)
        })
    
    add_heatmap_classes(rows, {'win_pct': True, 'kda': True})
    rows.sort(key=lambda x: to_number(x.get('games')) or 0, reverse=True)
    return rows[:10]


def build_teammate_stats(df: pd.DataFrame, player: str) -> list[dict]:
    if df.empty or 'match_id' not in df.columns or 'player_gamertag' not in df.columns:
        return []
    
    player_df = df[df['player_gamertag'] == player]
    if player_df.empty:
        return []
    
    match_players = df.groupby('match_id')['player_gamertag'].apply(set).to_dict()
    totals = {}
    
    for _, row in player_df.iterrows():
        match_id = row.get('match_id')
        if not match_id or match_id not in match_players:
            continue
        teammates = match_players[match_id] - {player}
        if not teammates:
            continue
        outcome = str(row.get('outcome', '')).lower()
        win = 1 if outcome == 'win' else 0
        kills = safe_float(row.get('kills', 0))
        deaths = safe_float(row.get('deaths', 0))
        assists = safe_float(row.get('assists', 0))
        for teammate in teammates:
            entry = totals.setdefault(teammate, {'games': 0, 'wins': 0, 'kills': 0, 'deaths': 0, 'assists': 0})
            entry['games'] += 1
            entry['wins'] += win
            entry['kills'] += kills
            entry['deaths'] += deaths
            entry['assists'] += assists
    
    rows = []
    for teammate, data in totals.items():
        games = data['games']
        if games == 0:
            continue
        win_pct = data['wins'] / games * 100
        kda = safe_kda(data['kills'] / games, data['assists'] / games, data['deaths'] / games)
        rows.append({
            'teammate': teammate,
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kda': format_float(kda, 2)
        })
    
    add_heatmap_classes(rows, {'win_pct': True, 'kda': True})
    rows.sort(key=lambda x: to_number(x.get('games')) or 0, reverse=True)
    return rows


def build_player_csr_history(df: pd.DataFrame, player: str) -> list[dict]:
    if df.empty or 'date' not in df.columns or 'player_gamertag' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    ranked_df = ranked_df[ranked_df['player_gamertag'] == player]
    if ranked_df.empty:
        return []
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], errors='coerce', utc=True)
    ranked_df = ranked_df.dropna(subset=['date'])
    if ranked_df.empty:
        return []
    
    try:
        ranked_df['date_local'] = ranked_df['date'].dt.tz_convert(TIMEZONE)
    except Exception:
        ranked_df['date_local'] = ranked_df['date']
    
    ranked_df['date_str'] = ranked_df['date_local'].dt.strftime('%Y-%m-%d')
    ranked_df['csr_value'] = extract_csr_values(ranked_df)
    ranked_df = ranked_df.dropna(subset=['csr_value'])
    if ranked_df.empty:
        return []
    
    daily = ranked_df.groupby('date_str')['csr_value'].last().reset_index()
    daily['date_key'] = pd.to_datetime(daily['date_str'], errors='coerce')
    daily = daily.sort_values('date_key')
    
    return [
        {'date': row['date_str'], 'csr': float(row['csr_value'])}
        for _, row in daily.iterrows()
    ]


def build_30day_overview(df: pd.DataFrame) -> dict:
    if df.empty or 'date' not in df.columns:
        return {}
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date'])
    if working.empty:
        return {}
    now = working['date'].max()
    cutoff = now - pd.Timedelta(days=30)
    working = working[working['date'] >= cutoff]
    if working.empty:
        return {}
    
    rows = {}
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player]
        stats = summarize_stats(player_df)
        if not stats:
            continue
        pre = pd.to_numeric(player_df.get('pre_match_csr', 0), errors='coerce').fillna(0)
        post = pd.to_numeric(player_df.get('post_match_csr', 0), errors='coerce').fillna(0)
        deltas = post - pre
        deltas = deltas[(pre > 0) & (post > 0)]
        avg_csr_change = deltas.mean() if not deltas.empty else 0
        
        rows[player] = {
            'games': stats.get('games', 0),
            'win_pct': stats.get('win_rate', 0),
            'kda': stats.get('kda', 0),
            'accuracy': stats.get('accuracy', 0),
            'avg_csr_change': avg_csr_change
        }
    
    if not rows:
        return {}
    
    def apply_heat(key, higher_better):
        values = [row.get(key, 0) for row in rows.values()]
        for row in rows.values():
            row[f'{key}_heat'] = get_heatmap_class(row.get(key), values, higher_better)
    
    apply_heat('win_pct', True)
    apply_heat('kda', True)
    apply_heat('accuracy', True)
    
    for player, row in rows.items():
        row.update({
            'games': format_int(row.get('games', 0)),
            'win_pct': format_float(row.get('win_pct', 0), 1),
            'kda': format_float(row.get('kda', 0), 2),
            'accuracy': format_float(row.get('accuracy', 0), 1),
            'avg_csr_change': format_signed(row.get('avg_csr_change', 0), 0)
        })
    
    return rows


def build_30day_comparison(df: pd.DataFrame, player: str) -> dict:
    if df.empty or 'date' not in df.columns or not player:
        return {}
    player_df = df[df['player_gamertag'] == player] if 'player_gamertag' in df.columns else pd.DataFrame()
    if player_df.empty:
        return {}
    player_df['date'] = pd.to_datetime(player_df['date'], errors='coerce', utc=True)
    player_df = player_df.dropna(subset=['date'])
    if player_df.empty:
        return {}
    now = player_df['date'].max()
    last_start = now - pd.Timedelta(days=30)
    prev_start = now - pd.Timedelta(days=60)
    
    last_df = player_df[player_df['date'] >= last_start]
    prev_df = player_df[(player_df['date'] < last_start) & (player_df['date'] >= prev_start)]
    
    last_stats = summarize_stats(last_df)
    prev_stats = summarize_stats(prev_df)
    
    win_diff = last_stats.get('win_rate', 0) - prev_stats.get('win_rate', 0)
    kda_diff = last_stats.get('kda', 0) - prev_stats.get('kda', 0)
    
    if win_diff > 1 or kda_diff > 0.1:
        trend = 'up'
    elif win_diff < -1 or kda_diff < -0.1:
        trend = 'down'
    else:
        trend = 'stable'
    
    return {
        'trend': trend,
        'win_pct_diff': format_signed(win_diff, 1),
        'kda_diff': format_signed(kda_diff, 2),
        'win_pct_class': 'heat-good' if win_diff > 0 else 'heat-poor' if win_diff < 0 else '',
        'kda_class': 'heat-good' if kda_diff > 0 else 'heat-poor' if kda_diff < 0 else ''
    }


def build_weapon_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    if ranked_df.empty:
        return []
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        fired = numeric_series(player_df, 'shots_fired').sum()
        hit = numeric_series(player_df, 'shots_hit').sum()
        accuracy = hit / fired * 100 if fired > 0 else 0
        
        kills = numeric_series(player_df, 'kills').sum()
        headshots = numeric_series(player_df, 'headshot_kills').sum()
        hs_pct = headshots / kills * 100 if kills > 0 else 0
        
        rows.append({
            'player': player,
            'shots_fired': format_int(fired),
            'shots_hit': format_int(hit),
            'accuracy': format_float(accuracy, 1),
            'headshots': format_int(headshots),
            'hs_pct': format_float(hs_pct, 1),
            'melee': format_int(numeric_series(player_df, 'melee_kills').sum()),
            'grenades': format_int(numeric_series(player_df, 'grenade_kills').sum()),
            'power': format_int(numeric_series(player_df, 'power_weapon_kills').sum()),
            'snipe_medals': format_int(safe_col_sum(player_df, 'medal_Snipe')),
            'no_scope_medals': format_int(safe_col_sum(player_df, 'medal_No_Scope'))
        })
    
    add_heatmap_classes(rows, {
        'accuracy': True,
        'headshots': True,
        'hs_pct': True,
        'melee': True,
        'grenades': True,
        'power': True,
        'snipe_medals': True,
        'no_scope_medals': True
    })
    
    return rows


def build_weapon_accuracy_trend(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    trend_df = apply_trend_range(normalize_trend_df(df), '30')
    if trend_df.empty:
        return {}
    
    working = trend_df.copy()
    fired = pd.to_numeric(working.get('shots_fired', 0), errors='coerce').fillna(0)
    hit = pd.to_numeric(working.get('shots_hit', 0), errors='coerce').fillna(0)
    working['accuracy_pct'] = 0.0
    nonzero = fired > 0
    working.loc[nonzero, 'accuracy_pct'] = hit[nonzero] / fired[nonzero] * 100
    if not nonzero.any() and 'accuracy' in working.columns:
        acc = pd.to_numeric(working['accuracy'], errors='coerce').fillna(0)
        if acc.max() <= 1:
            acc = acc * 100
        working['accuracy_pct'] = acc
    
    return build_trend_data(working, 'accuracy_pct', 'accuracy')

def add_trend_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by the trends charts."""
    if df.empty:
        return df

    working = df.copy()

    if 'outcome' in working.columns:
        outcome_lower = working['outcome'].astype(str).str.lower()
        working['win_rate'] = (outcome_lower == 'win').astype(float) * 100

    obj_score = extract_objective_score(working)
    if not obj_score.empty:
        working['obj_score'] = obj_score

    if 'dmg/min' in working.columns:
        working['dmg_min'] = pd.to_numeric(working['dmg/min'], errors='coerce').fillna(0)
    elif 'damage_dealt' in working.columns and 'duration' in working.columns:
        damage_dealt = pd.to_numeric(working['damage_dealt'], errors='coerce').fillna(0)
        duration = pd.to_numeric(working['duration'], errors='coerce').fillna(0)
        duration_min = duration / 60.0
        working['dmg_min'] = 0.0
        nonzero = duration_min > 0
        working.loc[nonzero, 'dmg_min'] = damage_dealt[nonzero] / duration_min[nonzero]

    if 'dmg_difference' in working.columns:
        working['dmg_diff'] = pd.to_numeric(
            working['dmg_difference'], errors='coerce'
        ).fillna(0)
    elif 'damage_dealt' in working.columns and 'damage_taken' in working.columns:
        damage_dealt = pd.to_numeric(working['damage_dealt'], errors='coerce').fillna(0)
        damage_taken = pd.to_numeric(working['damage_taken'], errors='coerce').fillna(0)
        working['dmg_diff'] = damage_dealt - damage_taken

    if 'max_killing_spree' in working.columns:
        working['max_spree'] = pd.to_numeric(
            working['max_killing_spree'], errors='coerce'
        ).fillna(0)

    if 'duration' in working.columns:
        working['duration_min'] = pd.to_numeric(
            working['duration'], errors='coerce'
        ).fillna(0) / 60.0

    if 'kills' in working.columns:
        working['kills_pg'] = pd.to_numeric(working['kills'], errors='coerce').fillna(0)
    if 'deaths' in working.columns:
        working['deaths_pg'] = pd.to_numeric(working['deaths'], errors='coerce').fillna(0)

    return working


def build_leaderboard(df: pd.DataFrame, category: str, limit: int = 10) -> list:
    """Build leaderboard for a specific category."""
    if df.empty:
        return []
    
    rows = []
    
    if category == 'csr':
        csr_overview = build_csr_overview(df)
        for row in csr_overview:
            csr_val = to_number(row.get('current_csr'))
            if csr_val:
                rows.append({
                    'rank': 0,
                    'player': row['player'],
                    'value': row['current_csr'],
                    'csr': row['current_csr'],
                    'context': ''
                })
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)
    
    elif category == 'kda':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum()
            deaths = pd.to_numeric(player_df.get('deaths', 0), errors='coerce').fillna(0).sum()
            assists = pd.to_numeric(player_df.get('assists', 0), errors='coerce').fillna(0).sum()
            games = len(player_df)
            
            kda = safe_kda(kills / games if games else 0, assists / games if games else 0, deaths / games if games else 0)
            
            rows.append({
                'rank': 0,
                'player': player,
                'value': format_float(kda, 2),
                'kda': format_float(kda, 2),
                'context': f'{games} games'
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)
    
    elif category == 'win_rate':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            games = len(player_df)
            outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
            wins = (outcomes == 'win').sum() if not outcomes.empty else 0
            
            if games:
                rows.append({
                    'rank': 0,
                    'player': player,
                    'value': format_float(wins / games * 100, 1),
                    'win_rate': format_float(wins / games * 100, 1),
                    'context': f'{wins}-{games - wins}'
                })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)

    elif category == 'accuracy':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            games = len(player_df)
            if games == 0:
                continue
            fired = pd.to_numeric(player_df.get('shots_fired', 0), errors='coerce').fillna(0).sum()
            hit = pd.to_numeric(player_df.get('shots_hit', 0), errors='coerce').fillna(0).sum()
            if fired > 0:
                accuracy = hit / fired * 100
            else:
                acc = pd.to_numeric(player_df.get('accuracy', 0), errors='coerce').fillna(0)
                accuracy = acc.mean()
                if accuracy <= 1:
                    accuracy *= 100
            rows.append({
                'rank': 0,
                'player': player,
                'value': format_float(accuracy, 1),
                'accuracy': format_float(accuracy, 1),
                'context': f'{games} games'
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)

    elif category == 'streak':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            if player_df.empty:
                continue
            win_streak, _ = compute_streaks(player_df)
            rows.append({
                'rank': 0,
                'player': player,
                'value': win_streak,
                'streak': format_int(win_streak),
                'context': ''
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)

    elif category == 'kills':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            if player_df.empty:
                continue
            kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum()
            rows.append({
                'rank': 0,
                'player': player,
                'value': format_int(kills),
                'kills': format_int(kills),
                'context': ''
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)

    elif category == 'games':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            games = len(player_df)
            rows.append({
                'rank': 0,
                'player': player,
                'value': format_int(games),
                'games': format_int(games),
                'context': ''
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)

    elif category == 'csr_gained':
        for player in unique_sorted(df['player_gamertag']):
            player_df = df[df['player_gamertag'] == player]
            if player_df.empty or 'date' not in player_df.columns:
                continue
            player_df = player_df.copy()
            player_df['date'] = pd.to_datetime(player_df['date'], errors='coerce', utc=True)
            player_df = player_df.dropna(subset=['date']).sort_values('date', ascending=True)
            if player_df.empty:
                continue
            csr_vals = extract_csr_values(player_df)
            if csr_vals.empty or len(csr_vals) < 2:
                continue
            gain = float(csr_vals.iloc[-1] - csr_vals.iloc[0])
            if gain <= 0:
                continue
            rows.append({
                'rank': 0,
                'player': player,
                'value': gain,
                'csr_gained': format_int(gain),
                'context': ''
            })
        
        rows.sort(key=lambda x: to_number(x['value']) or 0, reverse=True)
    
    # Add rank numbers
    for idx, row in enumerate(rows[:limit], 1):
        row['rank'] = idx
    
    return rows[:limit]


# Initialize at module level
ENGINE = get_engine()
ensure_indexes(ENGINE)
cache = DataCache(ENGINE)
count_cache = DbCountCache(ENGINE)
INSIGHTS_CACHE = {
    'last_ts': 0.0,
    'payload': None
}
PLAYER_HOVER_CACHE = {
    'last_ts': 0.0,
    'payload': {}
}


def build_summary_table(df: pd.DataFrame) -> list:
    """Build summary table comparing 30-day stats to lifetime stats."""
    if df.empty or 'date' not in df.columns:
        return []

    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []

    summary_rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue

        # Lifetime stats
        lifetime_games = len(player_df)
        lifetime_outcomes = player_df['outcome'].astype(str).str.lower() if 'outcome' in player_df.columns else pd.Series()
        lifetime_wins = (lifetime_outcomes == 'win').sum() if not lifetime_outcomes.empty else 0
        lifetime_win_pct = lifetime_wins / lifetime_games * 100 if lifetime_games > 0 else 0
        lifetime_stats = calculate_player_stats(player_df, lifetime_games)
        lifetime_stats['win_pct'] = lifetime_win_pct

        # 30-day stats
        max_date = player_df['date'].max()
        cutoff_date = max_date - pd.Timedelta(days=30)
        recent_df = player_df[player_df['date'] >= cutoff_date]
        recent_games = len(recent_df)
        recent_outcomes = recent_df['outcome'].astype(str).str.lower() if 'outcome' in recent_df.columns else pd.Series()
        recent_wins = (recent_outcomes == 'win').sum() if not recent_outcomes.empty else 0
        recent_win_pct = recent_wins / recent_games * 100 if recent_games > 0 else 0
        recent_stats = calculate_player_stats(recent_df, recent_games)
        recent_stats['win_pct'] = recent_win_pct

        stats_to_compare = [
            {'key': 'kda', 'name': 'KDA', 'format': '{:.2f}'},
            {'key': 'win_pct', 'name': 'Win %', 'format': '{:.1f}%'},
            {'key': 'accuracy', 'name': 'Accuracy', 'format': '{:.1f}%'},
            {'key': 'dmg_per_min', 'name': 'DMG/min', 'format': '{:.0f}'},
        ]

        for stat in stats_to_compare:
            recent_val = recent_stats.get(stat['key'], 0)
            lifetime_val = lifetime_stats.get(stat['key'], 0)
            trend = recent_val - lifetime_val
            trend_class = 'heat-excellent' if trend > 0 else 'heat-poor' if trend < 0 else ''
            
            summary_rows.append({
                'player': player,
                'stat': stat['name'],
                'recent': stat['format'].format(recent_val),
                'lifetime': stat['format'].format(lifetime_val),
                'trend': f'{"+" if trend > 0 else ""}{stat["format"].format(trend)}',
                'trend_class': trend_class,
            })

    summary_rows.sort(key=lambda x: to_number(x['trend']), reverse=True)
    return summary_rows


def build_match_details(df: pd.DataFrame, match_id: str) -> dict:
    """Build detailed view for a single match."""
    if df.empty or 'match_id' not in df.columns:
        return {}

    match_df = df[df['match_id'] == match_id]
    if match_df.empty:
        return {}

    # Get the first row for general match info
    match_info = match_df.iloc[0]

    # Scoreboard
    scoreboard = []
    for _, row in match_df.iterrows():
        scoreboard.append(format_player_stats_row(row['player_gamertag'], 1, 1 if row['outcome'] == 'win' else 0, row))

    # Medals
    medal_cols = [col for col in match_df.columns if str(col).startswith('medal_')]
    medals = []
    for col in medal_cols:
        total = match_df[col].sum()
        if total > 0:
            medals.append({'name': col.replace('medal_', '').replace('_', ' ').title(), 'count': total})
    
    return {
        'match_id': match_id,
        'map': match_info.get('map'),
        'game_type': match_info.get('game_type'),
        'playlist': match_info.get('playlist'),
        'date': format_date(match_info.get('date')),
        'duration': match_info.get('duration'),
        'scoreboard': scoreboard,
        'medals': medals,
    }


def build_player_analysis(df: pd.DataFrame) -> list:
    """Build player analysis table combining trends and outliers."""
    if df.empty or 'date' not in df.columns:
        return []

    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    ranked_df['date'] = pd.to_datetime(ranked_df['date'], utc=True, errors='coerce')
    ranked_df = ranked_df.dropna(subset=['date'])
    
    if ranked_df.empty:
        return []

    analysis_rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue

        # Lifetime stats
        lifetime_games = len(player_df)
        lifetime_stats = calculate_player_stats(player_df, lifetime_games)

        # 30-day stats
        max_date = player_df['date'].max()
        cutoff_date = max_date - pd.Timedelta(days=30)
        recent_df = player_df[player_df['date'] >= cutoff_date]
        recent_games = len(recent_df)
        recent_stats = calculate_player_stats(recent_df, recent_games)

        # Outlier spotlights
        outlier_highlights = build_outlier_spotlight(df)
        player_outliers = next((r['highlights'] for r in outlier_highlights if r['player'] == player), [])

        analysis_rows.append({
            'player': player,
            'kda_recent': '{:.2f}'.format(recent_stats.get('kda', 0)),
            'kda_lifetime': '{:.2f}'.format(lifetime_stats.get('kda', 0)),
            'kda_trend': recent_stats.get('kda', 0) - lifetime_stats.get('kda', 0),
            'win_pct_recent': '{:.1f}%'.format(recent_stats.get('win_pct', 0)),
            'win_pct_lifetime': '{:.1f}%'.format(lifetime_stats.get('win_pct', 0)),
            'win_pct_trend': recent_stats.get('win_pct', 0) - lifetime_stats.get('win_pct', 0),
            'outliers': player_outliers
        })

    analysis_rows.sort(key=lambda x: x['kda_trend'], reverse=True)
    return analysis_rows


@app.route('/')
def index():


    df = cache.get()


    player = request.args.get('player', 'all')


    playlist = request.args.get('playlist', 'all')


    mode = request.args.get('mode', 'all')


    filtered = apply_filters(df, player, playlist, mode)


    


    csr_overview_rows = build_csr_overview(df)


    csr_overview_trends = build_csr_trends(apply_trend_range(normalize_trend_df(df), '90'))


    ranked_arena_rows = build_ranked_arena_summary(df)


    ranked_arena_30day_rows = build_ranked_arena_30day(df)


    ranked_arena_90day_rows = build_ranked_arena_90day(df)


    ranked_arena_180day_rows = build_ranked_arena_180day(df)


    ranked_arena_1y_rows = build_ranked_arena_1y(df)


    ranked_arena_2y_rows = build_ranked_arena_2y(df)


    ranked_arena_lifetime_rows = build_ranked_arena_lifetime(df)


    player_analysis_rows = build_player_analysis(df)


    summary_rows = build_summary_table(df)


    players_list = unique_sorted(df['player_gamertag']) if not df.empty and 'player_gamertag' in df.columns else []


    map_rows = build_breakdown(filtered, 'map')


    playlist_rows = build_breakdown(filtered, 'playlist')


    cards = build_cards(filtered)


    # Outlier spotlight
    outlier_range = request.args.get('outliers', 'all')
    outlier_rows = build_outlier_spotlight(df, outlier_range)
    outlier_ranges = [
        {'key': '30', 'label': '30D', 'active': outlier_range == '30'},
        {'key': '90', 'label': '90D', 'active': outlier_range == '90'},
        {'key': '365', 'label': '1Y', 'active': outlier_range == '365'},
        {'key': 'all', 'label': 'Lifetime', 'active': outlier_range == 'all'}
    ]


    status = load_status()


    last_update = status.get('last_update')


    


    return render_template('index.html',


                          app_title=APP_TITLE,


                          csr_overview_rows=csr_overview_rows,


                          csr_overview_trends=csr_overview_trends,


                          player_analysis_rows=player_analysis_rows,


                          summary_rows=summary_rows,


                          ranked_arena_rows=ranked_arena_rows,


                          ranked_arena_30day_rows=ranked_arena_30day_rows,


                          ranked_arena_90day_rows=ranked_arena_90day_rows,


                          ranked_arena_180day_rows=ranked_arena_180day_rows,


                          ranked_arena_1y_rows=ranked_arena_1y_rows,


                          ranked_arena_2y_rows=ranked_arena_2y_rows,


                          ranked_arena_lifetime_rows=ranked_arena_lifetime_rows,


                          players=players_list,


                          map_rows=map_rows,


                          playlist_rows=playlist_rows,


                          cards=cards,


                          outlier_rows=outlier_rows,
                          outlier_ranges=outlier_ranges,
                          outlier_range=outlier_range,


                          last_update=last_update,


                          playlists=unique_sorted(df['playlist']) if not df.empty else [],


                          modes=unique_sorted(df['game_type']) if not df.empty else [],


                          selected_player=player,


                          selected_playlist=playlist,


                          selected_mode=mode,


                          db_row_count=count_cache.get())


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        new_settings = {
            'match_limit': int(request.form.get('match_limit', LIFETIME_LIMIT_DEFAULT)),
            'update_interval': int(request.form.get('update_interval', 60)),
            'force_refresh': request.form.get('force_refresh') == 'on'
        }
        save_settings(new_settings)
        cache.force_reload()
        return redirect(url_for('settings'))
    
    current_settings = load_settings()
    return render_template('settings.html',
                          app_title=APP_TITLE,
                          settings=current_settings,
                          db_row_count=count_cache.get())


@app.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    df = cache.get()
    status = load_status()
    message = None
    error = None
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        gamertag = request.form.get('gamertag', '').strip()
        contact = request.form.get('contact', '').strip()
        summary = request.form.get('summary', '').strip()
        details = request.form.get('details', '').strip()
        follow_up = request.form.get('follow_up', '').strip()
        
        if not summary:
            error = 'Please add a short summary.'
        else:
            payload = {
                'name': name or None,
                'gamertag': gamertag or None,
                'contact': contact or None,
                'summary': summary,
                'details': details or None,
                'follow_up': follow_up or None
            }
            save_suggestion(ENGINE, payload)
            message = 'Thanks! Your suggestion has been saved.'
    
    suggestions_rows = fetch_suggestions(ENGINE, limit=50)
    
    return render_template('suggestions.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          suggestions=suggestions_rows,
                          message=message,
                          error=error,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/lifetime')
def lifetime():
    """Lifetime stats page."""
    df = cache.get()
    player = request.args.get('player', 'all')
    playlist = request.args.get('playlist', 'all')
    mode = request.args.get('mode', 'all')
    limit_key = request.args.get('limit', str(LIFETIME_LIMIT_DEFAULT))
    
    filtered = apply_filters(df, player, playlist, mode)
    
    lifetime_rows = build_lifetime_stats(filtered)
    limit = None
    limit_note = None
    if limit_key and limit_key != 'all':
        try:
            limit = int(limit_key)
        except ValueError:
            limit = None
        if limit is not None and limit <= 0:
            limit = None
        if limit is not None and limit > LIFETIME_LIMIT_MAX:
            limit = LIFETIME_LIMIT_MAX
            limit_note = f'Showing latest {LIFETIME_LIMIT_MAX:,} matches for performance.'
    else:
        limit = LIFETIME_LIMIT_MAX
        limit_note = f'Showing latest {LIFETIME_LIMIT_MAX:,} matches for performance.'
    session_rows = build_session_history(filtered, limit=limit)
    
    status = load_status()
    
    return render_template('lifetime.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          lifetime_rows=lifetime_rows,
                          session_rows=session_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get(),
                          playlists=unique_sorted(df['playlist']) if not df.empty else [],
                          modes=unique_sorted(df['game_type']) if not df.empty else [],
                          selected_player=player,
                          selected_playlist=playlist,
                          selected_mode=mode,
                          selected_limit=limit_key,
                          limit_note=limit_note)





@app.route('/advanced')
def advanced():
    """Advanced/objective stats page."""
    df = cache.get()
    
    objective_session_rows = build_objective_stats(df, 'session')
    objective_30day_rows = build_objective_stats(df, '30day')
    objective_lifetime_rows = build_objective_stats(df, 'all')
    
    status = load_status()
    
    return render_template('advanced.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          objective_session_rows=objective_session_rows,
                          objective_30day_rows=objective_30day_rows,
                          objective_lifetime_rows=objective_lifetime_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/medals')
def medals():
    """Medal statistics page."""
    df = cache.get()
    
    medal_players, ranked_medal_rows, total_medal_rows = build_medal_stats(df)
    
    status = load_status()
    
    return render_template('medals.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          ranked_arena_medal_players=medal_players,
                          ranked_arena_medal_rows=ranked_medal_rows,
                          medal_players=medal_players,
                          medal_rows=total_medal_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/debug/medals')
def debug_medals():
    """Debug endpoint to see raw medal data."""
    from flask import jsonify
    df = cache.get()
    
    # Get medal columns
    medal_cols = [col for col in df.columns if col.startswith('medal_') and col != 'medal_count']
    
    # Get data for each player
    result = {}
    for player in unique_sorted(df['player_gamertag']):
        player_df = df[df['player_gamertag'] == player]
        games = len(player_df)
        
        player_medals = {}
        for col in medal_cols[:20]:  # First 20 medal types
            total = safe_col_sum(player_df, col)
            per_game = total / games if games > 0 else 0
            if total > 0:  # Only show medals they actually have
                player_medals[col] = {
                    'total': float(total),
                    'per_game': round(per_game, 3),
                    'games': games
                }
        
        result[player] = player_medals
    
    return jsonify(result)


@app.route('/highlights')
def highlights():
    """Highlight games page."""
    df = cache.get()
    
    highlight_rows = build_highlight_games(df, limit=50)
    
    status = load_status()
    
    return render_template('highlights.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          highlight_rows=highlight_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/columns')
def columns():
    """Show available columns/keys."""
    df = cache.get()
    status = load_status()
    
    wide_columns = []
    kv_keys = []
    
    try:
        if inspect(ENGINE).has_table('halo_match_stats'):
            wide_columns = [c.get('name') for c in inspect(ENGINE).get_columns('halo_match_stats') if c.get('name')]
    except:
        pass
    
    return render_template('columns.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          last_update=status.get('last_update'),
                          wide_columns=wide_columns,
                          kv_keys=kv_keys,
                          wide_count=len(wide_columns),
                          kv_count=len(kv_keys),
                          db_row_count=count_cache.get())


@app.route('/player/<player_name>')
@app.route('/player/<player_name>')
def player_profile(player_name: str):
    """Individual player profile page."""
    df = cache.get()
    
    if df.empty:
        return render_template('player.html',
                              app_title=APP_TITLE,
                              player_name=player_name,
                              players=[],
                              error='No data available',
                              last_update=load_status().get('last_update'),
                              db_row_count=count_cache.get())
    
    all_players = unique_sorted(df['player_gamertag']) if not df.empty else []
    
    if player_name not in all_players:
        return render_template('player.html',
                              app_title=APP_TITLE,
                              player_name=player_name,
                              players=all_players,
                              error='Player not found',
                              last_update=load_status().get('last_update'),
                              db_row_count=count_cache.get())
    
    presence = load_presence()
    status = load_status()
    
    # Get player-specific CSR info
    csr_overview = build_csr_overview(df)
    player_csr = next((r for r in csr_overview if r['player'] == player_name), {})
    
    # Last session
    ranked_sessions = build_ranked_arena_summary(df)
    last_session = next((r for r in ranked_sessions if r['player'] == player_name), {})
    
    player_df = df[df['player_gamertag'] == player_name] if not df.empty and 'player_gamertag' in df.columns else pd.DataFrame()
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    avg_30day_rows = build_30day_overview(ranked_df)
    avg_30day = avg_30day_rows.get(player_name, {})
    if not avg_30day:
        avg_30day = {
            'games': '0',
            'win_pct': '0',
            'kda': '0',
            'accuracy': '0',
            'avg_csr_change': '0',
            'win_pct_heat': '',
            'kda_heat': '',
            'accuracy_heat': ''
        }
    
    comparison = build_30day_comparison(ranked_df, player_name)
    if not comparison:
        comparison = {
            'trend': 'stable',
            'win_pct_diff': '-',
            'kda_diff': '-',
            'win_pct_class': '',
            'kda_class': ''
        }
    
    match_history = build_player_match_history(player_df, limit=20)
    map_stats = build_player_map_summary(player_df)
    teammate_stats = build_teammate_stats(ranked_df, player_name)
    csr_history = build_player_csr_history(df, player_name)
    current_streak = compute_current_streak(player_df)
    
    return render_template('player.html',
                          app_title=APP_TITLE,
                          player_name=player_name,
                          players=all_players,
                          is_online=is_player_online(presence, player_name),
                          current_csr=player_csr.get('current_csr', '-'),
                          max_csr=player_csr.get('max_csr', '-'),
                          current_streak=current_streak,
                          last_session=last_session,
                          avg_30day=avg_30day,
                          comparison=comparison,
                          match_history=match_history,
                          map_stats=map_stats,
                          teammate_stats=teammate_stats,
                          player_win_corr=build_win_corr(player_df),
                          csr_history=csr_history,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/weapons')
def weapons():
    """Weapon statistics page."""
    df = cache.get()
    
    weapon_rows = build_weapon_rows(df)
    accuracy_trend = build_weapon_accuracy_trend(df)
    
    status = load_status()
    
    return render_template('weapons.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          weapon_rows=weapon_rows,
                          accuracy_trend=accuracy_trend,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/hall')
def hall():
    """Hall of fame/shame page."""
    df = cache.get()
    
    # Hall of fame/shame (stub for now)
    hall_fame_rows = []
    hall_shame_rows = []
    hall_fame_rows, hall_shame_rows = build_hall_fame_shame(df)
    status = load_status()
    return render_template('hall.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          hall_fame_rows=hall_fame_rows,
                          hall_shame_rows=hall_shame_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/maps')
def maps():
    """Map statistics page."""
    df = cache.get()
    
    map_rows = build_map_stats(df)
    mode_rows = build_mode_stats(df)
    player_map_rows = build_player_map_stats(df)
    
    status = load_status()
    
    return render_template('maps.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          map_rows=map_rows,
                          mode_rows=mode_rows,
                          player_map_rows=player_map_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/trends')
def trends():
    """Trend analysis page."""
    df = cache.get()

    range_key = request.args.get('range', '90')
    trend_df = add_trend_metrics(apply_trend_range(normalize_trend_df(df), range_key))
    
    status = load_status()
    
    trend_ranges = [
        {'key': '7', 'label': '7 days', 'active': range_key == '7'},
        {'key': '30', 'label': '30 days', 'active': range_key == '30'},
        {'key': '90', 'label': '90 days', 'active': range_key == '90'},
        {'key': '180', 'label': '180 days', 'active': range_key == '180'},
        {'key': '365', 'label': '1 year', 'active': range_key == '365'},
        {'key': 'all', 'label': 'All time', 'active': range_key == 'all'}
    ]
    
    return render_template('trends.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          csr_trends=build_csr_trends(trend_df),
                          win_rate_trends=build_win_rate_trends(trend_df),
                          kda_trends=build_trend_data(trend_df, 'kda', 'kda') if 'kda' in trend_df.columns else {},
                          obj_score_trends=build_trend_data(trend_df, 'obj_score', 'obj_score') if 'obj_score' in trend_df.columns else {},
                          damage_min_trends=build_trend_data(trend_df, 'dmg_min', 'dmg_min') if 'dmg_min' in trend_df.columns else {},
                          damage_diff_trends=build_trend_data(trend_df, 'dmg_diff', 'dmg_diff') if 'dmg_diff' in trend_df.columns else {},
                          accuracy_trends=build_trend_data(trend_df, 'accuracy', 'accuracy') if 'accuracy' in trend_df.columns else {},
                          kills_pg_trends=build_trend_data(trend_df, 'kills_pg', 'kills_pg') if 'kills_pg' in trend_df.columns else {},
                          deaths_pg_trends=build_trend_data(trend_df, 'deaths_pg', 'deaths_pg') if 'deaths_pg' in trend_df.columns else {},
                          max_spree_trends=build_trend_data(trend_df, 'max_spree', 'max_spree') if 'max_spree' in trend_df.columns else {},
                          duration_trends=build_trend_data(trend_df, 'duration_min', 'duration_min') if 'duration_min' in trend_df.columns else {},
                          activity_heatmap=build_activity_heatmap(trend_df),
                          win_corr_overall=build_win_corr(trend_df),
                          player_moments=build_player_moments(trend_df),
                          trend_ranges=trend_ranges,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/insights')
def insights():
    """Insights and comparisons page."""
    df = cache.get()
    
    players_list = unique_sorted(df['player_gamertag']) if not df.empty else []
    player = request.args.get('player', 'all')
    start_a = request.args.get('start_a', '')
    end_a = request.args.get('end_a', '')
    start_b = request.args.get('start_b', '')
    end_b = request.args.get('end_b', '')
    
    status = load_status()
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    session_compare_rows = build_session_compare(ranked_df, player, start_a, end_a, start_b, end_b)
    insights_payload = get_insights_payload(ranked_df)
    clutch_rows = insights_payload['clutch_rows']
    role_rows = insights_payload['role_rows']
    momentum_rows = insights_payload['momentum_rows']
    veto_rows = insights_payload['veto_rows']
    consistency_rows = insights_payload['consistency_rows']
    notable_rows = insights_payload['notable_rows']
    change_rows = insights_payload['change_rows']
    lineup2_rows = insights_payload['lineup2_rows']
    lineup3_rows = insights_payload['lineup3_rows']
    lineup4_rows = insights_payload['lineup4_rows']
    
    range_a_label = 'Range A'
    range_b_label = 'Range B'
    if start_a or end_a:
        range_a_label = f"{start_a or '...'} to {end_a or '...'}"
    if start_b or end_b:
        range_b_label = f"{start_b or '...'} to {end_b or '...'}"
    
    return render_template('insights.html',
                          app_title=APP_TITLE,
                          players=players_list,
                          selected_player=player,
                          start_a=start_a,
                          end_a=end_a,
                          start_b=start_b,
                          end_b=end_b,
                          range_a_label=range_a_label,
                          range_b_label=range_b_label,
                          session_compare_rows=session_compare_rows,
                          clutch_rows=clutch_rows,
                          role_rows=role_rows,
                          momentum_rows=momentum_rows,
                          veto_rows=veto_rows,
                          consistency_rows=consistency_rows,
                          notable_rows=notable_rows,
                          change_rows=change_rows,
                          lineup2_rows=lineup2_rows,
                          lineup3_rows=lineup3_rows,
                          lineup4_rows=lineup4_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/leaderboard')
def leaderboard():
    """Leaderboards page."""
    df = cache.get()
    
    period = request.args.get('period', 'all')
    leaderboard_df = apply_leaderboard_period(df, period)
    
    leaders = {
        'csr_leaders': build_leaderboard(leaderboard_df, 'csr'),
        'csr_gained_leaders': build_leaderboard(leaderboard_df, 'csr_gained'),
        'win_rate_leaders': build_leaderboard(leaderboard_df, 'win_rate'),
        'kda_leaders': build_leaderboard(leaderboard_df, 'kda'),
        'accuracy_leaders': build_leaderboard(leaderboard_df, 'accuracy'),
        'streak_leaders': build_leaderboard(leaderboard_df, 'streak'),
        'kills_leaders': build_leaderboard(leaderboard_df, 'kills'),
        'games_leaders': build_leaderboard(leaderboard_df, 'games')
    }
    
    status = load_status()
    
    return render_template('leaderboard.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          period=period,
                          **leaders,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/api/debug')
def debug_data():
    """Debug endpoint - shows columns and data from both DB and cache."""
    engine = get_engine()
    try:
        # Get 5 rows from DB directly
        query = "SELECT * FROM halo_match_stats LIMIT 5"
        db_df = pd.read_sql_query(query, engine)
        
        # Get data from cache (what webapp uses)
        cache_df = cache.get()
        
        db_columns = list(db_df.columns) if not db_df.empty else []
        cache_columns = list(cache_df.columns) if not cache_df.empty else []
        
        # Find columns missing in cache
        missing_in_cache = [c for c in db_columns if c not in cache_columns]
        extra_in_cache = [c for c in cache_columns if c not in db_columns]
        
        # Sample data from cache
        sample_rows = []
        if not cache_df.empty:
            for idx, row in cache_df.head(3).iterrows():
                row_dict = {}
                for key in ['player_gamertag', 'date', 'kills', 'deaths', 'damage_dealt', 
                           'damage_taken', 'accuracy', 'post_match_csr', 'kda']:
                    if key in row:
                        val = row[key]
                        row_dict[key] = None if pd.isna(val) else val
                sample_rows.append(row_dict)
        
        return {
            "db_columns_count": len(db_columns),
            "cache_columns_count": len(cache_columns),
            "db_rows": len(db_df),
            "cache_rows": len(cache_df),
            "missing_in_cache": missing_in_cache,
            "extra_in_cache": extra_in_cache,
            "cache_columns": cache_columns,
            "sample_cache_data": sample_rows
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}



@app.route('/api/export')
def export_data():
    """Export filtered data as CSV or JSON."""
    df = cache.get()
    format_type = request.args.get('format', 'json')
    player = request.args.get('player', 'all')
    playlist = request.args.get('playlist', 'all')
    mode = request.args.get('mode', 'all')
    
    filtered = apply_filters(df, player, playlist, mode)
    
    if format_type == 'csv':
        csv_data = filtered.to_csv(index=False)
        return Response(csv_data,
                       mimetype='text/csv',
                       headers={'Content-Disposition': 'attachment;filename=halo_stats.csv'})
    else:
        return Response(filtered.to_json(orient='records'),
                       mimetype='application/json')

@app.route('/compare')
def compare():
    df = cache.get()
    player_summaries = build_player_summary(df)
    return render_template('compare.html',
                          app_title=APP_TITLE,
                          player_summaries=player_summaries,
                          db_row_count=count_cache.get())


def build_player_summary(df: pd.DataFrame) -> dict:
    """Build comprehensive player analysis comparing all players."""
    if df.empty or 'player_gamertag' not in df.columns:
        return {}
    
    all_players = unique_sorted(df['player_gamertag'])
    if not all_players:
        return {}

    medal_cols = [
        col for col in df.columns
        if col.startswith('medal_') and col != 'medal_count'
    ]
    medal_totals = []
    for col in medal_cols:
        total = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
        if total > 0:
            medal_totals.append((col, total))
    medal_totals.sort(key=lambda item: item[1], reverse=True)
    compare_medal_cols = [col for col, _ in medal_totals[:50]]

    player_metrics = {}
    for player in all_players:
        player_df = df[df['player_gamertag'] == player]
        if player_df.empty:
            continue

        games = len(player_df)
        wins = (player_df['outcome'].astype(str).str.lower() == 'win').sum()

        kills_pg = safe_col_sum(player_df, 'kills') / games
        deaths_pg = safe_col_sum(player_df, 'deaths') / games
        assists_pg = safe_col_sum(player_df, 'assists') / games

        total_shots_hit = safe_col_sum(player_df, 'shots_hit')
        total_shots_fired = safe_col_sum(player_df, 'shots_fired')
        if total_shots_fired > 0:
            accuracy = total_shots_hit / total_shots_fired * 100
        else:
            accuracy = numeric_series(player_df, 'accuracy').mean()
            if accuracy <= 1:
                accuracy *= 100

        total_dmg_dealt = safe_col_sum(player_df, 'damage_dealt')
        total_dmg_taken = safe_col_sum(player_df, 'damage_taken')
        dmg_pg = total_dmg_dealt / games
        dmg_diff_pg = (total_dmg_dealt - total_dmg_taken) / games

        total_duration = safe_col_sum(player_df, 'duration')
        dmg_per_min = total_dmg_dealt / (total_duration / 60) if total_duration > 0 else 0

        medals_pg = safe_col_sum(player_df, 'medal_count') / games

        win_pct = wins / games * 100
        kda = safe_kda(kills_pg, assists_pg, deaths_pg)

        total_kills = safe_col_sum(player_df, 'kills')
        headshot_pct = (safe_col_sum(player_df, 'headshot_kills') / total_kills * 100) if total_kills > 0 else 0
        melee_pct = (safe_col_sum(player_df, 'melee_kills') / total_kills * 100) if total_kills > 0 else 0
        grenade_pct = (safe_col_sum(player_df, 'grenade_kills') / total_kills * 100) if total_kills > 0 else 0
        power_pct = (safe_col_sum(player_df, 'power_weapon_kills') / total_kills * 100) if total_kills > 0 else 0

        avg_life = safe_col_sum(player_df, 'average_life_duration') / games if 'average_life_duration' in player_df.columns else 0
        obj_score_pg = objective_score_series(player_df).sum() / games if games else 0
        callouts_pg = safe_col_sum(player_df, 'callout_assists') / games
        score_pg = score_series(player_df).sum() / games if games else 0
        betrayals_pg = safe_col_sum(player_df, 'betrayals') / games
        suicides_pg = safe_col_sum(player_df, 'suicides') / games

        metrics = {
            'games': games,
            'win_pct': win_pct,
            'kda': kda,
            'kills_pg': kills_pg,
            'deaths_pg': deaths_pg,
            'assists_pg': assists_pg,
            'accuracy': accuracy,
            'dmg_pg': dmg_pg,
            'dmg_diff_pg': dmg_diff_pg,
            'dmg_per_min': dmg_per_min,
            'medals_pg': medals_pg,
            'headshot_pct': headshot_pct,
            'melee_pct': melee_pct,
            'grenade_pct': grenade_pct,
            'power_pct': power_pct,
            'avg_life': avg_life,
            'obj_score_pg': obj_score_pg,
            'callouts_pg': callouts_pg,
            'score_pg': score_pg,
            'betrayals_pg': betrayals_pg,
            'suicides_pg': suicides_pg
        }

        for col in compare_medal_cols:
            metrics[col] = safe_col_sum(player_df, col) / games if games else 0

        player_metrics[player] = metrics

    if not player_metrics:
        return {}

    def format_value(value: float, digits: int, suffix: str = '') -> str:
        return f"{format_float(value, digits)}{suffix}"

    stat_metric_defs = [
        {'key': 'kills_pg', 'label': 'Kills/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'deaths_pg', 'label': 'Deaths/Game', 'higher_is_better': False, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'assists_pg', 'label': 'Assists/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'kda', 'label': 'KDA', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 2)},
        {'key': 'accuracy', 'label': 'Accuracy', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'win_pct', 'label': 'Win %', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'dmg_pg', 'label': 'Damage/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 0)},
        {'key': 'dmg_diff_pg', 'label': 'Damage Diff/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_signed(v, 0)},
        {'key': 'dmg_per_min', 'label': 'Damage/Min', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 0)},
        {'key': 'score_pg', 'label': 'Score/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 0)},
        {'key': 'obj_score_pg', 'label': 'Objective Score/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'callouts_pg', 'label': 'Callouts/Game', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'headshot_pct', 'label': 'Headshot %', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'melee_pct', 'label': 'Melee Kill %', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'grenade_pct', 'label': 'Grenade Kill %', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'power_pct', 'label': 'Power Weapon %', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'avg_life', 'label': 'Avg Life', 'higher_is_better': True, 'is_medal': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'betrayals_pg', 'label': 'Betrayals/Game', 'higher_is_better': False, 'is_medal': False, 'format': lambda v: format_value(v, 2)},
        {'key': 'suicides_pg', 'label': 'Suicides/Game', 'higher_is_better': False, 'is_medal': False, 'format': lambda v: format_value(v, 2)},
    ]

    medal_metric_defs = [
        {'key': 'medals_pg', 'label': 'Medals/Game', 'higher_is_better': True, 'is_medal': True, 'format': lambda v: format_value(v, 2)}
    ]

    for col in compare_medal_cols:
        label = f"{col.replace('medal_', '').replace('_', ' ').title()} Medals/Game"
        medal_metric_defs.append({
            'key': col,
            'label': label,
            'higher_is_better': True,
            'is_medal': True,
            'format': lambda v: format_value(v, 2)
        })

    norm_cache = {}

    def get_norm(metric_key: str, higher_is_better: bool = True) -> dict:
        cache_key = (metric_key, higher_is_better)
        if cache_key in norm_cache:
            return norm_cache[cache_key]
        values = {player: safe_float(player_metrics[player].get(metric_key, 0)) for player in player_metrics}
        if not values:
            norm_cache[cache_key] = {player: 0.5 for player in player_metrics}
            return norm_cache[cache_key]
        min_val = min(values.values())
        max_val = max(values.values())
        if max_val == min_val:
            norm_cache[cache_key] = {player: 0.5 for player in values}
            return norm_cache[cache_key]
        if higher_is_better:
            norm_cache[cache_key] = {
                player: (value - min_val) / (max_val - min_val) for player, value in values.items()
            }
        else:
            norm_cache[cache_key] = {
                player: (max_val - value) / (max_val - min_val) for player, value in values.items()
            }
        return norm_cache[cache_key]

    title_order = ['Objective Player', 'Slayer', 'Support', 'Sniper', 'Survivor', 'All-Rounder']
    title_score_fns = {
        'Objective Player': lambda player: (
            0.6 * get_norm('obj_score_pg')[player]
            + 0.2 * get_norm('callouts_pg')[player]
            + 0.2 * get_norm('win_pct')[player]
        ),
        'Slayer': lambda player: (
            0.35 * get_norm('kills_pg')[player]
            + 0.25 * get_norm('dmg_per_min')[player]
            + 0.2 * get_norm('kda')[player]
            + 0.2 * get_norm('dmg_diff_pg')[player]
        ),
        'Support': lambda player: (
            0.5 * get_norm('assists_pg')[player]
            + 0.3 * get_norm('callouts_pg')[player]
            + 0.2 * get_norm('win_pct')[player]
        ),
        'Sniper': lambda player: (
            0.35 * get_norm('accuracy')[player]
            + 0.25 * get_norm('headshot_pct')[player]
            + 0.2 * get_norm('medal_Snipe')[player]
            + 0.2 * get_norm('medal_No_Scope')[player]
        ),
        'Survivor': lambda player: (
            0.5 * get_norm('avg_life')[player]
            + 0.3 * get_norm('kda')[player]
            + 0.2 * get_norm('deaths_pg', False)[player]
        ),
        'All-Rounder': lambda player: (
            get_norm('kills_pg')[player]
            + get_norm('assists_pg')[player]
            + get_norm('accuracy')[player]
            + get_norm('obj_score_pg')[player]
            + get_norm('win_pct')[player]
            + get_norm('kda')[player]
        ) / 6
    }

    writeup_metric_defs = [
        {'key': 'kills_pg', 'label': 'Kills/Game', 'higher_is_better': True, 'format': lambda v: format_value(v, 1)},
        {'key': 'assists_pg', 'label': 'Assists/Game', 'higher_is_better': True, 'format': lambda v: format_value(v, 1)},
        {'key': 'kda', 'label': 'KDA', 'higher_is_better': True, 'format': lambda v: format_value(v, 2)},
        {'key': 'accuracy', 'label': 'Accuracy', 'higher_is_better': True, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'dmg_per_min', 'label': 'Damage/Min', 'higher_is_better': True, 'format': lambda v: format_value(v, 0)},
        {'key': 'dmg_diff_pg', 'label': 'Damage Diff/Game', 'higher_is_better': True, 'format': lambda v: format_signed(v, 0)},
        {'key': 'obj_score_pg', 'label': 'Objective Score/Game', 'higher_is_better': True, 'format': lambda v: format_signed(v, 1)},
        {'key': 'callouts_pg', 'label': 'Callouts/Game', 'higher_is_better': True, 'format': lambda v: format_value(v, 1)},
        {'key': 'avg_life', 'label': 'Avg Life', 'higher_is_better': True, 'format': lambda v: format_value(v, 1)},
        {'key': 'medals_pg', 'label': 'Medals/Game', 'higher_is_better': True, 'format': lambda v: format_value(v, 2)},
        {'key': 'headshot_pct', 'label': 'Headshot %', 'higher_is_better': True, 'format': lambda v: format_value(v, 1, '%')},
        {'key': 'deaths_pg', 'label': 'Deaths/Game', 'higher_is_better': False, 'format': lambda v: format_value(v, 1)},
        {'key': 'suicides_pg', 'label': 'Suicides/Game', 'higher_is_better': False, 'format': lambda v: format_value(v, 2)},
        {'key': 'betrayals_pg', 'label': 'Betrayals/Game', 'higher_is_better': False, 'format': lambda v: format_value(v, 2)}
    ]

    def gap_score(values: list[float], value: float, higher_is_better: bool, is_best: bool) -> float:
        if len(values) < 2:
            return 0
        sorted_vals = sorted(values, reverse=higher_is_better)
        if is_best:
            return abs(value - sorted_vals[1])
        return abs(sorted_vals[-2] - value)

    def build_extremes(metric_defs: list[dict]) -> tuple[dict, dict, dict, dict]:
        strengths_strict = {player: [] for player in player_metrics}
        strengths_tied = {player: [] for player in player_metrics}
        weaknesses_strict = {player: [] for player in player_metrics}
        weaknesses_tied = {player: [] for player in player_metrics}

        for metric in metric_defs:
            values = {player: safe_float(player_metrics[player].get(metric['key'], 0)) for player in player_metrics}
            value_list = list(values.values())
            if not value_list:
                continue
            max_val = max(value_list)
            min_val = min(value_list)
            if max_val == min_val:
                continue

            if metric['higher_is_better']:
                best_players = [p for p, v in values.items() if v == max_val]
                worst_players = [p for p, v in values.items() if v == min_val]
            else:
                best_players = [p for p, v in values.items() if v == min_val]
                worst_players = [p for p, v in values.items() if v == max_val]

            for player in best_players:
                score = gap_score(value_list, values[player], metric['higher_is_better'], True)
                label = f"{metric['label']} ({metric['format'](values[player])})"
                entry = {'key': metric['key'], 'label': label, 'score': score, 'is_medal': metric['is_medal'], 'tier': 'extreme'}
                if len(best_players) == 1:
                    strengths_strict[player].append(entry)
                else:
                    strengths_tied[player].append(entry)

            for player in worst_players:
                score = gap_score(value_list, values[player], metric['higher_is_better'], False)
                label = f"{metric['label']} ({metric['format'](values[player])})"
                entry = {'key': metric['key'], 'label': label, 'score': score, 'is_medal': metric['is_medal'], 'tier': 'extreme'}
                if len(worst_players) == 1:
                    weaknesses_strict[player].append(entry)
                else:
                    weaknesses_tied[player].append(entry)

        return strengths_strict, strengths_tied, weaknesses_strict, weaknesses_tied

    def build_relative_candidates(metric_defs: list[dict]) -> tuple[dict, dict]:
        strength_candidates = {player: [] for player in player_metrics}
        weakness_candidates = {player: [] for player in player_metrics}

        for metric in metric_defs:
            values = {player: safe_float(player_metrics[player].get(metric['key'], 0)) for player in player_metrics}
            value_list = list(values.values())
            if not value_list:
                continue
            max_val = max(value_list)
            min_val = min(value_list)
            if max_val == min_val:
                continue

            for player, value in values.items():
                if metric['higher_is_better']:
                    strength_score = (value - min_val) / (max_val - min_val)
                else:
                    strength_score = (max_val - value) / (max_val - min_val)
                weakness_score = 1 - strength_score
                label = f"{metric['label']} ({metric['format'](value)})"
                strength_candidates[player].append({
                    'key': metric['key'],
                    'label': label,
                    'score': strength_score,
                    'is_medal': metric['is_medal'],
                    'tier': 'relative'
                })
                weakness_candidates[player].append({
                    'key': metric['key'],
                    'label': label,
                    'score': weakness_score,
                    'is_medal': metric['is_medal'],
                    'tier': 'relative'
                })

        return strength_candidates, weakness_candidates

    def build_writeup_candidates(metric_defs: list[dict]) -> tuple[dict, dict]:
        strength_candidates = {player: [] for player in player_metrics}
        weakness_candidates = {player: [] for player in player_metrics}

        for metric in metric_defs:
            values = {player: safe_float(player_metrics[player].get(metric['key'], 0)) for player in player_metrics}
            if not values:
                continue
            max_val = max(values.values())
            min_val = min(values.values())
            if max_val == min_val:
                continue
            norms = get_norm(metric['key'], metric['higher_is_better'])
            for player, value in values.items():
                entry = {
                    'key': metric['key'],
                    'label': metric['label'],
                    'value': value,
                    'score': norms[player],
                    'format': metric['format']
                }
                strength_candidates[player].append(entry)
                weakness_candidates[player].append(entry)

        return strength_candidates, weakness_candidates

    def select_top_entries(primary: list[dict], fallback: list[dict], extra: list[dict], target_count: int = 5) -> list[dict]:
        primary_sorted = sorted(primary, key=lambda x: x['score'], reverse=True)
        fallback_sorted = sorted(fallback, key=lambda x: x['score'], reverse=True)
        extra_sorted = sorted(extra, key=lambda x: x['score'], reverse=True)
        selected = []
        used_keys = set()
        for entry in primary_sorted:
            if len(selected) >= target_count:
                break
            selected.append(entry)
            used_keys.add(entry['key'])
        for entry in fallback_sorted:
            if len(selected) >= target_count:
                break
            if entry['key'] in used_keys:
                continue
            selected.append(entry)
            used_keys.add(entry['key'])
        for entry in extra_sorted:
            if len(selected) >= target_count:
                break
            if entry['key'] in used_keys:
                continue
            selected.append(entry)
            used_keys.add(entry['key'])
        return selected[:target_count]

    def select_writeup_entries(candidates: list[dict], count: int, reverse: bool, exclude_keys: set[str] | None = None) -> list[dict]:
        exclude_keys = exclude_keys or set()
        selected = []
        for entry in sorted(candidates, key=lambda x: x['score'], reverse=reverse):
            if entry['key'] in exclude_keys:
                continue
            selected.append(entry)
            exclude_keys.add(entry['key'])
            if len(selected) >= count:
                break
        return selected

    def join_phrases(items: list[str]) -> str:
        if not items:
            return ''
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def build_stat_span(entry: dict, css_class: str) -> str:
        label_text = html.escape(entry['label'])
        value_text = html.escape(entry['format'](entry['value']))
        return f"<span class='{css_class}'>{label_text} {value_text}</span>"

    stat_strengths_strict, stat_strengths_tied, stat_weaknesses_strict, stat_weaknesses_tied = build_extremes(stat_metric_defs)
    medal_strengths_strict, medal_strengths_tied, medal_weaknesses_strict, medal_weaknesses_tied = build_extremes(medal_metric_defs)
    stat_relative_strengths, stat_relative_weaknesses = build_relative_candidates(stat_metric_defs)
    medal_relative_strengths, medal_relative_weaknesses = build_relative_candidates(medal_metric_defs)
    writeup_strength_candidates, writeup_weakness_candidates = build_writeup_candidates(writeup_metric_defs)

    player_profiles = {}
    for player, metrics in player_metrics.items():
        strengths_entries = select_top_entries(
            stat_strengths_strict[player],
            stat_strengths_tied[player],
            stat_relative_strengths[player]
        )
        weakness_entries = select_top_entries(
            stat_weaknesses_strict[player],
            stat_weaknesses_tied[player],
            stat_relative_weaknesses[player]
        )
        medal_strengths_entries = select_top_entries(
            medal_strengths_strict[player],
            medal_strengths_tied[player],
            medal_relative_strengths[player]
        )
        medal_weaknesses_entries = select_top_entries(
            medal_weaknesses_strict[player],
            medal_weaknesses_tied[player],
            medal_relative_weaknesses[player]
        )

        title_scores = {name: score_fn(player) for name, score_fn in title_score_fns.items()}
        title = sorted(title_scores.items(), key=lambda item: (-item[1], title_order.index(item[0])))[0][0]
        article = 'an' if title[:1].lower() in 'aeiou' else 'a'

        top_medal_label = ''
        top_medal_value = 0.0
        for col in compare_medal_cols:
            value = safe_float(metrics.get(col, 0))
            if value > top_medal_value:
                top_medal_value = value
                top_medal_label = col.replace('medal_', '').replace('_', ' ').title()

        safe_player = html.escape(str(player))
        overview_sentence = (
            f"{safe_player} is {article} {title.lower()} averaging "
            f"{format_float(metrics['kills_pg'], 1)} kills/game, "
            f"{format_float(metrics['assists_pg'], 1)} assists/game, "
            f"and a {format_float(metrics['kda'], 2)} KDA with "
            f"{format_float(metrics['win_pct'], 1)}% wins, while objective score sits at "
            f"{format_signed(metrics['obj_score_pg'], 1)} per game."
        )

        writeup_strength_entries = select_writeup_entries(
            writeup_strength_candidates[player],
            2,
            True
        )
        writeup_strength_keys = {entry['key'] for entry in writeup_strength_entries}
        writeup_weakness_entries = select_writeup_entries(
            writeup_weakness_candidates[player],
            2,
            False,
            writeup_strength_keys
        )

        strength_bits = [build_stat_span(entry, 'stat-good') for entry in writeup_strength_entries]
        weakness_bits = [build_stat_span(entry, 'stat-bad') for entry in writeup_weakness_entries]
        strength_phrase = join_phrases(strength_bits)
        weakness_phrase = join_phrases(weakness_bits)

        if strength_phrase:
            strength_intro = f"Strengths show up in {strength_phrase}"
        else:
            strength_intro = "Strengths are spread across the stat line"

        if top_medal_label:
            safe_medal_label = html.escape(top_medal_label)
            medal_clause = f", led by {safe_medal_label} ({format_float(top_medal_value, 2)}/game)"
        else:
            medal_clause = ""

        strength_sentence = (
            f"{strength_intro}; medal pace is {format_float(metrics['medals_pg'], 2)} per game"
            f"{medal_clause}, and accuracy sits at {format_float(metrics['accuracy'], 1)}% with "
            f"{format_float(metrics['headshot_pct'], 1)}% headshots."
        )

        if weakness_phrase:
            improve_sentence = (
                f"Areas to sharpen include {weakness_phrase}, so tightening those should lift consistency."
            )
        else:
            improve_sentence = (
                "Areas to sharpen are minimal right now, but small gains in damage output and accuracy "
                "could lift consistency."
            )

        def format_entry(entry: dict, is_strength: bool) -> str:
            if entry.get('tier') == 'relative':
                prefix = 'Strong' if is_strength else 'Weak'
            else:
                prefix = 'Best' if is_strength else 'Worst'
            return f"{prefix} {entry['label']}"

        strengths = [format_entry(item, True) for item in strengths_entries]
        weaknesses = [format_entry(item, False) for item in weakness_entries]
        medal_strengths = [format_entry(item, True) for item in medal_strengths_entries]
        medal_weaknesses = [format_entry(item, False) for item in medal_weaknesses_entries]

        player_profiles[player] = {
            'games': metrics['games'],
            'win_pct': metrics['win_pct'],
            'kda': metrics['kda'],
            'kills_pg': metrics['kills_pg'],
            'deaths_pg': metrics['deaths_pg'],
            'assists_pg': metrics['assists_pg'],
            'accuracy': metrics['accuracy'],
            'dmg_pg': metrics['dmg_pg'],
            'dmg_diff_pg': metrics['dmg_diff_pg'],
            'dmg_per_min': metrics['dmg_per_min'],
            'medals_pg': metrics['medals_pg'],
            'strengths': strengths,
            'weaknesses': weaknesses,
            'medal_strengths': medal_strengths,
            'medal_weaknesses': medal_weaknesses,
            'title': title,
            'writeup': f"{overview_sentence} {strength_sentence} {improve_sentence}"
        }

    return player_profiles
@app.route('/match/<match_id>')
def match(match_id):
    df = cache.get()
    match_rows = build_match_details(df, match_id)
    return render_template('match.html',
                          app_title=APP_TITLE,
                          match_rows=match_rows,
                          match_id=match_id,
                          db_row_count=count_cache.get())


if __name__ == "__main__":
    port = int(os.getenv('HALO_WEB_PORT', '8091'))
    app.run(host='0.0.0.0', port=port, debug=False)
