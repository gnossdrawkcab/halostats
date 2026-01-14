# NOTE: This file was recovered from an earlier backup but appears to be incomplete.
# Many build_* functions are truncated or missing implementations.
# The file compiles but may have runtime issues until all functions are restored.

import json
import os
import re
import time
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
STATUS_PATH = data_path(os.getenv('HALO_STATUS_NAME', 'update_status.json'))
SETTINGS_PATH = data_path('settings.json')
CACHE_TTL = int(os.getenv('HALO_CACHE_TTL', '120'))
DB_COUNT_TTL = int(os.getenv('HALO_DB_COUNT_TTL', '60'))
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


@app.template_filter('player_class')
def player_class_filter(player_name):
    return get_player_class(player_name)


@app.context_processor
def utility_processor():
    return dict(get_player_class=get_player_class)


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
            tie_conditions.append('COALESCE(duration, 0) < 60')
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
            tie_conditions.append('COALESCE(duration, 0) < 60')
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


def build_outlier_spotlight(df: pd.DataFrame) -> list[dict]:
    """Build spotlight showing each player's best stats."""
    if df.empty or 'player_gamertag' not in df.columns:
        return []
    
    ranked_df = df[df['playlist'].astype(str).str.contains('Ranked', case=False, na=False)].copy() if 'playlist' in df.columns else df.copy()
    
    rows = []
    for player in unique_sorted(ranked_df['player_gamertag']):
        player_df = ranked_df[ranked_df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        highlights = []
        
        # Best KDA game
        if 'kills' in player_df.columns and 'deaths' in player_df.columns and 'assists' in player_df.columns:
            kills = pd.to_numeric(player_df['kills'], errors='coerce').fillna(0)
            deaths = pd.to_numeric(player_df['deaths'], errors='coerce').fillna(0).replace(0, 1)
            assists = pd.to_numeric(player_df['assists'], errors='coerce').fillna(0)
            kda = kills + assists / 3 - deaths
            max_kda = kda.max()
            highlights.append(f'Best KDA: {max_kda:.2f}')
        
        # Best accuracy
        if 'accuracy' in player_df.columns:
            acc = pd.to_numeric(player_df['accuracy'], errors='coerce').fillna(0)
            max_acc = acc.max()
            if max_acc <= 1:
                max_acc *= 100
            highlights.append(f'Best Accuracy: {max_acc:.1f}%')
        
        # Most kills
        if 'kills' in player_df.columns:
            kills = pd.to_numeric(player_df['kills'], errors='coerce').fillna(0)
            max_kills = int(kills.max())
            highlights.append(f'Most Kills: {max_kills}')
        
        # Most medals
        if 'medal_count' in player_df.columns:
            medals = pd.to_numeric(player_df['medal_count'], errors='coerce').fillna(0)
            max_medals = int(medals.max())
            highlights.append(f'Most Medals: {max_medals}')
        
        while len(highlights) < 4:
            highlights.append('-')
        
        rows.append({'player': player, 'highlights': highlights[:4]})
    
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
    working = working.dropna(subset=['date']).sort_values('date', ascending=False)
    
    if working.empty:
        return []
    
    if isinstance(limit, int) and limit > 0:
        working = working.head(limit)
    
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


def build_medal_stats(df: pd.DataFrame) -> tuple[list, list]:
    """Build medal statistics - returns (player_totals, medal_breakdown)."""
    if df.empty:
        return [], []
    
    # Get all medal columns
    medal_cols = [col for col in df.columns if col.startswith('medal_') and not col.startswith('medal_count')]
    
    player_rows = []
    for player in unique_sorted(df['player_gamertag']):
        player_df = df[df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        games = len(player_df)
        total_medals = pd.to_numeric(player_df['medal_count'], errors='coerce').fillna(0).sum() if 'medal_count' in player_df.columns else 0
        avg_medals = total_medals / games if games else 0
        
        player_rows.append({
            'player': player,
            'games': format_int(games),
            'total_medals': format_int(total_medals),
            'avg_medals': format_float(avg_medals, 1)
        })
    
    add_heatmap_classes(player_rows, {'total_medals': True, 'avg_medals': True})
    
    # Medal breakdown by type
    medal_rows = []
    for col in sorted(medal_cols)[:50]:  # Top 50 medals
        medal_name = col.replace('medal_', '').replace('_', ' ').title()
        total_count = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
        
        if total_count > 0:
            medal_rows.append({
                'medal_name': medal_name,
                'total': format_int(total_count),
                'avg_per_game': format_float(total_count / len(df), 2)
            })
    
    medal_rows.sort(key=lambda x: to_number(x['total']) or 0, reverse=True)
    
    return player_rows, medal_rows


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
    
    def streaks(player_df: pd.DataFrame) -> tuple[int, int]:
        if 'outcome' not in player_df.columns or 'date' not in player_df.columns:
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
    
    fame_rows = []
    shame_rows = []
    
    for player in unique_sorted(df['player_gamertag']):
        player_df = df[df['player_gamertag'] == player]
        if player_df.empty:
            continue
        
        win_streak, loss_streak = streaks(player_df)
        
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
            
            if games >= 10:  # Minimum games threshold
                rows.append({
                    'rank': 0,
                    'player': player,
                    'value': format_float(wins / games * 100, 1),
                    'win_rate': format_float(wins / games * 100, 1),
                    'context': f'{wins}-{games - wins}'
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


@app.route('/')
def index():
    df = cache.get()
    player = request.args.get('player', 'all')
    playlist = request.args.get('playlist', 'all')
    mode = request.args.get('mode', 'all')
    filtered = apply_filters(df, player, playlist, mode)
    
    csr_overview_rows = build_csr_overview(df)
    csr_overview_trends = build_csr_trends(apply_trend_range(normalize_trend_df(df), '90'))
    outlier_rows = build_outlier_spotlight(df)
    ranked_arena_rows = build_ranked_arena_summary(df)
    ranked_arena_30day_rows = build_ranked_arena_30day(df)
    ranked_arena_lifetime_rows = build_ranked_arena_lifetime(df)
    
    players_list = unique_sorted(df['player_gamertag']) if not df.empty and 'player_gamertag' in df.columns else []
    map_rows = build_breakdown(filtered, 'map')
    playlist_rows = build_breakdown(filtered, 'playlist')
    cards = build_cards(filtered)
    
    status = load_status()
    last_update = status.get('last_update')
    
    return render_template('index.html',
                          app_title=APP_TITLE,
                          csr_overview_rows=csr_overview_rows,
                          csr_overview_trends=csr_overview_trends,
                          outlier_rows=outlier_rows,
                          ranked_arena_rows=ranked_arena_rows,
                          ranked_arena_30day_rows=ranked_arena_30day_rows,
                          ranked_arena_lifetime_rows=ranked_arena_lifetime_rows,
                          players=players_list,
                          map_rows=map_rows,
                          playlist_rows=playlist_rows,
                          cards=cards,
                          last_update=last_update,
                          playlists=unique_sorted(df['playlist']) if not df.empty else [],
                          modes=unique_sorted(df['game_type']) if not df.empty else [],
                          selected_player=player,
                          selected_playlist=playlist,
                          selected_mode=mode,
                          db_row_count=count_cache.get())


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    df = cache.get()
    all_players = unique_sorted(df['player_gamertag']) if not df.empty else []
    
    if request.method == 'POST':
        match_limit_raw = request.form.get('match_limit', '100')
        try:
            match_limit = int(match_limit_raw)
            if match_limit < 0:
                match_limit = 0
        except ValueError:
            match_limit = 100
        
        try:
            update_interval = int(request.form.get('update_interval', '60'))
            if update_interval < 1:
                update_interval = 60
        except ValueError:
            update_interval = 60
        
        force_refresh = request.form.get('force_refresh') in ['1', 'true', 'on', 'yes']
        
        settings = {
            'match_limit': match_limit,
            'update_interval': update_interval,
            'force_refresh': force_refresh
        }
        save_settings(settings)
        
        return render_template('settings.html',
                              app_title=APP_TITLE,
                              players=all_players,
                              match_limit=match_limit,
                              update_interval=update_interval,
                              force_refresh=force_refresh,
                              message='Settings saved!',
                              last_update=load_status().get('last_update'),
                              db_row_count=count_cache.get())
    
    settings = load_settings()
    return render_template('settings.html',
                          app_title=APP_TITLE,
                          players=all_players,
                          match_limit=settings.get('match_limit', 100),
                          update_interval=settings.get('update_interval', 60),
                          force_refresh=bool(settings.get('force_refresh', False)),
                          message=None,
                          last_update=load_status().get('last_update'),
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
    limit_key = request.args.get('limit', 'all')
    
    filtered = apply_filters(df, player, playlist, mode)
    
    lifetime_rows = build_lifetime_stats(filtered)
    limit = None
    if limit_key and limit_key != 'all':
        try:
            limit = int(limit_key)
        except ValueError:
            limit = None
        if limit is not None and limit <= 0:
            limit = None
    session_rows = build_session_history(filtered, limit=limit)
    
    status = load_status()
    
    return render_template('lifetime.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          lifetime_rows=lifetime_rows,
                          session_rows=session_rows,
                          last_update=status.get('last_update'),
                          playlists=unique_sorted(df['playlist']) if not df.empty else [],
                          modes=unique_sorted(df['game_type']) if not df.empty else [],
                          selected_player=player,
                          selected_playlist=playlist,
                          selected_mode=mode,
                          selected_limit=limit_key,
                          db_row_count=count_cache.get())


@app.route('/compare')
def compare():
    """Player comparison page."""
    df = cache.get()
    
    comparison_rows = build_ranked_arena_lifetime(df)
    players_list = unique_sorted(df['player_gamertag']) if not df.empty else []
    status = load_status()
    
    return render_template('compare.html',
                          app_title=APP_TITLE,
                          comparison_rows=comparison_rows,
                          players=players_list,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


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
    
    medal_players, medal_rows = build_medal_stats(df)
    
    status = load_status()
    
    return render_template('medals.html',
                          app_title=APP_TITLE,
                          players=unique_sorted(df['player_gamertag']) if not df.empty else [],
                          ranked_arena_medal_players=medal_players,
                          ranked_arena_medal_rows=medal_rows,
                          medal_players=medal_players,
                          medal_rows=medal_rows,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


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
def player_profile(player_name: str):
    """Individual player profile page."""
    df = cache.get()
    
    if df.empty:
        return render_template('player.html',
                              app_title=APP_TITLE,
                              player_name=player_name,
                              error='No data available',
                              db_row_count=count_cache.get())
    
    all_players = unique_sorted(df['player_gamertag']) if not df.empty else []
    
    if player_name not in all_players:
        return render_template('player.html',
                              app_title=APP_TITLE,
                              player_name=player_name,
                              error='Player not found',
                              db_row_count=count_cache.get())
    
    presence = load_presence()
    status = load_status()
    
    # Get player-specific CSR info
    csr_overview = build_csr_overview(df)
    player_csr = next((r for r in csr_overview if r['player'] == player_name), {})
    
    # Last session
    ranked_sessions = build_ranked_arena_summary(df)
    last_session = next((r for r in ranked_sessions if r['player'] == player_name), {})
    
    return render_template('player.html',
                          app_title=APP_TITLE,
                          player_name=player_name,
                          players=all_players,
                          is_online=is_player_online(presence, player_name),
                          current_csr=player_csr.get('current_csr', '-'),
                          max_csr=player_csr.get('max_csr', '-'),
                          current_streak=0,  # Stub
                          last_session=last_session,
                          avg_30day={},  # Stub
                          comparison={},  # Stub
                          match_history=[],  # Stub
                          map_stats=[],  # Stub
                          teammate_stats=[],  # Stub
                          player_win_corr=[],  # Stub
                          csr_history=[],  # Stub - CSR history chart data
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/weapons')
def weapons():
    """Weapon statistics page."""
    df = cache.get()
    
    # Weapon stats (stub for now)
    weapon_rows = []
    accuracy_trend = {}
    
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
                          win_rate_trends=build_trend_data(trend_df, 'win_rate', 'win_rate') if 'win_rate' in trend_df.columns else {},
                          kda_trends=build_trend_data(trend_df, 'kda', 'kda') if 'kda' in trend_df.columns else {},
                          obj_score_trends=build_trend_data(trend_df, 'obj_score', 'obj_score') if 'obj_score' in trend_df.columns else {},
                          damage_min_trends=build_trend_data(trend_df, 'dmg_min', 'dmg_min') if 'dmg_min' in trend_df.columns else {},
                          damage_diff_trends=build_trend_data(trend_df, 'dmg_diff', 'dmg_diff') if 'dmg_diff' in trend_df.columns else {},
                          accuracy_trends=build_trend_data(trend_df, 'accuracy', 'accuracy') if 'accuracy' in trend_df.columns else {},
                          kills_pg_trends=build_trend_data(trend_df, 'kills_pg', 'kills_pg') if 'kills_pg' in trend_df.columns else {},
                          deaths_pg_trends=build_trend_data(trend_df, 'deaths_pg', 'deaths_pg') if 'deaths_pg' in trend_df.columns else {},
                          max_spree_trends=build_trend_data(trend_df, 'max_spree', 'max_spree') if 'max_spree' in trend_df.columns else {},
                          duration_trends=build_trend_data(trend_df, 'duration_min', 'duration_min') if 'duration_min' in trend_df.columns else {},
                          activity_heatmap=[],
                          win_corr_overall=[],
                          trend_ranges=trend_ranges,
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/insights')
def insights():
    """Insights and comparisons page."""
    df = cache.get()
    
    players_list = unique_sorted(df['player_gamertag']) if not df.empty else []
    player = request.args.get('player', 'all')
    
    status = load_status()
    
    return render_template('insights.html',
                          app_title=APP_TITLE,
                          players=players_list,
                          selected_player=player,
                          start_a='',
                          end_a='',
                          start_b='',
                          end_b='',
                          range_a_label='Range A',
                          range_b_label='Range B',
                          session_compare_rows=[],  # Stub
                          clutch_rows=[],  # Stub
                          role_rows=[],  # Stub
                          momentum_rows=[],  # Stub
                          veto_rows=[],  # Stub
                          consistency_rows=[],  # Stub
                          notable_rows=[],  # Stub
                          last_update=status.get('last_update'),
                          db_row_count=count_cache.get())


@app.route('/leaderboard')
def leaderboard():
    """Leaderboards page."""
    df = cache.get()
    
    period = request.args.get('period', 'all')
    
    leaders = {
        'csr_leaders': build_leaderboard(df, 'csr'),
        'csr_gained_leaders': [],
        'win_rate_leaders': build_leaderboard(df, 'win_rate'),
        'kda_leaders': build_leaderboard(df, 'kda'),
        'accuracy_leaders': [],
        'streak_leaders': [],
        'kills_leaders': [],
        'games_leaders': []
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


if __name__ == '__main__':
    port = int(os.getenv('HALO_WEB_PORT', '8090'))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        print(f'🔥 Starting in DEBUG mode with hot reload on port {port}')
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
    else:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port)
