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
MATCH_COLUMNS = ['match_id', 'date', 'player_gamertag', 'playlist', 'game_type', 'map', 'outcome', 'kills', 'deaths', 'assists', 'accuracy', 'shots_fired', 'shots_hit', 'damage_dealt', 'damage_taken', 'personal_score', 'score', 'callout_assists', 'medal_count', 'average_life_duration', 'headshots', 'headshot_kills', 'melee_kills', 'grenade_kills', 'power_weapon_kills', 'max_killing_spree', 'duration', 'pre_match_csr', 'post_match_csr', 'team_damage_dealt', 'enemy_team_damage_dealt', 'team_personal_score', 'team_score', 'kda', 'dmg/ka', 'dmg/death', 'dmg/min', 'dmg_difference']
MAJOR_STAT_COLUMNS = [('kills', 'Kills'), ('deaths', 'Deaths'), ('assists', 'Assists'), ('kda', 'KDA'), ('accuracy', 'Accuracy'), ('damage_dealt', 'Damage Dealt'), ('damage_taken', 'Damage Taken'), ('dmg_difference', 'Damage Diff'), ('shots_fired', 'Shots Fired'), ('shots_hit', 'Shots Hit'), ('medal_count', 'Medals'), ('personal_score', 'Personal Score'), ('obj_score', 'Objective Score'), ('callout_assists', 'Callouts'), ('headshots', 'Headshots'), ('melee_kills', 'Melee Kills'), ('grenade_kills', 'Grenade Kills'), ('power_weapon_kills', 'Power Weapon Kills'), ('average_life_duration', 'Avg Life'), ('dmg/death', 'Damage/Death')]
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
        select_cols = select_match_columns(columns)
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
        
        kills = pd.to_numeric(session_df.get('kills', 0), errors='coerce').fillna(0).sum() / games if games else 0
        deaths = pd.to_numeric(session_df.get('deaths', 0), errors='coerce').fillna(0).sum() / games if games else 0
        assists = pd.to_numeric(session_df.get('assists', 0), errors='coerce').fillna(0).sum() / games if games else 0
        kda = safe_kda(kills, assists, deaths)
        
        accuracy = 0
        if 'shots_fired' in session_df.columns and 'shots_hit' in session_df.columns:
            fired = pd.to_numeric(session_df['shots_fired'], errors='coerce').fillna(0).sum()
            hit = pd.to_numeric(session_df['shots_hit'], errors='coerce').fillna(0).sum()
            accuracy = hit / fired * 100 if fired > 0 else 0
        
        latest_csr = None
        if 'post_match_csr' in session_df.columns:
            post_vals = pd.to_numeric(session_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[-1]
        
        csr_delta = 0
        if 'pre_match_csr' in session_df.columns and 'post_match_csr' in session_df.columns:
            sorted_session = session_df.sort_values('date')
            pre_vals = pd.to_numeric(sorted_session['pre_match_csr'], errors='coerce')
            post_vals = pd.to_numeric(sorted_session['post_match_csr'], errors='coerce')
            pre_vals = pre_vals[pre_vals > 0]
            post_vals = post_vals[post_vals > 0]
            if not pre_vals.empty and not post_vals.empty:
                csr_delta = post_vals.iloc[-1] - pre_vals.iloc[0]
        
        rows.append({
            'player': player,
            'session_date': format_date(session_df['date'].max()),
            'csr': format_float(latest_csr, 1) if latest_csr else '-',
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kills': format_float(kills, 1),
            'deaths': format_float(deaths, 1),
            'assists': format_float(assists, 1),
            'kda': format_float(kda, 2),
            'accuracy': format_float(accuracy, 1),
            'csr_delta': format_signed(csr_delta, 0)
        })
    
    add_heatmap_classes(rows, {
        'csr': True, 'games': True, 'win_pct': True, 'kda': True,
        'kills': True, 'deaths': False, 'assists': True,
        'accuracy': True, 'csr_delta': True
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
        win_pct = wins / games * 100 if games > 0 else 0
        
        kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum() / games if games else 0
        deaths = pd.to_numeric(player_df.get('deaths', 0), errors='coerce').fillna(0).sum() / games if games else 0
        assists = pd.to_numeric(player_df.get('assists', 0), errors='coerce').fillna(0).sum() / games if games else 0
        kda = safe_kda(kills, assists, deaths)
        
        accuracy = 0
        if 'shots_fired' in player_df.columns and 'shots_hit' in player_df.columns:
            fired = pd.to_numeric(player_df['shots_fired'], errors='coerce').fillna(0).sum()
            hit = pd.to_numeric(player_df['shots_hit'], errors='coerce').fillna(0).sum()
            accuracy = hit / fired * 100 if fired > 0 else 0
        
        latest_csr = None
        if 'post_match_csr' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
        
        rows.append({
            'player': player,
            'csr': format_float(latest_csr, 1) if latest_csr else '-',
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kills': format_float(kills, 1),
            'deaths': format_float(deaths, 1),
            'assists': format_float(assists, 1),
            'kda': format_float(kda, 2),
            'accuracy': format_float(accuracy, 1)
        })
    
    add_heatmap_classes(rows, {
        'csr': True, 'games': True, 'win_pct': True, 'kda': True,
        'kills': True, 'deaths': False, 'assists': True, 'accuracy': True
    })
    
    rows.sort(key=lambda x: to_number(x['kda']) or 0, reverse=True)
    return rows


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
        win_pct = wins / games * 100 if games > 0 else 0
        
        total_kills = pd.to_numeric(player_df.get('kills', 0), errors='coerce').fillna(0).sum()
        total_deaths = pd.to_numeric(player_df.get('deaths', 0), errors='coerce').fillna(0).sum()
        total_assists = pd.to_numeric(player_df.get('assists', 0), errors='coerce').fillna(0).sum()
        
        kills_pg = total_kills / games if games else 0
        deaths_pg = total_deaths / games if games else 0
        assists_pg = total_assists / games if games else 0
        kda = safe_kda(kills_pg, assists_pg, deaths_pg)
        
        accuracy = 0
        if 'shots_fired' in player_df.columns and 'shots_hit' in player_df.columns:
            fired = pd.to_numeric(player_df['shots_fired'], errors='coerce').fillna(0).sum()
            hit = pd.to_numeric(player_df['shots_hit'], errors='coerce').fillna(0).sum()
            accuracy = hit / fired * 100 if fired > 0 else 0
        
        latest_csr = None
        if 'post_match_csr' in player_df.columns and 'date' in player_df.columns:
            sorted_df = player_df.sort_values('date', ascending=False)
            post_vals = pd.to_numeric(sorted_df['post_match_csr'], errors='coerce')
            post_vals = post_vals[post_vals > 0]
            if not post_vals.empty:
                latest_csr = post_vals.iloc[0]
        
        rows.append({
            'player': player,
            'csr': format_float(latest_csr, 1) if latest_csr else '-',
            'games': format_int(games),
            'win_pct': format_float(win_pct, 1),
            'kills': format_float(kills_pg, 1),
            'deaths': format_float(deaths_pg, 1),
            'assists': format_float(assists_pg, 1),
            'kda': format_float(kda, 2),
            'accuracy': format_float(accuracy, 1)
        })
    
    add_heatmap_classes(rows, {
        'csr': True, 'games': True, 'win_pct': True, 'kda': True,
        'kills': True, 'deaths': False, 'assists': True, 'accuracy': True
    })
    
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
        
        rows.append({
            'player': player,
            'games': format_int(games),
            'wins': format_int(wins),
            'losses': format_int(losses),
            'ties': format_int(ties),
            'win_pct': format_float(wins / games * 100 if games else 0, 1),
            'kills_pg': format_float(kills_pg, 1),
            'deaths_pg': format_float(deaths_pg, 1),
            'assists_pg': format_float(assists_pg, 1),
            'kda': format_float(kda, 2),
            'kd_ratio': format_float(kd_ratio, 2),
            'damage_pg': format_float(damage_pg, 0),
            'accuracy': format_pct(accuracy),
            'total_kills': format_int(total_kills),
            'total_deaths': format_int(total_deaths),
            'damage_diff': format_signed(total_damage_dealt - total_damage_taken, 0)
        })
    
    add_heatmap_classes(rows, {
        'games': True, 'win_pct': True, 'kda': True, 'kd_ratio': True,
        'kills_pg': True, 'deaths_pg': False, 'assists_pg': True,
        'damage_pg': True, 'accuracy': True
    })
    
    return rows


def build_session_history(df: pd.DataFrame, limit: int = 20) -> list:
    """Build recent session history across all players."""
    if df.empty or 'date' not in df.columns:
        return []
    
    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce', utc=True)
    working = working.dropna(subset=['date']).sort_values('date', ascending=False)
    
    if working.empty:
        return []
    
    sessions = []
    seen_matches = set()
    
    for player in unique_sorted(working['player_gamertag']):
        player_df = working[working['player_gamertag'] == player].sort_values('date', ascending=False)
        
        if player_df.empty:
            continue
        
        session_groups = []
        current_session = []
        prev_time = None
        
        for idx, row in player_df.iterrows():
            match_id = row.get('match_id')
            if match_id in seen_matches:
                continue
            
            match_time = row['date']
            
            if prev_time is None or (prev_time - match_time).total_seconds() / 60 <= 30:
                current_session.append(row)
                prev_time = match_time
            else:
                if current_session:
                    session_groups.append(pd.DataFrame(current_session))
                current_session = [row]
                prev_time = match_time
            
            seen_matches.add(match_id)
        
        if current_session:
            session_groups.append(pd.DataFrame(current_session))
        
        for session_df in session_groups[:5]:  # Top 5 sessions per player
            if session_df.empty:
                continue
            
            games = len(session_df)
            outcomes = session_df['outcome'].astype(str).str.lower() if 'outcome' in session_df.columns else pd.Series()
            wins = (outcomes == 'win').sum() if not outcomes.empty else 0
            
            sessions.append({
                'player': player,
                'date': format_date(session_df['date'].max()),
                'date_iso': format_iso(session_df['date'].max()),
                'games': games,
                'wins': wins,
                'record': f"{wins}-{games - wins}",
                'win_pct': format_float(wins / games * 100 if games else 0, 1)
            })
    
    sessions.sort(key=lambda x: x['date_iso'], reverse=True)
    return sessions[:limit]


def extract_objective_score(df: pd.DataFrame) -> pd.Series:
    """Extract objective score from various objective stat columns."""
    if df.empty:
        return pd.Series(dtype=float)
    
    obj_score = pd.Series(0.0, index=df.index)
    
    for prefix in OBJECTIVE_PREFIXES:
        score_col = f"{prefix}score"
        if score_col in df.columns:
            vals = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
            obj_score = obj_score + vals
    
    return obj_score


def build_objective_stats(df: pd.DataFrame, period: str = 'all') -> list:
    """Build objective statistics (CTF, Oddball, etc.)."""
    if df.empty:
        return []
    
    # Filter by period if needed
    working = df.copy()
    if period == 'session':
        # Get last session (stub for now - use last 10 games)
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
        obj_score = extract_objective_score(player_df)
        total_obj = obj_score.sum()
        avg_obj = total_obj / games if games else 0
        
        # CTF specific
        ctf_caps = pd.to_numeric(player_df.get('capture_the_flag_stats_flag_captures', 0), errors='coerce').fillna(0).sum()
        ctf_returns = pd.to_numeric(player_df.get('capture_the_flag_stats_flag_returns', 0), errors='coerce').fillna(0).sum()
        
        # Oddball specific
        oddball_time = pd.to_numeric(player_df.get('oddball_stats_time_as_skull_carrier_seconds', 0), errors='coerce').fillna(0).sum()
        oddball_kills = pd.to_numeric(player_df.get('oddball_stats_kills_as_skull_carrier', 0), errors='coerce').fillna(0).sum()
        
        # Zones specific
        zones_caps = pd.to_numeric(player_df.get('zones_stats_total_zone_captures', 0), errors='coerce').fillna(0).sum()
        zones_time = pd.to_numeric(player_df.get('zones_stats_total_zone_occupation_time_seconds', 0), errors='coerce').fillna(0).sum()
        
        rows.append({
            'player': player,
            'games': format_int(games),
            'obj_score': format_float(avg_obj, 1),
            'total_obj': format_int(total_obj),
            'ctf_caps': format_int(ctf_caps),
            'ctf_returns': format_int(ctf_returns),
            'oddball_time': format_int(oddball_time),
            'oddball_kills': format_int(oddball_kills),
            'zones_caps': format_int(zones_caps),
            'zones_time': format_int(zones_time)
        })
    
    add_heatmap_classes(rows, {
        'obj_score': True, 'total_obj': True, 'ctf_caps': True,
        'ctf_returns': True, 'oddball_time': True, 'zones_caps': True
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
        total_medals = pd.to_numeric(player_df.get('medal_count', 0), errors='coerce').fillna(0).sum()
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
                'map': normalize_map_name(row.get('map', '')),
                'mode': row.get('game_type', ''),
                'kills': format_int(row.get('kills', 0)),
                'deaths': format_int(row.get('deaths', 0)),
                'assists': format_int(row.get('assists', 0)),
                'kda': format_float(row.get('_kda_score', 0), 2),
                'reason': 'High KDA',
                'outcome': str(row.get('outcome', '')).title()
            })
    
    return highlights[:limit]


def build_hall_fame_shame(df: pd.DataFrame) -> tuple[list, list]:
    """Build hall of fame and hall of shame."""
    if df.empty:
        return [], []
    
    fame = []
    shame = []
    
    # Best KDA game
    if 'kills' in df.columns and 'deaths' in df.columns and 'assists' in df.columns:
        kills = pd.to_numeric(df['kills'], errors='coerce').fillna(0)
        deaths = pd.to_numeric(df['deaths'], errors='coerce').fillna(0).replace(0, 1)
        assists = pd.to_numeric(df['assists'], errors='coerce').fillna(0)
        kda = kills + assists / 3 - deaths
        
        best_idx = kda.idxmax()
        worst_idx = kda.idxmin()
        
        fame.append({
            'category': 'Best KDA',
            'player': df.loc[best_idx, 'player_gamertag'],
            'value': format_float(kda.loc[best_idx], 2),
            'context': f"{format_int(df.loc[best_idx, 'kills'])}K/{format_int(df.loc[best_idx, 'deaths'])}D/{format_int(df.loc[best_idx, 'assists'])}A"
        })
        
        shame.append({
            'category': 'Worst KDA',
            'player': df.loc[worst_idx, 'player_gamertag'],
            'value': format_float(kda.loc[worst_idx], 2),
            'context': f"{format_int(df.loc[worst_idx, 'kills'])}K/{format_int(df.loc[worst_idx, 'deaths'])}D/{format_int(df.loc[worst_idx, 'assists'])}A"
        })
    
    # Most kills
    if 'kills' in df.columns:
        kills_series = pd.to_numeric(df['kills'], errors='coerce').fillna(0)
        max_idx = kills_series.idxmax()
        fame.append({
            'category': 'Most Kills',
            'player': df.loc[max_idx, 'player_gamertag'],
            'value': format_int(kills_series.loc[max_idx]),
            'context': normalize_map_name(df.loc[max_idx, 'map']) if 'map' in df.columns else ''
        })
    
    # Most deaths (shame)
    if 'deaths' in df.columns:
        deaths_series = pd.to_numeric(df['deaths'], errors='coerce').fillna(0)
        max_idx = deaths_series.idxmax()
        shame.append({
            'category': 'Most Deaths',
            'player': df.loc[max_idx, 'player_gamertag'],
            'value': format_int(deaths_series.loc[max_idx]),
            'context': normalize_map_name(df.loc[max_idx, 'map']) if 'map' in df.columns else ''
        })
    
    # Best accuracy
    if 'accuracy' in df.columns:
        acc_series = pd.to_numeric(df['accuracy'], errors='coerce').fillna(0)
        max_idx = acc_series.idxmax()
        fame.append({
            'category': 'Best Accuracy',
            'player': df.loc[max_idx, 'player_gamertag'],
            'value': format_pct(acc_series.loc[max_idx]),
            'context': ''
        })
    
    return fame, shame


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
        
        avg_kills = pd.to_numeric(map_df.get('kills', 0), errors='coerce').fillna(0).mean()
        avg_duration = pd.to_numeric(map_df.get('duration', 0), errors='coerce').fillna(0).mean()
        
        rows.append({
            'map': map_name,
            'games': format_int(games),
            'wins': format_int(wins),
            'win_rate': format_float(wins / games * 100 if games else 0, 1),
            'avg_kills': format_float(avg_kills, 1),
            'avg_duration': format_int(avg_duration)
        })
    
    add_heatmap_classes(rows, {'games': True, 'win_rate': True, 'avg_kills': True})
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
            
            rows.append({
                'player': player,
                'map': map_name,
                'games': format_int(games),
                'win_rate': format_float(wins / games * 100 if games else 0, 1)
            })
    
    add_heatmap_classes(rows, {'win_rate': True})
    rows.sort(key=lambda x: (x['player'], -to_number(x['win_rate']) or 0))
    
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
    
    filtered = apply_filters(df, player, playlist, mode)
    
    lifetime_rows = build_lifetime_stats(filtered)
    session_rows = build_session_history(filtered, limit=20)
    
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
    mode_rows = build_breakdown(df, 'game_type', limit=50) if 'game_type' in df.columns else []
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
    trend_df = apply_trend_range(normalize_trend_df(df), range_key)
    
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
                          win_rate_trends={},
                          kda_trends=build_trend_data(trend_df, 'kda', 'kda') if 'kda' in trend_df.columns else {},
                          obj_score_trends={},
                          damage_min_trends={},
                          damage_diff_trends={},
                          accuracy_trends=build_trend_data(trend_df, 'accuracy', 'accuracy') if 'accuracy' in trend_df.columns else {},
                          kills_pg_trends=build_trend_data(trend_df, 'kills', 'kills') if 'kills' in trend_df.columns else {},
                          deaths_pg_trends=build_trend_data(trend_df, 'deaths', 'deaths') if 'deaths' in trend_df.columns else {},
                          max_spree_trends={},
                          duration_trends={},
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
