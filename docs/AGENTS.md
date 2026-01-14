# Agent Notes

## Project summary
- Python scraper runs via `entrypoint.py` (loops `auth.py` + `stats.py`).
- Flask UI lives in `webapp.py` with Jinja templates in `templates/` and assets in `static/`.
- Data is stored in Postgres table `halo_match_stats`; files like `tokens.json` live under `HALO_DATA_DIR`.

## Setup and run
- Copy `.env.example` to `.env` and fill in `HALO_DB_PASSWORD`, `HALO_CLIENT_ID`, `HALO_CLIENT_SECRET`.
- Docker (recommended): `docker compose up --build`
  - Web UI: http://localhost:8091
  - Adminer: http://localhost:8088
- Local (no Docker):
  - `python -m venv .venv`
  - `pip install -r requirements.txt`
  - Run scraper: `python entrypoint.py`
  - Run web: `python webapp.py`

## Testing
- No automated test suite.
- Optional smoke check (requires DB): `python test_ranked.py`

## Pointers for edits
- Table sorting logic: `static/app.js`
- Templates: `templates/`
- Styles: `static/styles.css`
