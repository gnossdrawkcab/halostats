# 🎮 Halo Stats - Competitive Ranking & Performance Analytics

A comprehensive web application for tracking and analyzing competitive Halo match statistics, CSR progression, and team performance metrics.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey?logo=flask)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)

## ✨ Features

- **CSR Tracking** - Real-time ranking with historical trends
- **Match Analytics** - Detailed stats per player (KDA, accuracy, win rate)
- **Leaderboards** - Global rankings and player comparison
- **Trend Analysis** - 7/30/90/365 day performance trends
- **Map Statistics** - Win rates per map and game mode
- **Advanced Metrics** - Objective stats, medals, highlights, and more
- **Hall of Fame/Shame** - Notable achievements and records

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/halostats.git
cd halostats

# Setup environment
cp .env.example .env
# Edit .env - add HALO_CLIENT_ID, HALO_CLIENT_SECRET, HALO_DB_PASSWORD

# Start application
docker compose -f config/compose.yaml up --build

# Open in browser: http://localhost:8090
```

### Option 2: Local Python

```bash
# Clone repository
git clone https://github.com/yourusername/halostats.git
cd halostats

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install & run
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials

python src/entrypoint.py    # In terminal 1
python src/webapp.py        # In terminal 2
```

## 📁 Project Structure

```
halostats/
├── src/                    # Source code
│   ├── webapp.py          # Flask web application
│   ├── auth.py            # Authentication
│   ├── stats.py           # Statistics processing
│   ├── entrypoint.py      # Scraper & entry point
│   └── halo_paths.py      # API utilities
├── config/                # Configuration
│   ├── compose.yaml       # Docker Compose
│   └── Dockerfile         # Docker image
├── tests/                 # Test suite
├── templates/             # HTML templates
├── static/                # CSS, JavaScript
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## 🔧 Configuration

See `.env.example` for all variables. Key ones:

```bash
HALO_CLIENT_ID=your_api_id
HALO_CLIENT_SECRET=your_api_secret
HALO_DB_PASSWORD=postgres_password
HALO_WEB_PORT=8090
HALO_SITE_TITLE=👑 Scrim Kings
```

## 🔑 Getting Halo API Credentials

1. Visit [Halo Developer Portal](https://developer.xbox.com/)
2. Register application
3. Get OAuth credentials
4. Add to `.env`

## 📊 Pages & Features

- **Home** - CSR overview, recent matches
- **Lifetime** - All-time statistics
- **Compare** - Player comparison
- **Leaderboard** - Global rankings
- **Trends** - Historical analysis
- **Maps** - Map-specific stats
- **Advanced** - Objective modes
- **Medals** - Achievements
- **Highlights** - Best games
- **Hall of Fame** - Records

## 🧪 Testing

```bash
python -m pytest tests/
```

## 🐛 Troubleshooting

**Database connection?** Check `.env` variables and ensure PostgreSQL is running.

**API auth issues?** Verify `HALO_CLIENT_ID` and `HALO_CLIENT_SECRET`.

**Port in use?** Change `HALO_WEB_PORT` in `.env`.

## 📞 Support

- GitHub Issues for bug reports
- GitHub Discussions for questions
- Check [docs/](docs/) for detailed guides

## 📄 License

MIT License - see LICENSE file

## ⚠️ Disclaimer

Unofficial project. Not affiliated with Bungie, Microsoft, or Xbox.
Halo is a trademark of Bungie and/or Microsoft Corporation.
