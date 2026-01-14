# 🎮 Halo Stats - Competitive Ranking & Performance Analytics

A comprehensive web application for tracking and analyzing competitive Halo Infinite match statistics, CSR progression, and team performance metrics with customizable player tracking.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey?logo=flask)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)

## ✨ Features

- **Flexible Player Tracking** - Track any players by their Xbox XUID/gamertag
- **CSR Tracking** - Real-time ranking with historical trends
- **Match Analytics** - Detailed stats per player (KDA, accuracy, win rate)
- **Leaderboards** - Global rankings and player comparison
- **Trend Analysis** - 7/30/90/365 day performance trends
- **Map Statistics** - Win rates per map and game mode
- **Advanced Metrics** - Objective stats, medals, highlights, and more
- **Hall of Fame/Shame** - Notable achievements and records

## 📋 Prerequisites

Before you start, you'll need:

1. **Xbox Developer Account** - For API credentials
2. **PostgreSQL 13+** - For data storage (Docker handles this)
3. **Python 3.9+** - For local development
4. **Xbox Gamertag(s)** - The player(s) you want to track

## 🚀 Quick Start (5 Minutes)

### Option 1: Docker (Recommended) ✨

**Step 1: Clone and setup**
```bash
git clone https://github.com/yourusername/halostats.git
cd halostats
cp .env.example .env
```

**Step 2: Configure players to track**

Edit `.env` and set `HALO_TRACKED_PLAYERS` with your players:

```bash
# Single player
HALO_TRACKED_PLAYERS='[{"gamertag": "YourGamertag", "xuid": "1234567890123456"}]'

# Multiple players
HALO_TRACKED_PLAYERS='[{"gamertag": "Player1", "xuid": "1234567890123456"}, {"gamertag": "Player2", "xuid": "9876543210987654"}]'
```

**Step 3: Add Xbox API credentials**

Edit `.env` and add:
```bash
HALO_CLIENT_ID=your_client_id
HALO_CLIENT_SECRET=your_client_secret
HALO_DB_PASSWORD=your_secure_password
```

**Step 4: Run**
```bash
docker compose -f config/compose.yaml up --build
```

Open http://localhost:8090 in your browser!

### Option 2: Local Python Setup

```bash
# Clone & setup
git clone https://github.com/yourusername/halostats.git
cd halostats
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your credentials and players

# Run (in two terminals)
python src/entrypoint.py    # Terminal 1 - Scraper
python src/webapp.py        # Terminal 2 - Web UI
```

---

## 🔐 Getting Xbox API Credentials (Azure)

The app uses Xbox API via Azure credentials. Here's how to set up:

### Step 1: Create Azure App Registration

1. Visit [Azure Portal](https://portal.azure.com/)
2. Sign in with your Microsoft account
3. Search for **"App registrations"** and click it
4. Click **"New registration"**
5. Enter app name: `Halo Stats`
6. Select **"Single tenant"**
7. Click **"Register"**

### Step 2: Get Client Credentials

1. Go to your new app
2. Click **"Certificates & secrets"** (left menu)
3. Click **"New client secret"**
4. Add description: `Halo Stats App`
5. Click **"Add"**
6. **Copy the secret VALUE** (not the ID!) - add this to `.env` as `HALO_CLIENT_SECRET`

### Step 3: Get Client ID

1. Go back to **"Overview"** tab
2. Copy **"Application (client) ID"** - add this to `.env` as `HALO_CLIENT_ID`

### Step 4: Configure API Permissions

1. Click **"API permissions"** (left menu)
2. Click **"Add a permission"**
3. Search for **"Xbox Services"**
4. Select it and check the permissions your app needs
5. Click **"Add permissions"**

Your `.env` should now have:
```bash
HALO_CLIENT_ID=your_application_client_id
HALO_CLIENT_SECRET=your_client_secret_value
```

---

## 👥 Tracking Players

### Add Players to Track

Edit your `.env` file and update `HALO_TRACKED_PLAYERS`:

```bash
HALO_TRACKED_PLAYERS='[
  {"gamertag": "Player One", "xuid": "2533274818160056"},
  {"gamertag": "Player Two", "xuid": "2533274965035069"},
  {"gamertag": "Player Three", "xuid": "2533274804338345"}
]'
```

### How to Find a Player's XUID

You can find XUID from several sources:

1. **Halo Waypoint** - Visit [halowaypoint.com](https://www.halowaypoint.com) and search for the player
2. **TrueAchievements** - Search the player on [trueachievements.com](https://www.trueachievements.com)
3. **API Call** - Use the Halo API directly with their gamertag

### Changing Players

To track different players, just update the `.env` file and restart:

```bash
# Docker
docker compose -f config/compose.yaml restart

# Local
# Stop and re-run python src/entrypoint.py
```

---

## 📁 Project Structure

```
halostats/
├── src/                        # Application source code
│   ├── webapp.py              # Flask web interface
│   ├── auth.py                # Xbox authentication
│   ├── stats.py               # Data scraping & processing
│   ├── entrypoint.py          # Main scraper loop
│   └── halo_paths.py          # Halo API utilities
│
├── config/                     # Configuration & deployment
│   ├── compose.yaml           # Docker Compose setup
│   └── Dockerfile             # Container definition
│
├── templates/                  # Web interface (HTML)
│   ├── index.html            # Home page
│   ├── lifetime.html         # Lifetime stats
│   ├── compare.html          # Player comparison
│   ├── leaderboard.html      # Rankings
│   ├── trends.html           # Trend analysis
│   └── ...                   # Other pages
│
├── static/                     # Web assets
│   ├── app.js                # JavaScript
│   └── styles.css            # Styling
│
├── docs/                       # Additional documentation
├── tests/                      # Test suite
├── requirements.txt           # Python dependencies
├── .env.example              # Config template (COPY THIS!)
└── README.md                 # This file
```

---

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `HALO_TRACKED_PLAYERS` | Players to track (JSON) | See above |
| `HALO_CLIENT_ID` | Xbox API client ID | From Azure |
| `HALO_CLIENT_SECRET` | Xbox API secret | From Azure |
| `HALO_DB_PASSWORD` | Database password | Any secure string |
| `HALO_DB_HOST` | Database host | `localhost` or `halostatsapi` |
| `HALO_WEB_PORT` | Web server port | `8090` |
| `HALO_SITE_TITLE` | Browser title | `Halo Stats` |
| `HALO_TZ` | Timezone | `US/Eastern` |
| `HALO_UPDATE_INTERVAL` | Scraper interval (seconds) | `60` |
| `HALO_MATCH_LIMIT` | Matches to fetch per call | `500` |

Full reference: See `.env.example`

---

## 🌐 Web Pages

Once running, access:

- **Home** - http://localhost:8090 - CSR overview
- **Lifetime** - Player lifetime statistics
- **Compare** - Side-by-side player comparison
- **Leaderboard** - Global rankings
- **Trends** - Historical analysis
- **Maps** - Map-specific statistics
- **Advanced** - Objective mode stats
- **Medals** - Achievement tracking
- **Highlights** - Notable games
- **Hall of Fame** - Records & achievements

---

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

---

## 🐛 Troubleshooting

### "Connection refused" - Database

**Docker:**
```bash
docker compose -f config/compose.yaml ps
docker compose -f config/compose.yaml logs db
```

**Local:**
- Ensure PostgreSQL is running
- Check connection settings in `.env`

### "Invalid credentials" - Xbox API

- Verify `HALO_CLIENT_ID` and `HALO_CLIENT_SECRET` in `.env`
- Ensure they're from Azure App Registration
- Check that app has required API permissions

### "Port 8090 in use"

Change in `.env`:
```bash
HALO_WEB_PORT=8091
```

### "Can't find XUID"

Verify the XUID format (should be 16 digits):
- Wrong: `Player123` (that's a gamertag)
- Right: `2533274818160056` (that's a XUID)

Check [halowaypoint.com](https://www.halowaypoint.com) for the correct XUID.

---

## 📊 Viewing Data

The app stores data in PostgreSQL. To inspect:

**Via Web UI:**
- All stats visible in the web interface

**Via Database:**
```bash
# Docker
docker compose -f config/compose.yaml exec db psql -U postgres -d halostatsapi

# Local
psql -U postgres -d halostatsapi
```

---

## 📞 Support

- 📖 [Setup Guide](docs/SETUP.md) - Detailed installation
- 🤝 [Contributing Guide](CONTRIBUTING.md) - How to help
- 🐛 Issues - Report bugs on GitHub
- 💬 Discussions - Ask questions

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## ⚠️ Disclaimer

Unofficial project. Not affiliated with Bungie, Microsoft, or Xbox.
Halo is a trademark of Bungie and/or Microsoft Corporation.

**Use at your own risk** and respect Microsoft's API terms of service.
