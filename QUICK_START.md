# 🚀 Quick Start Guide - Halo Stats Setup

Get Halo Stats running in **5 minutes**!

---

## What You Need

- 🎮 Xbox gamertag(s) to track
- ☁️ Microsoft account (free)
- 🐳 Docker + Docker Compose (optional)
- 📁 Git installed

---

## 3-Step Setup

### Step 1: Clone & Configure Environment

```bash
git clone https://github.com/yourusername/halostats.git
cd halostats
cp .env.example .env
```

### Step 2: Get Xbox API Credentials (5 min)

Go to [Azure Portal](https://portal.azure.com/):

1. Search **"App registrations"** → Click **"New registration"**
2. Name: `Halo Stats` → Register
3. Copy **Application (client) ID** → Add to `.env` as `HALO_CLIENT_ID`
4. Click **"Certificates & secrets"** → **"New client secret"**
5. Copy the **VALUE** (not ID) → Add to `.env` as `HALO_CLIENT_SECRET`

**Result in `.env`:**
```bash
HALO_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
HALO_CLIENT_SECRET=your_long_secret_string_here
HALO_DB_PASSWORD=any_secure_password_here
```

### Step 3: Add Players & Run

Edit `.env` and set players to track:

```bash
# Single player:
HALO_TRACKED_PLAYERS='[{"gamertag": "YourGamertag", "xuid": "2533274818160056"}]'

# Multiple players (find XUIDs at halowaypoint.com):
HALO_TRACKED_PLAYERS='[{"gamertag": "Player1", "xuid": "2533274818160056"}, {"gamertag": "Player2", "xuid": "2533274965035069"}]'
```

**Run with Docker:**
```bash
docker compose -f config/compose.yaml up --build
```

**Or Local Python:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/entrypoint.py    # Terminal 1
python src/webapp.py        # Terminal 2
```

---

## 🎯 Access Your App

Open your browser:

```
http://localhost:8090
```

✨ You're done! Stats should start loading.

---

## 📊 What It Shows

- **CSR Rankings** - Real-time rating with history
- **Match Stats** - Kills, deaths, accuracy, etc.
- **Leaderboards** - Rankings across all players
- **Trends** - Performance over time
- **Maps** - Win rates per map
- **Medals** - Achievements tracked
- And more!

---

## 🔍 Finding Player XUIDs

Need to find a player's XUID?

1. Go to [halowaypoint.com](https://www.halowaypoint.com)
2. Search the gamertag
3. Copy XUID from URL (e.g., `halowaypoint.com/stats/profile/2533274818160056`)

---

## ⚙️ Common Changes

### Change Players

Edit `.env`, update `HALO_TRACKED_PLAYERS`, restart app.

### Change Port

Edit `.env`:
```bash
HALO_WEB_PORT=8091  # Instead of 8090
```

### Change Update Frequency

Edit `.env`:
```bash
HALO_UPDATE_INTERVAL=30  # Check every 30 seconds (default: 60)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8090 in use | Change `HALO_WEB_PORT` in `.env` |
| "Invalid credentials" | Check Azure Client ID and Secret |
| Database error | Ensure `HALO_DB_PASSWORD` is set |
| No players loading | Check `HALO_TRACKED_PLAYERS` JSON format |
| Players not updating | Check internet connection and API limits |

---

## 📚 Full Guides

- **[Azure Setup](docs/AZURE_SETUP.md)** - Detailed Azure configuration
- **[Player Tracking](docs/PLAYER_TRACKING.md)** - How to add/change players
- **[Installation](docs/SETUP.md)** - Complete installation guide
- **[README](README.md)** - Full documentation

---

## 🆘 Need Help?

1. Check the docs in the `docs/` folder
2. Look at `.env.example` for all options
3. Check app logs for error messages
4. Open a GitHub issue with details

---

## 🎮 Your 5 Default Players

If you don't set `HALO_TRACKED_PLAYERS`, it uses:
- l 0cty l
- Zaidster7
- l P1N1 l
- l Viper18 l
- l Jordo l

Just replace them in `.env` with your own!

---

## ✅ You're Ready!

You now have a personal Halo stats tracker for any players you want!

**Next:** Add your own players to `.env` and watch the data come in. 🚀

---

**Questions?** Check the documentation or create a GitHub issue!
