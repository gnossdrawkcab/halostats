# Testing Instructions for Halo Stats Web App

## ⚠️ IMPORTANT: Test Before Pushing to Production!

This guide covers how to test your webapp locally before deploying to the live site.

---

## Quick Test (Syntax & Structure)

Run the automated test suite:

```bash
python test_webapp.py
```

This will verify:
- ✓ All imports work
- ✓ All 17 routes are defined
- ✓ Cache initializes correctly
- ✓ Database connection works (if available)
- ✓ All required build functions exist

---

## Full Local Test (Docker - Recommended)

The safest way to test is using Docker, which mirrors production:

### 1. Start the full stack:
```bash
docker compose up --build
```

### 2. Access the site:
- Web UI: http://localhost:8091
- Adminer (DB): http://localhost:8088

### 3. Test all pages:
Click through each page and verify:
- [ ] Home (/) - CSR overview, arena stats
- [ ] Lifetime (/lifetime) - Lifetime stats
- [ ] Compare (/compare) - Player comparison
- [ ] Settings (/settings) - Configuration
- [ ] Advanced (/advanced) - Objective stats
- [ ] Medals (/medals) - Medal statistics
- [ ] Highlights (/highlights) - Best games
- [ ] Columns (/columns) - Available data columns
- [ ] Player (/player/PlayerName) - Individual profiles
- [ ] Weapons (/weapons) - Weapon stats
- [ ] Hall (/hall) - Hall of fame/shame
- [ ] Maps (/maps) - Map statistics
- [ ] Trends (/trends) - Trend analysis
- [ ] Insights (/insights) - Advanced insights
- [ ] Suggestions (/suggestions) - Feature requests
- [ ] Leaderboard (/leaderboard) - Top players

### 4. Stop when done:
```bash
docker compose down
```

---

## Alternative: Local Python Test (No Docker)

If you can't use Docker:

### 1. Setup:
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set environment variables:
```bash
# Windows PowerShell
$env:HALO_DB_PASSWORD="your_password"
$env:HALO_DB_HOST="localhost"  # or your DB host
$env:FLASK_DEBUG="True"
```

### 3. Run the app:
```bash
python webapp.py
```

### 4. Test:
- Open http://localhost:8090
- Click through all pages
- Check browser console for JavaScript errors
- Check terminal for Python errors

### 5. Stop:
Press `Ctrl+C`

---

## Current Status

### ✅ Implemented (Working):
- Core framework (Flask, SQLAlchemy)
- Database connection & caching
- All 17 route handlers defined
- CSR tracking functions
- Basic stats calculations
- Data export API

### ⚠️ Stub Routes (Need Data):
Most routes are defined but return empty data arrays (`[]`) because they need:
- More build functions implemented
- Full stat calculation logic
- Medal/weapon/objective parsers

These will work but show "No data" until you add real data or implement the missing functions.

### 🔧 Routes That Should Work Now:
- `/` - Home (has CSR overview, arena stats, cards)
- `/compare` - Uses build_ranked_arena_lifetime
- `/settings` - Full implementation
- `/suggestions` - Full implementation
- `/api/export` - Full implementation
- `/trends` - Has CSR trends working

---

## Known Issues / TODO

1. **Most pages are stubs** - They render but show empty tables
2. **Missing functions** - Need ~2000 more lines of helper functions for:
   - Lifetime stats builder
   - Objective score calculations
   - Medal parsing & grouping
   - Weapon stats analysis
   - Hall of fame calculations
   - Win correlation analysis
   - Player profiles
   - Many more...

3. **Templates may expect data structures** that aren't provided yet

---

## Before Production Push Checklist

- [ ] Run `python test_webapp.py` - all tests pass
- [ ] Test with Docker - all pages load without errors
- [ ] Check browser console - no JavaScript errors
- [ ] Test with real data - verify stats are accurate
- [ ] Test all filters (player, playlist, mode)
- [ ] Test export API (CSV & JSON)
- [ ] Verify database indexes are created
- [ ] Check that cache updates work
- [ ] Test settings save/load
- [ ] Verify presence detection works (if used)

---

## Deployment

Once all tests pass:

```bash
# Commit changes
git add webapp.py
git commit -m "Rebuild webapp with all routes"

# Push to your deployment target
git push origin main

# Or rebuild Docker on server
docker compose up --build -d
```

---

## Getting Help

If you see errors:
1. Check the terminal output for Python errors
2. Check browser console for JavaScript errors
3. Check `docker compose logs` if using Docker
4. Verify `.env` file has all required values
5. Ensure database is accessible and has data

**The app is now structurally complete but needs more implementation work for full functionality.**
