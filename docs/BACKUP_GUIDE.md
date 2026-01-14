# Backup & Version Control Guide

This guide explains how to protect your files from accidental changes.

---

## Option 1: Quick Backup Script (Easiest)

Run anytime to create a timestamped backup:

```bash
powershell -ExecutionPolicy Bypass -File backup.ps1
```

**What it does:**
- Creates a timestamped folder in `./backups/`
- Backs up all critical Python files
- Keeps only the last 10 backups (auto-cleanup)
- Fast and simple

**Restore from backup:**
```bash
# Copy files from backup folder back to main directory
Copy-Item "backups\backup_2026-01-14_153022\*" "." -Recurse -Force
```

**Backups are stored in:**
```
backups/
  ├── backup_2026-01-14_150000/
  │   ├── webapp.py
  │   ├── auth.py
  │   └── ...
  ├── backup_2026-01-14_153022/
  └── ...
```

---

## Option 2: Git Version Control (Best Practice)

Git tracks every change with full history. Recommended!

### Initialize Git (One-time setup)

```bash
cd \\pathtpc\appdata\halo
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git add .
git commit -m "Initial commit - full webapp implementation"
```

### After Each Change Session

```bash
git add webapp.py  # or other modified files
git commit -m "Add leaderboard functions"
```

### View History

```bash
git log --oneline
```

Output shows every change:
```
a1b2c3d Add leaderboard functions
d4e5f6g Update trends routes
h7i8j9k Initial commit
```

### Revert a Single File

```bash
git checkout HEAD -- webapp.py
```

### Revert to Specific Commit

```bash
git reset --hard a1b2c3d
```

### Create a Backup Branch Before Big Changes

```bash
# Save current state as branch
git branch backup-before-major-refactor

# Later: switch back if needed
git checkout backup-before-major-refactor
```

---

## Option 3: Combine Both (Maximum Safety)

Best approach for critical work:

```bash
# Before major changes:
powershell -ExecutionPolicy Bypass -File backup.ps1  # Quick backup
git add .
git commit -m "Before implementing [feature]"

# Make changes...

# After testing:
git add .
git commit -m "Implement [feature] - tested and working"
```

---

## Schedule Automatic Backups (Windows Task Scheduler)

Create a scheduled task to backup automatically:

### PowerShell Script to Create Task

```powershell
$scriptPath = "C:\path\to\backup.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At "10:00 AM"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File $scriptPath"
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "Halo Backup" -Trigger $trigger -Action $action -Principal $principal
```

Or set up daily backups manually:
1. Open Task Scheduler
2. Create Basic Task → "Halo Backup"
3. Set to run daily at preferred time
4. Action: Run `powershell -ExecutionPolicy Bypass -File C:\path\to\backup.ps1`

---

## Disaster Recovery Examples

### Scenario: AI accidentally deleted code

**With backup script:**
```bash
# List available backups
Get-ChildItem backups

# Restore from most recent
Copy-Item "backups\backup_2026-01-14_153022\*" "." -Recurse -Force
```

**With git:**
```bash
# See what changed
git diff

# Undo last commit
git reset --soft HEAD~1

# Or revert to known good state
git checkout a1b2c3d -- webapp.py
```

### Scenario: Want to try something risky

**With git:**
```bash
# Create experimental branch
git checkout -b experimental-feature

# Make risky changes...
# Test...

# If it works, merge back
git checkout main
git merge experimental-feature

# If it doesn't work, just delete branch
git branch -D experimental-feature
```

---

## File Protection

All methods back up these critical files:
- `webapp.py` - Main web application
- `auth.py` - Authentication logic
- `stats.py` - Stats processing
- `entrypoint.py` - Scraper entry point
- `requirements.txt` - Dependencies
- `compose.yaml` - Docker configuration
- `.env` - Environment variables
- `Dockerfile` - Container definition

---

## Recommendations

### For Development:
- ✅ Use **git** - full history, easy branching
- ✅ Run `backup.ps1` before big changes
- ✅ Commit frequently with clear messages

### For Production:
- ✅ Use both **git** and **backup.ps1**
- ✅ Schedule automatic daily backups
- ✅ Keep backups off the main server too

### Before Major Changes:
```bash
# Always do this:
powershell -ExecutionPolicy Bypass -File backup.ps1  # Create backup
git add .                                             # Stage changes
git commit -m "Checkpoint before major change"       # Save state

# Make changes...
# Test thoroughly...

git add .
git commit -m "Major change complete and tested"     # Save final state
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Quick backup now | `backup.ps1` |
| See git history | `git log --oneline` |
| Revert one file | `git checkout HEAD -- webapp.py` |
| See what changed | `git diff` |
| Undo last commit | `git reset --soft HEAD~1` |
| Create safe branch | `git checkout -b my-feature` |
| Restore from backup | `Copy-Item backups\backup_XXX\* . -Recurse` |

---

## That's It!

You're now protected! The next time someone (or something) makes a change, you can:
1. See exactly what changed
2. Revert it quickly
3. Compare versions
4. Keep a full history

Happy coding! 🎮
