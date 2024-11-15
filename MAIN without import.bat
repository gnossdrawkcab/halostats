@echo on
pushd \\PATHTPC\appdata\nginx\www
REM py import1.py
py pivot.py
copy "jordo_matches.csv" "C:\Users\pat\Documents\HaloCode\jordo_matches.csv"
copy "octy_matches.csv" "C:\Users\pat\Documents\HaloCode\octy_matches.csv"
copy "viper18_matches.csv" "C:\Users\pat\Documents\HaloCode\viper18_matches.csv"
copy "zaidster7_matches.csv" "C:\Users\pat\Documents\HaloCode\zaidster7_matches.csv"
copy "p1n1_matches.csv" "C:\Users\pat\Documents\HaloCode\p1n1_matches.csv"
popd \\PATHTPC\appdata\nginx\www
pause