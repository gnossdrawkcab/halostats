# main.py

import subprocess

def run_import():
    # Replace 'script1.py' with the name of your first script
    subprocess.run(['python', 'import.py'])

def run_pivot():
    # Replace 'script2.py' with the name of your second script
    subprocess.run(['python', 'pivot.py'])

if __name__ == "__main__":
    run_import()
    run_pivot()

input()
