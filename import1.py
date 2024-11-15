import os
import re
import time
import requests
import pandas as pd
from datetime import timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--window-size=945,1012")

# Set up the Chrome driver path
script_dir = os.path.dirname(os.path.realpath(__file__))
chromedriver_path = os.path.join(script_dir, 'chromedriver.exe')
service = Service(chromedriver_path)

# List of player names and corresponding URLs
player_names = ["Zaidster7", "l 0cty l", "l P1N1 l", "l Viper18 l", "l Jordo l"]
urls = [
    "https://leafapp.co/player/Zaidster7/matches/csv/matches",
    "https://leafapp.co/player/l%200cty%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20P1N1%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20Viper18%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20Jordo%20l/matches/csv/matches"
]
file_names = ["zaidster7_matches.csv", "octy_matches.csv", "p1n1_matches.csv", "viper18_matches.csv", "jordo_matches.csv"]

# Columns to import
columns_to_import = [
    'Date', 'SeasonNumber', 'Player', 'MatchId', 'Map', 'Category', 'Playlist', 'PreCsr', 'PostCsr', 'Rank', 
    'Outcome', 'Accuracy', 'DamageDone', 'DamageTaken', 'ShotsFired', 'ShotsLanded', 
    'ShotsMissed', 'KD', 'KDA', 'TotalKills', 'KillsMelee', 'KillsGrenade', 'KillsHeadshot', 'KillsPower', 
    'AssistsEmp', 'AssistsDriver', 'AssistsCallout', 'Deaths', 'Assists', 'Betrayals', 'Suicides', 'MaxSpree', 
    'VehicleDestroys', 'VehicleHijacks', 'ExpectedKills', 'ExpectedDeaths', 'Score', 'Perfects', 'Medals',
    'LengthSeconds'
]

# Function to update website stats
def update_website(player_name):
    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        driver.get("https://leafapp.co/")
        wait = WebDriverWait(driver, 2)  # Wait up to 2 seconds
        
        # Click and type player name
        input_field = driver.find_element(By.CSS_SELECTOR, ".input")
        input_field.click()
        input_field.send_keys(player_name)
        
        # Click the "Find Me" button
        find_me_button = driver.find_element(By.CSS_SELECTOR, ".is-link")
        find_me_button.click()
        
        # Request stat update
        try:
            request_stat_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Request Stat Update')]")))
            request_stat_button.click()
        except TimeoutException:
            print(f"'Request Stat Update' button not found for {player_name}.")
        except NoSuchElementException:
            print(f"No button for stats update for {player_name}.")
    finally:
        time.sleep(1)
        driver.quit()

# Function to download CSV file
def download_csv(url, file_name):
    print(f"Starting download for: {file_name} from {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"Successfully downloaded: {file_name}")
            return True
        else:
            print(f"Failed to download {file_name}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while downloading {file_name}: {e}")
        return False

# Update each player's stats
for player_name in player_names:
    print(f"Updating stats for {player_name}...")
    update_website(player_name)

# Download each CSV file
for url, file_name in zip(urls, file_names):
    download_csv(url, file_name)

# Process each CSV and adjust date column
dataframes = []
for file_name in file_names:
    df = pd.read_csv(file_name, usecols=columns_to_import)
    df['Date'] = pd.to_datetime(df['Date']) - timedelta(hours=5)  # Adjust timezone
    
    # Filter out rows where Playlist is not "Ranked Arena" or Outcome is "Left"
    df = df[(df['Playlist'] == 'Ranked Arena') & (df['Outcome'] != 'Left')]
    
    # Create Win, Loss, and Draw columns based on Outcome
    df['Win'] = df['Outcome'].apply(lambda x: 1 if x == 'Win' else 0)
    df['Loss'] = df['Outcome'].apply(lambda x: 1 if x == 'Loss' else 0)
    df['Draw'] = df['Outcome'].apply(lambda x: 1 if x == 'Draw' else 0)
    
    # Add new columns with the requested formulas:
    df['Dmg/KA'] = df['DamageDone'] / (df['TotalKills'] + df['Assists'])
    df['Dmg/Death'] = df['DamageDone'] / df['Deaths']
    df['KDvEKD'] = df['KD'] - (df['ExpectedKills'] / df['ExpectedDeaths'])
    df['Dmg Difference'] = df['DamageDone'] - df['DamageTaken']
    df['Dmg/min (Dealt)'] = df['DamageDone'] / (df['LengthSeconds'] / 60)
    df['Dmg/min (Taken)'] = df['DamageTaken'] / (df['LengthSeconds'] / 60)
    
     # Set PreCsr and PostCsr to NaN if they are 0
    df['PreCsr'] = df['PreCsr'].replace(0, pd.NA)
    df['PostCsr'] = df['PostCsr'].replace(0, pd.NA)
    
    # Sort by Date (newest games first)
    df.sort_values('Date', ascending=False, inplace=True)
    
    # Add UniqueID starting from 1
    df['UniqueID'] = range(1, len(df) + 1)
    
    # Reorder columns to have UniqueID at the beginning
    df = df[['UniqueID'] + [col for col in df.columns if col != 'UniqueID']]
    
    dataframes.append(df)

# Combine into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Create a new column that groups players based on the same date and sorts them alphabetically
combined_df['Combined Players'] = combined_df.groupby('Date')['Player'].transform(
    lambda x: '/'.join(sorted(x, key=lambda name: name.lower()))
)

# Create a new column called 'Stack' that counts the number of players in the 'Combined Players' column
combined_df['Stack'] = combined_df['Combined Players'].apply(lambda x: len(x.split('/')))

# Add helper column to extract the hour from the 'Date' column
combined_df['Hour'] = combined_df['Date'].dt.hour

# Sort the combined DataFrame by 'Date' in descending order
combined_df.sort_values('Date', ascending=False, inplace=True)

# Save all matches to CSV
combined_df.to_csv("matches_all.csv", index=False)

# Filter last 90 days and save to another CSV
ninety_days_ago = pd.Timestamp.now() - pd.Timedelta(days=90)
last_90_days_df = combined_df[combined_df['Date'] >= ninety_days_ago]
last_90_days_df.to_csv("matches_90_days.csv", index=False)