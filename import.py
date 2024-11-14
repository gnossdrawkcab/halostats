import requests
import pandas as pd
from datetime import timedelta
import numpy as np

# URLs of the CSV files
urls = [
    "https://leafapp.co/player/Zaidster7/matches/csv/matches",
    "https://leafapp.co/player/l%200cty%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20P1N1%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20Viper18%20l/matches/csv/matches",
    "https://leafapp.co/player/l%20Jordo%20l/matches/csv/matches"
]

# File names to save the CSVs locally
file_names = ["zaidster7_matches.csv", "octy_matches.csv", "p1n1_matches.csv", "viper18_matches.csv", "jordo_matches.csv"]

# Columns to import (using names)

columns_to_import = [
    'Date', 'SeasonNumber', 'Player', 'MatchId', 'Map', 'Category', 'Playlist', 'PreCsr', 'PostCsr', 'Rank', 
    'Outcome', 'Accuracy', 'DamageDone', 'DamageTaken', 'ShotsFired', 'ShotsLanded', 
    'ShotsMissed', 'KD', 'KDA', 'TotalKills', 'KillsMelee', 'KillsGrenade', 'KillsHeadshot', 'KillsPower', 
    'AssistsEmp', 'AssistsDriver', 'AssistsCallout', 'Deaths', 'Assists', 'Betrayals', 'Suicides', 'MaxSpree', 
    'VehicleDestroys', 'VehicleHijacks', 'ExpectedKills', 'ExpectedDeaths', 'Score', 'Perfects', 'Medals',
    'LengthSeconds'
]

# Download and save CSV files
for url, file_name in zip(urls, file_names):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)

# List to store DataFrames
dataframes = []

# Load CSVs into DataFrames and adjust time in 'Date' column
for file_name in file_names:
    df = pd.read_csv(file_name, usecols=columns_to_import)  # Select only specific columns
    
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Subtract 4 hours from the datetime in the 'Date' column
    df['Date'] = df['Date'] - timedelta(hours=4)

    # Create columns for Win, Loss, and Draw based on 'Outcome'
    df['Win'] = df['Outcome'].apply(lambda x: 1 if x == 'Win' else 0)
    df['Loss'] = df['Outcome'].apply(lambda x: 1 if x == 'Loss' else 0)
    df['Draw'] = df['Outcome'].apply(lambda x: 1 if x == 'Draw' else 0)

    dataframes.append(df)

# Combine all CSV files into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Create a dictionary for renaming columns
rename_dict = {
    'SeasonNumber': 'Season',
    'Category': 'Gametype',
    'Playlist': 'Playlist',
    'DamageDone': 'Dmg Done',
    'DamageTaken': 'Dmg Taken',
    'ShotsFired': 'Shots Fired',
    'ShotsLanded': 'Shots Landed',
    'ShotsMissed': 'Shots Missed',
    'TotalKills': 'Kills',
    'KillsMelee': 'Melee Kills',
    'KillsGrenade': 'Gren. Kills',
    'KillsHeadshot': 'HS Kills',
    'KillsPower': 'PW Kills',
    'AssistsCallout': 'Callout Assists',
    'MaxSpree': 'Max Spree',
    'ExpectedKills': 'Exp Kills',
    'ExpectedDeaths': 'Exp Deaths',
    'LengthSeconds': 'Match Length (s)',
    'Match Length (min)': 'Match Length (m)',
}

# Rename columns in the combined DataFrame
combined_df.rename(columns=rename_dict, inplace=True)# Sort the data by the 'Date' column in descending order
combined_df.sort_values(by='Date', ascending=False, inplace=True)

# Remove rows where Playlist is NOT 'Ranked Arena'
combined_df = combined_df[combined_df['Playlist'] == 'Ranked Arena']

# Remove rows with 0 in 'TotalKills' or 'Deaths'
combined_df = combined_df[(combined_df['Kills'] != 0) & (combined_df['KDA'] != 0)]

# Remove rows where 'Outcome' has the value 'Left'
combined_df = combined_df[combined_df['Outcome'] != 'Left']

# Columns to check for zeros
#columns_to_check_csr = ['PreCsr', 'PostCsr']

# Replace 0s with NaN in the specified columns
#combined_df[columns_to_check_csr] = combined_df[columns_to_check_csr].replace(0, np.nan)

# Calculate the KA/D column
combined_df['KA/D'] = (combined_df['Kills'] + combined_df['Assists']) / combined_df['Deaths']

# Add 'Dmg/KA' and 'Dmg/Death' columns
combined_df['Dmg/KA'] = combined_df['Dmg Done'] / (combined_df['Kills'] + combined_df['Assists'])
combined_df['Dmg/Death'] = combined_df['Dmg Done'] / combined_df['Deaths']

# Create the 'EKD' column (DamageDone / KDA), handling division by zero
combined_df['EKD'] = combined_df['Exp Kills'] / combined_df['Exp Deaths'].replace(0, pd.NA)

# Convert 'EKD' to numeric and fill NaN values with 0
combined_df['EKD'] = pd.to_numeric(combined_df['EKD'], errors='coerce').fillna(0)

# Create a new column called 'KD vs. EKD'
combined_df['KDvEKD'] = combined_df['KD'] - combined_df['EKD']

# Add UniqueID column starting from 1
combined_df.insert(0, 'UniqueID', range(1, len(combined_df) + 1))

# Remove " - Ranked" from any rows in 'Map'
combined_df['Map'] = combined_df['Map'].str.replace(" - Ranked", "", regex=False)

# Add Dmg Difference column (DamageDone - DamageTaken)
combined_df['Dmg Difference'] = combined_df['Dmg Done'] - combined_df['Dmg Taken']

# Function to get oldest values for 'PreCsr' and newest values for 'PostCsr' for a specified number of days
def get_oldest_g_values(threshold_days):
    today = pd.Timestamp.now()
    threshold_date = today - pd.Timedelta(days=threshold_days)
    
    # Filter for rows in the last threshold_days
    filtered_df = combined_df[combined_df['Date'] >= threshold_date]
    
    # Get the oldest 'PreCsr' value for each unique player
    oldest_g_values = filtered_df.loc[filtered_df.groupby('Player')['Date'].idxmin()]
    oldest_g_values = oldest_g_values[['Player', 'PreCsr']]
    oldest_g_values.columns = ['Unique_Player', f'PreCsr{threshold_days}']
    
    return oldest_g_values

def get_newest_postcsr_values(threshold_days):
    today = pd.Timestamp.now()
    threshold_date = today - pd.Timedelta(days=threshold_days)
    
    # Filter for rows in the last threshold_days
    filtered_df = combined_df[combined_df['Date'] >= threshold_date]
    
    # Get the newest 'PostCsr' value for each unique player
    newest_postcsr_values = filtered_df.loc[filtered_df.groupby('Player')['Date'].idxmax()]
    newest_postcsr_values = newest_postcsr_values[['Player', 'PostCsr']]
    newest_postcsr_values.columns = ['Unique_Player', f'PostCsr{threshold_days}']
    
    return newest_postcsr_values

# Get oldest 'PreCsr' and newest 'PostCsr' values for specific days
oldest_g_today_minus_0 = get_oldest_g_values(0.5)
oldest_g_today_minus_1 = get_oldest_g_values(1)
oldest_g_today_minus_8 = get_oldest_g_values(8)
oldest_g_today_minus_16 = get_oldest_g_values(16)
oldest_g_today_minus_31 = get_oldest_g_values(31)
newest_postcsr_today_minus_31 = get_newest_postcsr_values(31)

# Merge the oldest and newest values into the combined DataFrame
combined_df = combined_df.merge(oldest_g_today_minus_0, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])
combined_df = combined_df.merge(oldest_g_today_minus_1, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])
combined_df = combined_df.merge(oldest_g_today_minus_8, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])
combined_df = combined_df.merge(oldest_g_today_minus_16, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])
combined_df = combined_df.merge(oldest_g_today_minus_31, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])
combined_df = combined_df.merge(newest_postcsr_today_minus_31, left_on='Player', right_on='Unique_Player', how='left').drop(columns=['Unique_Player'])

# Add column to detect games played Yesterday or before 5am today
today = pd.Timestamp.now()
start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)
start_of_yesterday = start_of_today - pd.Timedelta(days=1)
cutoff_time_today = start_of_today + pd.Timedelta(hours=5)

# Games played Yesterday but after 5am up to 5am today
combined_df['Game Yesterday or Before 5am Today'] = (
    (combined_df['Date'] >= start_of_yesterday + pd.Timedelta(hours=5)) &  
    (combined_df['Date'] < cutoff_time_today)
)

# Games played Today after 5am
combined_df['Game Today After 5am'] = (
    (combined_df['Date'] >= cutoff_time_today) &  
    (combined_df['Date'] <= today)
)

# Create a new column that groups players based on the same date and sorts them alphabetically
combined_df['Combined Players'] = combined_df.groupby('Date')['Player'].transform(lambda x: '/'.join(sorted(x, key=lambda name: name.lower())))

# Create a new column called 'Stack' that counts the number of players in the 'Combined Players' column
combined_df['Stack'] = combined_df['Combined Players'].apply(lambda x: len(x.split('/')))

# Add helper column to extract the hour from the 'Date' column
combined_df['Hour'] = combined_df['Date'].dt.hour

# Move 'PreCsr', 'PostCsr', and 'Outcome' columns to the end
cols_to_move = ['PreCsr', 'PostCsr', 'Outcome']
combined_df = combined_df[[col for col in combined_df.columns if col not in cols_to_move] + cols_to_move]

# Create the 'Match Length (min)' and 'Dmg/min (Dealt)' columns
combined_df['Match Length (min)'] = combined_df['Match Length (s)'] / 60
combined_df['Dmg/min (Dealt)'] = combined_df['Dmg Done'] / combined_df['Match Length (min)']

# Add 'Dmg/min (Taken)' column
combined_df['Dmg/min (Taken)'] = combined_df['Dmg Taken'] / combined_df['Match Length (min)']

# Save the final DataFrame into a new CSV file
combined_df.to_csv("matches_all.csv", index=False)

# Save the filtered DataFrame for the last 90 days into another CSV file
ninety_days_ago = pd.Timestamp.now() - pd.Timedelta(days=90)
last_90_days_df = combined_df[combined_df['Date'] >= ninety_days_ago]
last_90_days_df.to_csv("matches_90_days.csv", index=False)
