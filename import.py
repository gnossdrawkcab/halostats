import os

import webbrowser
import glob
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import requests
import pandas as pd
from datetime import timedelta

def run_test(test_name, input_text):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--window-size=945,1012")

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    chromedriver_path = os.path.join(script_dir, 'chromedriver.exe')

    # Set up the Chrome driver with the correct path
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Step 1: Open the URL
        driver.get("https://leafapp.co/")
        
        # Create a wait instance
        wait = WebDriverWait(driver, 2)  # Wait up to 2 seconds
    
        # Step 2: Click on the input field
        input_field = driver.find_element(By.CSS_SELECTOR, ".input")
        input_field.click()
        
        # Step 3: Type the input text
        input_field.send_keys(input_text)
        
        # Step 4: Click on the "Find Me" button
        find_me_button = driver.find_element(By.CSS_SELECTOR, ".is-link")
        find_me_button.click()
        
        try:
            # Wait for the button to be clickable
            request_stat_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Request Stat Update')]")))
            request_stat_button.click()
        except TimeoutException:
            print("The 'Request Stat Update' button was not found. Continuing with the program.")
        except NoSuchElementException:
            print("The button does not exist on the page.")
        except Exception as e:
            print(f"An error occurred: {e}")

    finally:
        # Close the browser
        time.sleep(1)  # Adjust this sleep time as needed
        driver.quit()

# Function to collect player names
def get_player_names(existing_names=None):
    player_names = []
    
    if existing_names:
        # Ask if the user wants to use existing player names
        use_existing = input(f"Found existing player names: {', '.join(existing_names)}. Would you like to use these? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            return existing_names

    print("Enter new player names (type 'done' when finished):")
    
    while True:
        name = input("Enter player name: ").strip()
        if name.lower() == 'done':
            break
        if name:
            player_names.append(name)
        else:
            print("Name cannot be empty. Please enter a valid name.")

    return player_names

# Function to find existing CSV files
def find_existing_csvs():
    current_directory = os.getcwd()
    csv_pattern = re.compile(r'^(.*)_matches\.csv$')
    existing_files = [f[:-11].rstrip('_') for f in os.listdir(current_directory) if csv_pattern.match(f)]
    return existing_files

# Function to download the CSV file
def download_csv(url, file_name):
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

# Check for existing CSV files
existing_player_names = find_existing_csvs()

# Collect player names
player_names = get_player_names(existing_player_names)

# Prepare the test input list based on the player names
tests = [(name, f"{name}") for name in player_names]

# Run all tests
for test_name, input_value in tests:
    print(f"Running test: {test_name}")
    run_test(test_name, input_value)

# URLs of the CSV files
urls = [f"https://leafapp.co/player/{name}/matches/csv/matches" for name in player_names]

# File names to save the CSVs locally
file_names = [f"{name.lower()}_matches.csv" for name in player_names]

# Flag to check if any download failed
download_failed = False

# Download the CSV files
for url, file_name in zip(urls, file_names):
    if not download_csv(url, file_name):
        download_failed = True  # Set the flag if a download fails

# If the user did not want to use existing files and all downloads failed
if player_names != existing_player_names and download_failed:
    user_response = input("All download attempts failed. Would you like to continue? (y/n): ").strip().lower()
    
    if user_response != 'y':
        print("Exiting the program.")
        exit(1)  # Exit

# If some downloads failed, notify the user
if download_failed:
    print("Some downloads failed. Please wait a while to run this script again.")

# Proceed to read the files if they exist
dataframes = []  # List to store DataFrames
columns_to_import = [0, 1, 3, 5, 6, 10, 11, 24, 29] + list(range(30, 59))  # Columns to import (using 0-based indexing)

# Columns to import (using 0-based indexing)
columns_to_import = [0, 1, 3, 5, 6, 10, 11, 24, 29] + list(range(30, 59))

# Proceed to read the files if they exist
for file_name in file_names:
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name, usecols=columns_to_import)  # Specify your columns_to_import
            # (Additional processing on df can go here)
        except FileNotFoundError as e:
            print(f"Error: {e}. File may not have been downloaded successfully.")
    else:
        print(f"{file_name} does not exist, skipping reading.")

# List to store DataFrames
dataframes = []

# Load CSVs into DataFrames and adjust time in Column A
for file_name in file_names:
    df = pd.read_csv(file_name, usecols=columns_to_import)  # Select only specific columns
    
    # Ensure column A (the first column in the selection) is in datetime format
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

    # Subtract 4 hours from the datetime in the first column
    df[df.columns[0]] = df[df.columns[0]] - timedelta(hours=4)

    # Create columns for Win, Loss, and Draw based on column I (Outcome)
    df['Win'] = df[df.columns[8]].apply(lambda x: 1 if x == 'Win' else 0)  # Column I for Outcome
    df['Loss'] = df[df.columns[8]].apply(lambda x: 1 if x == 'Loss' else 0)
    df['Draw'] = df[df.columns[8]].apply(lambda x: 1 if x == 'Draw' else 0)

    dataframes.append(df)

# Combine all CSV files into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Sort the data by the first column (date) in descending order
combined_df.sort_values(by=combined_df.columns[0], ascending=False, inplace=True)

# Remove rows with 0 in columns G or H (index 6 or 7)
combined_df = combined_df[(combined_df.iloc[:, 6] != 0) & (combined_df.iloc[:, 7] != 0)]

# Remove rows where column I (index 8) has the value 'Left'
combined_df = combined_df[combined_df.iloc[:, 8] != 'Left']

# Add Dmg/KA and Dmg/Death columns
combined_df['Dmg/KA'] = combined_df['DamageDone'] / (combined_df['TotalKills'] + combined_df['Assists'])
combined_df['Dmg/Death'] = combined_df['DamageDone'] / combined_df['Deaths']

# Create the EKD column (column 32 divided by column 33), handling division by zero
combined_df['EKD'] = combined_df.iloc[:, 32] / combined_df.iloc[:, 33].replace(0, pd.NA)

# Convert EKD to numeric and fill NaN values with 0
combined_df['EKD'] = pd.to_numeric(combined_df['EKD'], errors='coerce').fillna(0)

# Create a new column called "KD vs. EKD"
combined_df['KD vs. EKD'] = combined_df.iloc[:, 15] - combined_df.iloc[:, 40]

# Add UniqueID column starting from 1
combined_df.insert(0, 'UniqueID', range(1, len(combined_df) + 1))

# Remove " - Ranked" from any rows in column E
combined_df.iloc[:, 4] = combined_df.iloc[:, 4].str.replace(" - Ranked", "", regex=False)

# Create the first new column (L - M)
combined_df['Dmg Difference'] = combined_df.iloc[:, 11] - combined_df.iloc[:, 12]  # Subtract column M from L

# Define function to get oldest values for G and newest values for H
def get_oldest_g_values(threshold_days):
    today = pd.Timestamp.now()
    threshold_date = today - pd.Timedelta(days=threshold_days)
    
    # Filter for rows in the last threshold_days
    filtered_df = combined_df[combined_df.iloc[:, 1] >= threshold_date]
    
    # Get the oldest G value for each unique player (Column D)
    oldest_g_values = filtered_df.loc[filtered_df.groupby(filtered_df.columns[3])[filtered_df.columns[1]].idxmin()]
    
    # Select relevant columns (D and G) for the result
    oldest_g_values = oldest_g_values[[filtered_df.columns[3], filtered_df.columns[6]]]  # Keep only D and G columns
    oldest_g_values.columns = ['Unique_D', f'PRECsr{threshold_days})']
    
    return oldest_g_values

def get_newest_postcsr_values(threshold_days):
    today = pd.Timestamp.now()
    threshold_date = today - pd.Timedelta(days=threshold_days)
    
    # Filter for rows in the last threshold_days
    filtered_df = combined_df[combined_df.iloc[:, 1] >= threshold_date]
    
    # Get the newest PostCsr value for each unique player (Column D)
    newest_postcsr_values = filtered_df.loc[filtered_df.groupby(filtered_df.columns[3])[filtered_df.columns[1]].idxmax()]
    
    # Select relevant columns (D and H) for the result
    newest_postcsr_values = newest_postcsr_values[[filtered_df.columns[3], filtered_df.columns[7]]]  # Keep only D and H columns
    newest_postcsr_values.columns = ['Unique_D', f'POSTCsr{threshold_days})']
    
    return newest_postcsr_values

# Get oldest G values and newest PostCsr values for Today()-1, Today()-8, Today()-16, and Today()-31
oldest_g_today_minus_1 = get_oldest_g_values(1)
oldest_g_today_minus_8 = get_oldest_g_values(8)
oldest_g_today_minus_16 = get_oldest_g_values(16)
oldest_g_today_minus_31 = get_oldest_g_values(31)

newest_postcsr_today_minus_31 = get_newest_postcsr_values(31)

# Merge the oldest G and newest PostCsr values into the combined DataFrame
combined_df = combined_df.merge(oldest_g_today_minus_1, left_on=combined_df.columns[3], right_on='Unique_D', how='left').drop(columns=['Unique_D'])
combined_df = combined_df.merge(oldest_g_today_minus_8, left_on=combined_df.columns[3], right_on='Unique_D', how='left').drop(columns=['Unique_D'])
combined_df = combined_df.merge(oldest_g_today_minus_16, left_on=combined_df.columns[3], right_on='Unique_D', how='left').drop(columns=['Unique_D'])
combined_df = combined_df.merge(oldest_g_today_minus_31, left_on=combined_df.columns[3], right_on='Unique_D', how='left').drop(columns=['Unique_D'])

combined_df = combined_df.merge(newest_postcsr_today_minus_31, left_on=combined_df.columns[3], right_on='Unique_D', how='left').drop(columns=['Unique_D'])

# Add column to detect games played Yesterday or before 5am today
today = pd.Timestamp.now()
start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)
start_of_yesterday = start_of_today - pd.Timedelta(days=1)
cutoff_time_today = start_of_today + pd.Timedelta(hours=5)

# Only flag games played Yesterday or before 5am today
combined_df['Game Yesterday or Before 5am Today'] = (
    (combined_df.iloc[:, 1] >= start_of_yesterday) &  # Games starting from Yesterday
    (combined_df.iloc[:, 1] < cutoff_time_today)  # Games before 5am today
)

# Now, only games within this timeframe will be marked as TRUE.

# Add column to detect games played today but after 5am
combined_df['Game Today After 5am'] = (
    (combined_df.iloc[:, 1] >= cutoff_time_today) &  # Games after 5am today
    (combined_df.iloc[:, 1] <= today)  # Games up to the current time
)

# Step: Create a new column that groups players based on the same date in column B and sorts them alphabetically (case insensitive)
# Group by the date in column B and concatenate the player names from column D in alphabetical order (case insensitive but retains original case)
combined_df['Combined Players'] = combined_df.groupby(combined_df.columns[1])[combined_df.columns[3]].transform(lambda x: '/'.join(sorted(x, key=lambda name: name.lower())))

# Step: Create a new column called 'Stack' that counts the number of players in the 'Combined Players' column
combined_df['Stack'] = combined_df['Combined Players'].apply(lambda x: len(x.split('/')))

# Add helper column to extract the hour from the datetime (first column in the dataset)
combined_df['Hour'] = combined_df.iloc[:, 1].dt.hour  # Add 1 to make it range from 1 to 24

# Move columns AK (index 36) through AM (index 38) to the end
cols_to_move = combined_df.columns[36:39]  # Selecting columns AK through AM by their index
combined_df = combined_df[[col for col in combined_df.columns if col not in cols_to_move] + list(cols_to_move)]

# Assume BE is at index 56 (adjust based on your data)
be_index = 56  # Adjust based on the correct index for column BE

# Create the new column 'BE/60' by dividing column BE by 60, and add it at the end
combined_df['Match Length (min)'] = combined_df.iloc[:, be_index] / 60

# Assume BF is at index 57 (adjust based on your data)
bf_index = 57  # Adjust based on the correct index for column BE

# Add 'Dmg/s (Dealt)' column (L divided by BF) at the end of the DataFrame
combined_df['Dmg/min (Dealt)'] = combined_df.iloc[:, 11] / combined_df.iloc[:, bf_index]

# Add 'Dmg/s (Taken)' column (M divided by BF) at the end of the DataFrame
combined_df['Dmg/min (Taken)'] = combined_df.iloc[:, 12] / combined_df.iloc[:, bf_index]

# Move column AR (index 44) to between AO (index 42) and AP (index 43)
combined_df.insert(41, combined_df.columns[43], combined_df.pop(combined_df.columns[43]))  # Move AR to 43

# Move column BG (index 57) to between AP (index 43) and AQ (index 44)
combined_df.insert(42, combined_df.columns[58], combined_df.pop(combined_df.columns[58]))  # Move BG to 44
# Move column BH (index 58) to between AQ (index 44) and AR (now at index 43)
combined_df.insert(43, combined_df.columns[59], combined_df.pop(combined_df.columns[59]))  # Move BH to 45

# Save the final DataFrame into a new CSV file
combined_df.to_csv("matches_all.csv", index=False)

# Save the filtered DataFrame for the last 90 days based on Column B
ninety_days_ago = pd.Timestamp.now() - pd.Timedelta(days=90)
filtered_df_90_days = combined_df[combined_df.iloc[:, 1] >= ninety_days_ago]

# Save the filtered DataFrame into a new CSV file
filtered_df_90_days.to_csv("matches_90_days.csv", index=False)

# Load the matches_90_days.csv file
df = pd.read_csv("matches_90_days.csv")

# Create the pivot table with multiple aggregations
pivot_table = pd.pivot_table(
    df,
    index='Player',  # Assuming "Player" is the column name for player names
    aggfunc={
        'Date': 'count',  # Count of matches
        'TotalKills': 'mean',   # Average kills
        'Deaths': 'mean',  # Average deaths
        'Assists': 'mean', # Average assists
        'KD': 'mean',      # Average KD (if you have this column)
        'KDA': 'mean'      # Average KDA
    },
    fill_value=0  # Fill missing values with 0
)

# Rename the columns for clarity
pivot_table.rename(columns={
    'Date': 'Matches',
    'TotalKills': 'Average Kills',
    'Deaths': 'Average Deaths',
    'Assists': 'Average Assists',
    'KD': 'Average KD',
    'KDA': 'Average KDA'
}, inplace=True)

# Sort the pivot table by KDA in descending order
pivot_table.sort_values(by='Average KDA', ascending=False, inplace=True)

# Save the pivot table to an HTML file
html_file_path = "kda_pivot_table.html"
pivot_table.to_html(html_file_path)

# Open the HTML file in the default web browser
webbrowser.open('file://' + os.path.realpath(html_file_path))

