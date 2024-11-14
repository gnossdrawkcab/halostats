import pandas as pd
from datetime import datetime, timedelta
import webbrowser
import os
import numpy as np

# Load the matches_90_days.csv file with the specified columns
columns_to_import = [
    'Player', 'Date', 'MatchId', 'Rank', 'Accuracy', 'Dmg Done', 'Dmg Taken',
    'Shots Fired', 'Shots Landed', 'Shots Missed', 'KD', 'KDA', 'Kills',
    'Melee Kills', 'Gren. Kills', 'HS Kills', 'PW Kills', 'AssistsEmp',
    'AssistsDriver', 'Callout Assists', 'Deaths', 'Assists', 'Betrayals',
    'Suicides', 'Max Spree', 'VehicleDestroys', 'VehicleHijacks', 'Exp Kills',
    'Exp Deaths', 'Score', 'Perfects', 'Medals', 'Match Length (s)', 'Win',
    'Loss', 'Draw', 'KA/D', 'Dmg/KA', 'Dmg/Death', 'EKD', 'KDvEKD',
    'Dmg Difference', 'PreCsr0.5', 'PreCsr1', 'PreCsr8', 'PreCsr16',
    'PreCsr31', 'PostCsr31', 'Game Yesterday or Before 5am Today',
    'Game Today After 5am', 'Combined Players', 'Stack', 'Hour', 'PreCsr',
    'PostCsr', 'Outcome', 'Match Length (min)', 'Dmg/min (Dealt)', 'Dmg/min (Taken)'
]

# Load the data
df_90_days = pd.read_csv("matches_90_days.csv", usecols=columns_to_import)

# Ensure 'Date' is in datetime format
df_90_days['Date'] = pd.to_datetime(df_90_days['Date'], errors='coerce')

# Replace all 0 values in numeric columns with NaN to avoid division by zero issues
numeric_columns = [
    'KD', 'KDA', 'Kills', 'Deaths', 'Assists', 'KA/D', 'KDvEKD',
    'Dmg Done', 'Dmg Taken', 'Dmg Difference', 'Dmg/KA', 'Dmg/Death',
    'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Shots Fired', 'Shots Landed',
    'Accuracy', 'Melee Kills', 'Gren. Kills', 'PW Kills', 'HS Kills',
    'AssistsEmp', 'AssistsDriver', 'Score', 'Medals', 'Win'
]

# Replace 0 with NaN to avoid division by zero issues
df_90_days[numeric_columns] = df_90_days[numeric_columns].replace(0, np.nan)

# Convert columns to numeric to handle invalid calculations (set invalid to NaN)
df_90_days[numeric_columns] = df_90_days[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Replace any remaining inf values with NaN
df_90_days.replace([np.inf, -np.inf], np.nan, inplace=True)

# Filter DataFrames by date ranges
now = datetime.now()

time_windows = {
    'Last 30 Days': df_90_days[df_90_days['Date'] >= (now - timedelta(days=30))],
    'Last 15 Days': df_90_days[df_90_days['Date'] >= (now - timedelta(days=15))],
    'Last 7 Days': df_90_days[df_90_days['Date'] >= (now - timedelta(days=7))],
    'Yesterday': df_90_days[
        (df_90_days['Date'] >= (now - timedelta(days=1)).replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < now.replace(hour=5, minute=0, second=0))
    ],
    'Today': df_90_days[df_90_days['Date'] >= now.replace(hour=5, minute=0, second=0)]
}

# Function to format specified columns in the DataFrame to have a certain number of decimal places
def format_decimal_places(df, format_dict):
    for column, decimals in format_dict.items():
        if column in df.columns:
            df[column] = df[column].apply(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else '0')
    return df

# Function to create a styled pivot table sorted by KDA descending
def create_html_pivot_table(df, label):
    if df.empty:
        print(f"No data available for {label}.")
        return None

    # Create the pivot table
    pivot_table = pd.pivot_table(
        df,
        index='Player',
        values=[
            'Kills', 'Deaths', 'Assists', 'KD', 'KDA', 'KA/D', 'KDvEKD',
            'Dmg Done', 'Dmg Taken', 'Dmg Difference', 'Dmg/KA', 'Dmg/Death',
            'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Shots Fired', 'Shots Landed',
            'Accuracy', 'Melee Kills', 'Gren. Kills', 'PW Kills', 'HS Kills',
            'Score', 'Medals', 'Win'
        ],
        aggfunc='mean',
        fill_value=np.nan  # Keep NaN for proper styling and calculations
    )

    # Format the decimal places as required
    format_dict = {
        'KDA': 2,
        'KD': 2,
        'KA/D': 2,
        'Kills': 1,
        'Deaths': 1,
        'Assists': 1,
        'Accuracy': 2,
        'Dmg Done': 0,
        'Dmg Taken': 0,
        'Dmg Difference': 0,
        'Dmg/KA': 0,
        'Dmg/Death': 0,
        'Dmg/min (Dealt)': 0,
        'Dmg/min (Taken)': 0,
        'Shots Fired': 0,
        'Shots Landed': 0,
        'Score': 0,
        'Medals': 0,
        'Win': 2,
        'KDvEKD': 2,
        'Gren. Kills': 1,
        'Melee Kills': 1,
        'PW Kills': 1,
        'HS Kills': 1,
    }

    pivot_table = format_decimal_places(pivot_table, format_dict)

    # Check if 'KDA' exists in the pivot table before sorting
    if 'KDA' in pivot_table.columns:
        pivot_table = pivot_table.sort_values(by='KDA', ascending=False)
    else:
        print(f"'KDA' column not found in pivot table for {label}. Skipping sorting.")

    # Add the count of matches
    match_count = df.groupby('Player')['MatchId'].count()
    pivot_table['Matches'] = match_count

    # Replace any remaining NaNs with zeros if needed for display purposes
    pivot_table.fillna(0, inplace=True)

    # Generate HTML for the pivot table
    # Convert the DataFrame to HTML with borders around cells
    html_table = pivot_table.to_html(
        border=1,
        index=True,
        justify='center',
        classes='table table-bordered',
        table_id=label
    )

    # Return the complete HTML as a string
    return html_table

# Create pivot tables for each time window
html_output = """
<html>
<head>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            padding: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
"""

for label, df in time_windows.items():
    print(f"Creating HTML pivot table for {label}...")  # Notify which table is being created
    html_table = create_html_pivot_table(df, label)
    if html_table:
        html_output += f"<h2>Pivot Table for {label}</h2>"
        html_output += html_table

html_output += "</body></html>"

# Save the HTML output to a file
html_filename = "styled_pivot_tables.html"
with open(html_filename, "w") as f:
    f.write(html_output)

print("Styled pivot tables have been successfully exported to HTML.")

# Open the generated HTML file in the default web browser
webbrowser.open('file://' + os.path.realpath(html_filename))
