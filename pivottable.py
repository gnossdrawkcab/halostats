import pandas as pd
import webbrowser
import os
import numpy as np
from matplotlib import colors as mcolors
from datetime import datetime, timedelta

# Load the matches_90_days.csv file
df_90_days = pd.read_csv("matches_90_days.csv")

# Remove 'UniqueID' from numeric_columns to avoid incorrect aggregation
numeric_columns = [
    'TotalKills', 'Deaths', 'Assists', 'KD', 'KDA',
    'ExpectedKills', 'ExpectedDeaths', 'Score',
    'Dmg/KA', 'Dmg/Death', 'KD vs. EKD', 'Dmg Difference',
    'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Accuracy',
    'DamageDone', 'DamageTaken', 'ShotsFired',
    'ShotsLanded', 'KillsGrenade', 'KillsHeadshot',
    'KillsPower', 'KillsMelee', 'AssistsCallout', 'Medals', 'Win'
]

df_90_days[numeric_columns] = df_90_days[numeric_columns].apply(pd.to_numeric, errors='coerce')
df_90_days['Date'] = pd.to_datetime(df_90_days['Date'], errors='coerce')
df_90_days.replace([np.inf, -np.inf], np.nan, inplace=True)

# Create a mapping of old column names to new column names
new_column_names = {
    'UniqueID': 'Matches',
    'TotalKills': 'Kills',
    'DamageDone': 'Damage Done',
    'DamageTaken': 'Damage Taken',
    'ShotsFired': 'Shots Fired',
    'ExpectedDeaths': 'Exp. Deaths',
    'ExpectedKills': 'Exp. Kills',
    'ShotsLanded': 'Shots Landed',
    'KillsGrenade': 'Grenade Kills',
    'KillsHeadshot': 'Headshot Kills',
    'KillsPower': 'Power Weapon Kills',
    'KillsMelee': 'Melee Kills',
    'AssistsCallout': 'Callout Assists',
    'Win': 'Win Rate'
}

desired_column_order = [
    'Matches', 'Kills', 'Deaths', 'Assists', 'KD', 'KDA',
    'Exp. Kills', 'Exp. Deaths', 'Damage Done', 'Damage Taken', 'Dmg/KA', 'Dmg/Death', 
    'Dmg Difference', 'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Shots Fired', 'Shots Landed',
    'Grenade Kills', 'Headshot Kills', 'Power Weapon Kills', 'Melee Kills', 
    'Callout Assists', 'Accuracy', 'Score', 'KD vs. EKD', 'Medals','Win Rate'
]


# Time windows for pivot tables (e.g. 90, 30, 15, 7, 1 days)
time_windows = {
    'All Time': df_90_days,
    'Last 30 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=30))],
    'Last 15 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=15))],
    'Last 7 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=7))],
    'Last 1 Day': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=1)).replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < datetime.now().replace(hour=5, minute=0, second=0))
    ]
}

# Function to create pivot tables
def create_pivot_table(df, label):
    pivot_table = pd.pivot_table(
        df,
        index='Player',
        values=numeric_columns,
        aggfunc={col: 'mean' for col in numeric_columns},
        fill_value=0
    ).rename(columns=new_column_names)

    # Add the count of matches (or unique matches if UniqueID is used)
    match_count = df.groupby('Player')['UniqueID'].nunique()  # Count of unique matches
    pivot_table['Matches'] = match_count  # Add this as a new column in the pivot table

    # Sort by KDA
    pivot_table.sort_values(by='KDA', ascending=False, inplace=True)

    # Reindex columns
    pivot_table = pivot_table.reindex(columns=desired_column_order)

    # Format columns
    for column in pivot_table.columns:
        if column in ['KD', 'KDA', 'KD vs. EKD','Win Rate']:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{x:.2f}")
        elif column in ['Grenade Kills', 'Power Weapon Kills', 'Melee Kills', 'Headshot Kills', 'Kills', 'Deaths', 'Assists', 'Exp. Deaths', 'Exp. Kills']:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{x:.1f}")
        else:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{int(x):d}" if x == round(x) else f"{x:.0f}")

    return pivot_table

# Loop through time windows to create and format pivot tables
styled_tables = {}
for label, df in time_windows.items():
    print(f"{label} - Number of Matches: {df.shape[0]}")  # Check number of matches

    # Check for duplicates in the DataFrame
    print("Number of duplicate rows:", df.duplicated().sum())

    pivot_table = create_pivot_table(df, label)

    # Inspect the pivot table
    print(f"{label} - Unique Players: {pivot_table.index.nunique()}")  # Check unique players

    # Replace NaN values with 0 or appropriate default value
    pivot_table.fillna(0, inplace=True)

    # Ensure that only valid numeric columns are used for background gradients
    numeric_cols = pivot_table.select_dtypes(include=['number']).columns

    # Define the subsets for the background gradient
    non_reverse_gradient_subset = numeric_cols.difference(['Damage Taken', 'Dmg/KA', 'Exp. Deaths', 'Dmg/min (Taken)'])
    reverse_gradient_subset = ['Damage Taken', 'Dmg/KA', 'Exp. Deaths', 'Dmg/min (Taken)']

    styled_table = pivot_table.style \
        .set_caption(f"Player Performance Metrics ({label})") \
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '12px'), ('background-color', '#D3D3D3'), ('border', '1px solid #ccc')]},
            {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid #ccc')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5')]},
        ]) \
        .background_gradient(cmap='YlGnBu', subset=non_reverse_gradient_subset) \
        .background_gradient(cmap='YlGnBu_r', subset=reverse_gradient_subset) \
        .set_table_attributes('style="font-family: Verdana;"')  # Set Verdana font

    styled_tables[label] = styled_table

# Combine the styled tables' HTML into a single string
html_content = ""
for label, styled_table in styled_tables.items():
    html_content += styled_table.to_html() + "<br><br>"

# Save the HTML to a file
html_file = "player_performance_metrics.html"
with open(html_file, "w") as file:
    file.write(html_content)

# Open the saved HTML file in the default web browser
webbrowser.open("file://" + os.path.realpath(html_file))
