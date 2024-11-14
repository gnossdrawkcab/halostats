import pandas as pd
import webbrowser
import os
import numpy as np
from matplotlib import colors as mcolors
from datetime import datetime, timedelta
from matplotlib import cm

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
    'Matches', 'Kills', 'Deaths', 'Assists', 'KD', 'KDA','KD vs. EKD',
    'Exp. Kills', 'Exp. Deaths', 'Damage Done', 'Damage Taken', 'Dmg/KA', 'Dmg/Death', 
    'Dmg Difference', 'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Shots Fired', 'Shots Landed','Accuracy',
    'Grenade Kills', 'Headshot Kills', 'Power Weapon Kills', 'Melee Kills', 
    'Callout Assists',  'Score', 'Medals','Win Rate'
]


time_windows = {
    'Last 60 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=60))],
    'Last 30 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=30))],
    'Last 15 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=15))],
    'Last 7 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=7))],
    'Yesterday': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=1)).replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < datetime.now().replace(hour=5, minute=0, second=0))
    ],
    'Today': df_90_days[
        (df_90_days['Date'] >= datetime.now().replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < (datetime.now() + timedelta(days=1)).replace(hour=5, minute=0, second=0))
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

    # Define subsets for applying different gradient directions
    reverse_gradient_subset = ['Damage Taken', 'Dmg/KA', 'Exp. Deaths', 'Dmg/min (Taken)']
    regular_gradient_columns = [
        'Matches', 'Kills', 'Deaths', 'Assists', 'KD', 'KDA', 'KD vs. EKD',
        'Exp. Kills', 'Damage Done', 'Dmg/Death', 'Dmg Difference', 'Dmg/min (Dealt)',
        'Shots Fired', 'Shots Landed', 'Accuracy', 'Grenade Kills', 'Headshot Kills',
        'Power Weapon Kills', 'Melee Kills', 'Callout Assists', 'Score', 'Medals', 'Win Rate'
    ]

    # Initialize the styled table with basic styling
    styled_table = pivot_table.style \
        .set_caption(f"Player Performance Metrics ({label})") \
        .set_table_styles([
            # Apply word wrapping to headers, reduce column width, and customize header color
            {'selector': 'th', 'props': [
                ('text-align', 'center'),
                ('font-size', '13px'),
                ('background-color', '#f1f1f1'),  # Custom header background color
                ('color', 'black'),  # Optional: Change text color of the header
                ('border', '1px solid #ccc'),
                ('white-space', 'normal'),  # Enable word wrapping
                ('word-wrap', 'break-word'),  # Allow words to break for wrapping
                ('max-width', '85px'), # Restrict header column width
                ('font-weight', 'normal')
            ]},
            {'selector': 'td', 'props': [
                ('font-size', '13px'),
                ('text-align', 'center'),
                ('border', '0px solid #fff'),
                ('max-width', '70px')  # Restrict data column width
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#f5f5f5')
            ]},
        ])


    # Apply reverse gradient to the specified subset columns
    for col in reverse_gradient_subset:
        styled_table = styled_table.background_gradient(cmap='Greens_r', subset=[col])

    # Apply regular gradient to each remaining column individually
    for col in regular_gradient_columns:
        if col not in reverse_gradient_subset:  # Ensure no overlap with reverse gradient columns
            styled_table = styled_table.background_gradient(cmap='Greens', subset=[col])

    # Set font family for the entire table
    styled_table.set_table_attributes('style="font-family: Verdana;"')

    # Store the styled table in the dictionary
    styled_tables[label] = styled_table


# Reverse the order of the tables in styled_tables dictionary
reversed_styled_tables = dict(reversed(list(styled_tables.items())))

# Combine the styled tables' HTML into a single string
html_content = ""
for label, styled_table in reversed_styled_tables.items():
    html_content += styled_table.to_html() + "<br><br>"

# Save the HTML to a file
html_file = "player_performance_metrics.html"
with open(html_file, "w") as file:
    file.write(html_content)

# Open the saved HTML file in the default web browser
webbrowser.open("file://" + os.path.realpath(html_file))

