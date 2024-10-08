import pandas as pd
import webbrowser
import os

# Load the matches_90_days.csv file
df_90_days = pd.read_csv("matches_90_days.csv")

# Create the pivot table with the specified columns and their averages
pivot_table = pd.pivot_table(
    df_90_days,
    index='Player',  # Assuming 'Player' is the name of the player column
    values=[
        'Date', 'TotalKills', 'Deaths', 'Assists', 'KD', 'KDA',
        'ExpectedKills', 'ExpectedDeaths', 'Score', 
        'Dmg/KA', 'Dmg/Death', 'KD vs. EKD', 'Dmg Difference', 
        'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Accuracy', 
        'DamageDone', 'DamageTaken', 'ShotsFired', 
        'ShotsLanded', 'KillsGrenade', 'KillsHeadshot', 
        'KillsPower', 'KillsMelee', 'AssistsCallout', 'Medals'
    ],
    aggfunc={
        'Date': 'count',  # Count for Matches
        'TotalKills': 'mean',  # Average for Kills
        'Deaths': 'mean',  # Average for Deaths
        'Assists': 'mean',  # Average for Assists
        'KD': 'mean',  # Average for KD
        'KDA': 'mean',  # Average for KDA
        'ExpectedKills': 'mean',
        'ExpectedDeaths': 'mean',
        'Score': 'mean',
        'Dmg/KA': 'mean',
        'Dmg/Death': 'mean',
        'KD vs. EKD': 'mean',
        'Dmg Difference': 'mean',
        'Dmg/min (Dealt)': 'mean',
        'Dmg/min (Taken)': 'mean',
        'Accuracy': 'mean',
        'DamageDone': 'mean',
        'DamageTaken': 'mean',
        'ShotsFired': 'mean',
        'ShotsLanded': 'mean',
        'KillsGrenade': 'mean',
        'KillsHeadshot': 'mean',
        'KillsPower': 'mean',
        'KillsMelee': 'mean',
        'AssistsCallout': 'mean',
        'Medals': 'mean'
    },
    fill_value=0  # Fill missing values with 0
)

# Sort the pivot table by KDA in descending order
pivot_table.sort_values(by='KDA', ascending=False, inplace=True)

# Format the pivot table
for column in pivot_table.columns:
    if column in ['KD', 'KDA']:
        pivot_table[column] = pivot_table[column].round(2)
    elif column == 'KillsGrenade':
        pivot_table[column] = pivot_table[column].round(1)
    else:
        pivot_table[column] = pivot_table[column].round(0)

# Style the pivot table
styled_table = pivot_table.style\
    .set_caption("Player Performance Metrics")\
    .highlight_max(axis=0, color='lightgreen')\
    .highlight_min(axis=0, color='salmon')\
    .bar(subset=['KDA', 'KD'], color=['lightblue', 'lightcoral'])\
    .format(
        {
            'KDA': '{:.2f}',  # Two decimal places for KDA
            'KD': '{:.2f}',   # Two decimal places for KD
            'KillsGrenade': '{:.1f}',  # One decimal place for KillsGrenade
            'ExpectedKills': '{:.0f}',  # Zero decimal places for ExpectedKills
            'ExpectedDeaths': '{:.0f}',  # Zero decimal places for ExpectedDeaths
            'Score': '{:.0f}',  # Zero decimal places for Score
            'Dmg/KA': '{:.0f}',  # Zero decimal places for Dmg/KA
            'Dmg/Death': '{:.0f}',  # Zero decimal places for Dmg/Death
            'KD vs. EKD': '{:.0f}',  # Zero decimal places for KD vs. EKD
            'Dmg Difference': '{:.0f}',  # Zero decimal places for Dmg Difference
            'Dmg/min (Dealt)': '{:.0f}',  # Zero decimal places for Dmg/min (Dealt)
            'Dmg/min (Taken)': '{:.0f}',  # Zero decimal places for Dmg/min (Taken)
            'Accuracy': '{:.0f}',  # Zero decimal places for Accuracy
            'DamageDone': '{:.0f}',  # Zero decimal places for DamageDone
            'DamageTaken': '{:.0f}',  # Zero decimal places for DamageTaken
            'ShotsFired': '{:.0f}',  # Zero decimal places for ShotsFired
            'ShotsLanded': '{:.0f}',  # Zero decimal places for ShotsLanded
            'KillsHeadshot': '{:.0f}',  # Zero decimal places for KillsHeadshot
            'KillsPower': '{:.0f}',  # Zero decimal places for KillsPower
            'KillsMelee': '{:.0f}',  # Zero decimal places for KillsMelee
            'AssistsCallout': '{:.0f}',  # Zero decimal places for AssistsCallout
            'Medals': '{:.0f}'  # Zero decimal places for Medals
        }, na_rep='--'  # Move na_rep here
    )\
    .set_table_attributes('style="width: 100%; border-collapse: collapse;"')\
    .set_properties(**{'border': '1px solid black', 'padding': '5px'})\
    .set_properties(subset=['KDA', 'KD'], **{'text-align': 'right'})\
    .set_properties(subset=['KillsGrenade'], **{'text-align': 'center'})

# Save the styled pivot table to an HTML file
html_file_path = "kda_pivot_table.html"
styled_table.to_html(html_file_path, render_links=True)

# Open the HTML file in the default web browser
webbrowser.open('file://' + os.path.realpath(html_file_path))
