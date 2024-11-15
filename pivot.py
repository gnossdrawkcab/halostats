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
    'Stack', 'PreCsr', 'PostCsr', 'TotalKills', 'Deaths', 'Assists', 'KD', 'KDA',
    'ExpectedKills', 'ExpectedDeaths', 'Score',
    'Dmg/KA', 'Dmg/Death', 'KDvEKD', 'Dmg Difference',
    'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Accuracy',
    'DamageDone', 'DamageTaken', 'ShotsFired',
    'ShotsLanded', 'KillsGrenade', 'KillsHeadshot',
    'KillsPower', 'KillsMelee', 'AssistsCallout', 'Medals', 'Win'
]

df_90_days[numeric_columns] = df_90_days[numeric_columns].apply(pd.to_numeric, errors='coerce')
df_90_days['Date'] = pd.to_datetime(df_90_days['Date'], errors='coerce')
df_90_days.replace([np.inf, -np.inf], np.nan, inplace=True)

# Ensure KDA exists in the CSV after loading
if 'KDA' in df_90_days.columns:
    print("KDA exists in the original data!")
else:
    print("KDA column is missing in the data!")
    
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
    'Matches', 'Kills', 'Deaths', 'Assists', 'KD', 'KDA','KDvEKD',
    'Exp. Kills', 'Exp. Deaths', 'Damage Done', 'Damage Taken', 'Dmg/KA', 'Dmg/Death', 
    'Dmg Difference', 'Dmg/min (Dealt)', 'Dmg/min (Taken)', 'Shots Fired', 'Shots Landed','Accuracy',
    'Grenade Kills', 'Headshot Kills', 'Power Weapon Kills', 'Melee Kills', 
    'Callout Assists',  'Score', 'Medals','Win Rate'
]

# Check columns after renaming
print(df_90_days.columns)

time_windows = {
    'Last 60 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=60))],
    'Last 30 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=30))],
    'Last 15 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=15))],
    'Last 7 Days': df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=7))],
    'Last 7 Days - 4 Stack': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=7))) & 
        (df_90_days['Stack'].isin([4]))
    ],
    'Last 7 Days - 3 or 4 Stack': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=7))) & 
        (df_90_days['Stack'].isin([3, 4]))
    ],
    'Yesterday - 3 or 4 Stack': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=1)).replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < datetime.now().replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Stack'].isin([3, 4]))],
    'Yesterday': df_90_days[
        (df_90_days['Date'] >= (datetime.now() - timedelta(days=1)).replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < datetime.now().replace(hour=5, minute=0, second=0))
    ],
    'Today': df_90_days[
        (df_90_days['Date'] >= datetime.now().replace(hour=5, minute=0, second=0)) & 
        (df_90_days['Date'] < (datetime.now() + timedelta(days=1)).replace(hour=5, minute=0, second=0))
    ],

}

def create_pivot_table(df, label):
    # Ensure 'TotalKills' and 'Deaths' are included in numeric_columns if not already there
    required_columns = ['TotalKills', 'Deaths', 'PreCsr']  # Include PreCSR for the calculation
    for col in required_columns:
        if col not in numeric_columns:
            numeric_columns.append(col)

    # Create the pivot table with aggregation
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

    # Add Initial CSR column: Get PreCSR value from the earliest date for each player
    df_sorted_by_date = df.sort_values('Date')  # Sort by date to get the earliest match
    initial_csr = df_sorted_by_date.groupby('Player').first()['PreCsr']  # Get the first PreCSR for each player
    pivot_table['Initial CSR'] = pivot_table.index.map(initial_csr)  # Map Initial CSR to the pivot table players

    # Add Final CSR column: Get PostCSR value from the latest date for each player
    df_sorted_by_date_desc = df.sort_values('Date', ascending=False)  # Sort by date descending to get the latest match
    final_csr = df_sorted_by_date_desc.groupby('Player').first()['PostCsr']  # Get the last PostCSR for each player
    pivot_table['Final CSR'] = pivot_table.index.map(final_csr)  # Map Final CSR to the pivot table players

    # Add CSR Change column: Final CSR minus Initial CSR
    pivot_table['CSR Change'] = pivot_table['Final CSR'] - pivot_table['Initial CSR']  # Calculate CSR Change

    # Reindex columns to ensure the correct order
    pivot_table = pivot_table.reindex(columns=desired_column_order + ['Initial CSR', 'Final CSR', 'CSR Change'])

    # Sort the table by KDA in descending order
    pivot_table = pivot_table.sort_values(by='KDA', ascending=False)

    # Format columns
    for column in pivot_table.columns:
        if column in ['KD', 'KDA', 'KDvEKD', 'Win Rate']:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{x:.2f}")
        elif column in ['Grenade Kills', 'Accuracy', 'Power Weapon Kills', 'Melee Kills', 'Medals', 'Headshot Kills', 'Kills', 'Deaths', 'Assists', 'Exp. Deaths', 'Exp. Kills', 'Callout Assists']:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{x:.1f}")
        else:
            pivot_table[column] = pivot_table[column].apply(lambda x: f"{int(x):d}" if x == round(x) else f"{x:.0f}")

    return pivot_table


# Loop through time windows to create and format pivot tables
styled_tables = {}
for label, df in time_windows.items():
    # Check if the DataFrame is empty
    if df.empty:
        print(f"Skipping {label} - No data available")
        continue  # Skip creating a pivot table if there is no data
    
    print(f"{label} - Number of Matches: {df.shape[0]}")  # Check number of matches

    # Check for duplicates in the DataFrame
    print("Number of duplicate rows:", df.duplicated().sum())

    pivot_table = create_pivot_table(df, label)

    # Manipulate player names to remove ' l ' and ' l' for display only
    pivot_table.index = pivot_table.index.to_series().apply(lambda x: x.replace("l ", "").replace(" l", "") if isinstance(x, str) else x)

    # Inspect the pivot table
    print(f"{label} - Unique Players: {pivot_table.index.nunique()}")  # Check unique players

    # Replace NaN values with 0 or appropriate default value
    pivot_table.fillna(0, inplace=True)

    # Define subsets for applying different gradient directions
    reverse_gradient_subset = [#'Damage Taken', 
                               'Dmg/KA', 
                               #'Exp. Deaths', 
                               #'Dmg/min (Taken)'
                               ]
    regular_gradient_columns = [
        #'Matches', 
        'Kills', 
        'Deaths', 
        'Assists', 
        'KD', 
        'KDA', 
        'KDvEKD',
        #'Exp. Kills', 
        #'Damage Done', 
        'Dmg/Death', 
        'Dmg Difference', 
        #'Dmg/min (Dealt)',
        #'Shots Fired', 
        #'Shots Landed', 
        'Accuracy', 
        #'Grenade Kills', 
        #'Headshot Kills',
        #'Power Weapon Kills', 
        #'Melee Kills', 
        'Callout Assists', 
        #'Score', 
        #'Medals', 
        'Win Rate'
    ]

    # Initialize the styled table with basic styling
    styled_table = pivot_table.style \
        .set_caption(f"{label}") \
        .set_table_styles([
            {'selector': 'caption', 'props': [  # Style for the caption (table title)
            ('text-align', 'left'),  # Left-justify the caption
            ('font-size', '16px'),  # Set font size for better visibility
            ('font-weight', 'bold'),  # Make the caption bold
            ('color', 'white'),  # Set the caption text color to white
            ('background-color', '393939'),  # Set the background color to black
            ('padding', '10px 15px')  # Add padding to the caption for better spacing
        ]},
        {'selector': 'th', 'props': [
                ('text-align', 'left'),
                ('font-size', '13px'),
                ('background-color', 'f6f6f6'),
                ('color', 'black'),
                ('border', '1px solid #ccc'),
                ('white-space', 'normal'),
                ('word-wrap', 'break-word'),
                ('max-width', '150px'),
                ('font-weight', 'normal')
            ]},
            # Style for the player name column (left-justify the player names)
            {'selector': 'td:nth-child(1)', 'props': [
                ('text-align', 'left'),  # Left-justify player names
                ('font-size', '13px'),
                ('border', '0px solid #fff'),
                ('max-width', '85px')
            ]},
            {'selector': 'td', 'props': [
                ('font-size', '13px'),
                ('text-align', 'center'),
                ('border', '0px solid #fff'),
                ('max-width', '70px')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', 'f6f6f6')
            ]},
        ])

    # Apply reverse gradient to the specified subset columns
    for col in reverse_gradient_subset:
        styled_table = styled_table.background_gradient(cmap='summer', subset=[col])

    # Apply regular gradient to each remaining column individually
    for col in regular_gradient_columns:
        if col not in reverse_gradient_subset:
            styled_table = styled_table.background_gradient(cmap='summer_r', subset=[col])

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

# Sort the entire dataset by date in descending order
sorted_db_table = df_90_days.sort_values(by='Date', ascending=False)

# Filter the dataset to show only the last 15 days for the final table
final_table_data = df_90_days[df_90_days['Date'] >= (datetime.now() - timedelta(days=15))]

# List of columns to exclude from the final table
columns_to_exclude = ['UniqueID', 'SeasonNumber', 'MatchId']

# Drop the specified columns from the DataFrame if they exist
final_table_data = sorted_db_table.drop(columns=columns_to_exclude, errors='ignore')

# Apply styling to the final table
styled_final_table = final_table_data.style \
    .set_caption('ALL MATCHES (Last 15 Days)') \
    .set_table_styles([
        {'selector': 'caption', 'props': [  # Style for the caption (table title)
            ('text-align', 'left'),  # Left-justify the caption
            ('font-size', '16px'),  # Set font size for better visibility
            ('font-weight', 'bold'),  # Make the caption bold
            ('color', 'white'),  # Set the caption text color to white
            ('background-color', '#393939'),  # Set the background color to black
            ('padding', '10px 15px')  # Add padding to the caption for better spacing
        ]},
        {'selector': 'th', 'props': [
            ('text-align', 'center'),  # Center-align header text for better aesthetics
            ('font-size', '13px'),
            ('background-color', 'f6f6f6'),
            ('color', 'black'),
            ('border', '1px solid #ccc'),
            ('white-space', 'nowrap'),  # Prevent word wrapping
            ('font-weight', 'normal')
        ]},
        # Style for the player name column (left-justify the player names)
        {'selector': 'td:nth-child(1)', 'props': [
            ('text-align', 'left'),
            ('font-size', '13px'),
            ('border', '0px solid #fff'),
            ('white-space', 'nowrap')
        ]},
        {'selector': 'td', 'props': [
            ('font-size', '13px'),
            ('text-align', 'center'),
            ('border', '0px solid #fff'),
            ('white-space', 'nowrap'),  # Prevent word wrapping in data
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', 'f6f6f6')
        ]},
    ]) \
    .format({col: "{:.2f}" if col in ['KD', 'KDA', 'KDvEKD', 'Win Rate'] else
             "{:.1f}" if col in ['Grenade Kills', 'Accuracy', 'Power Weapon Kills', 'Melee Kills',
                                 'Medals', 'Headshot Kills', 'Kills', 'Deaths', 'Assists',
                                 'Exp. Deaths', 'Exp. Kills', 'Callout Assists'] else
             "{:.0f}" for col in numeric_columns if col in final_table_data.columns}) \
    .set_table_attributes('style="font-family: Verdana;"')
    

""" # Filter data for Stack = 4
filtered_stack_4_data = final_table_data[final_table_data['Stack'] == 4]

# Group the data by Date and reset index to make Date the row index
grouped_by_date = filtered_stack_4_data.groupby('Date').agg('sum').reset_index()

# Apply the same styling as before, but for the new table (grouped by Date)
styled_stack_4_table = grouped_by_date.style \
    .set_caption('4-STACK GAMES') \
    .set_table_styles([
        {'selector': 'caption', 'props': [
            ('text-align', 'left'),
            ('font-size', '16px'),
            ('font-weight', 'bold'),
            ('color', 'white'),
            ('background-color', '#464e46'),
            ('padding', '10px 15px')
        ]},
        {'selector': 'th', 'props': [
            ('text-align', 'center'),
            ('font-size', '13px'),
            ('background-color', '#f1f1f1'),
            ('color', 'black'),
            ('border', '1px solid #ccc'),
            ('white-space', 'nowrap'),
            ('font-weight', 'normal')
        ]},
        {'selector': 'td', 'props': [
            ('font-size', '13px'),
            ('text-align', 'center'),
            ('border', '0px solid #fff'),
            ('white-space', 'nowrap')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]},
    ]) \
    .format({col: "{:.2f}" if col in ['KD', 'KDA', 'KDvEKD', 'Win Rate'] else
             "{:.1f}" if col in ['Grenade Kills', 'Accuracy', 'Power Weapon Kills', 'Melee Kills',
                                 'Medals', 'Headshot Kills', 'Kills', 'Deaths', 'Assists',
                                 'Exp. Deaths', 'Exp. Kills', 'Callout Assists'] else
             "{:.0f}" for col in numeric_columns if col in grouped_by_date.columns}) \
    .set_table_attributes('style="font-family: Verdana;"') """

# Initialize html_content only once
html_content = ""

# Add the styled pivot tables (reversed_styled_tables) to the HTML content
for label, styled_table in reversed_styled_tables.items():
    html_content += styled_table.to_html() + "<br><br>"

""" # Add the styled table for Stack = 4
html_content += f"<br><br>{styled_stack_4_table.to_html()}" """

# Add the styled final table at the end
html_content += f"<br><br>{styled_final_table.to_html()}"

# Save the combined HTML content to a file
html_file_path = "index.html"
with open(html_file_path, "w") as f:
    f.write(html_content)

# Open the saved HTML file in the browser
webbrowser.open(f"file://{os.path.abspath(html_file_path)}")
