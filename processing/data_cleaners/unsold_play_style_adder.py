import pandas as pd

# 1. Load your datasets using the paths you provided
try:
    sold_df = pd.read_csv(r'final\data\auction_sold_final_csv.csv')
    unsold_df = pd.read_csv(r'final\data\auction_unsold_final_csv.csv')
except FileNotFoundError as e:
    print(f"❌ Path Error: {e}")
    exit()

# 2. Name Normalization (Crucial for matching)
# This creates a hidden version of names with no spaces to ensure a match
def normalize(name):
    return str(name).replace(" ", "").lower().strip()

sold_df['norm_name'] = sold_df['Player Name'].apply(normalize)
unsold_df['norm_name'] = unsold_df['Player'].apply(normalize)

# 3. Create the Mapping (Using normalized names)
# We take the most recent 'Play type' for each player
style_map = sold_df[['norm_name', 'Play type']].drop_duplicates(subset=['norm_name'])
style_lookup = style_map.set_index('norm_name')['Play type']

# 4. Create the NEW Column in Unsold
# This matches based on the hidden 'norm_name' and fills it into 'Play type'
unsold_df['Play type'] = unsold_df['norm_name'].map(style_lookup)

# 5. Clean up: Fill remaining unknowns and remove the helper column
unsold_df['Play type'] = unsold_df['Play type'].fillna('Unknown')
unsold_df = unsold_df.drop(columns=['norm_name'])

# 6. Save to the new destination
# Make sure the 'datasets' folder exists before running this
import os
if not os.path.exists('datasets'):
    os.makedirs('datasets')

unsold_df.to_csv(r'datasets\auction_unsold_csv_playstyle.csv', index=False)

# Quick stats for you
matched_count = unsold_df[unsold_df['Play type'] != 'Unknown'].shape[0]
print(f"✅ Mapping Complete!")
print(f"📊 Total Unsold Players: {len(unsold_df)}")
print(f"🎯 Successfully matched Play Styles for: {matched_count} players")
print(f"📂 File saved to: datasets\\auction_unsold_csv_playstyle.csv")