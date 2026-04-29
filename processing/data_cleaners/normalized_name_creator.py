import pandas as pd
from thefuzz import fuzz, process

df_stats = pd.read_excel(r'datasets\auction\retentions.xlsx')
df_sold = pd.read_csv(r'datasets\auction\auction_sold_csv.csv')
df_unsold=pd.read_csv(r'datasets\auction\auction_unsold_csv.csv')


official_names = list(set(df_sold['Player Name'].unique().tolist()).union(set(df_unsold['Player'].unique().tolist())))

stats_names = df_stats['player'].unique().tolist()

mapping_data = []

for s_name in stats_names:
    match, score = process.extractOne(s_name, official_names, scorer=fuzz.token_sort_ratio)
    if score >= 80: 
        mapping_data.append({
            'Stats_Name': s_name,
            'Official_Name': match,
            'Confidence': score,
            'Status': 'Matched'
        })
    else:
        mapping_data.append({
            'Stats_Name': s_name,
            'Official_Name': None,
            'Confidence': score,
            'Status': 'Check Manually'
        })
mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv(r'datasets\name_mapping_review.csv', index=False)

name_map = dict(zip(mapping_df['Stats_Name'], mapping_df['Official_Name']))
df_stats['player'] = df_stats['player'].map(name_map).fillna(df_stats['player'])

df_stats.to_csv(r'datasets\auction\retentions_normalized.csv', index=False)

print("Mapping complete. Check 'name_mapping_review.csv' to verify the changes.")