import pandas as pd

df1 = pd.read_csv(r'datasets\stats\ipl_stats_csv.csv', low_memory=False)
df_map = pd.read_csv(r'datasets\normalized\sold-stats.csv', low_memory=False)

yrs = ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']
df1 = df1[df1['season'].astype(str).isin(yrs)].copy()

name_map = dict(zip(df_map['stats'], df_map['norm']))

df1['player'] = df1['player'].map(name_map).fillna(df1['player'])

df1.to_csv(r'datasets\stats\ipl_stats_normalized.csv', index=False)

print("Mapping complete. File saved.")