import pandas as pd
from collections import Counter

try:
    df=pd.read_csv('datasets\ipl_dataset_csv.csv',low_memory=False)
except FileNotFoundError:
    print("File Not Found")

df1=df[['season','batter','batting_team']].drop_duplicates().reset_index().rename(columns={'batting_team':'team'}) #make unique columns to find number of matches

temp=df.groupby(['season','batter'])['match_number'].nunique().reset_index().rename(columns={'match_number':'innings'})
df1=df1.merge(temp,on=['season','batter'],how='left')

temp=df.groupby(['season','batter'])['runs_batter'].sum().reset_index().rename(columns={'runs_batter':'runs'})
df1=df1.merge(temp,on=['season','batter'],how='left')

temp=df.groupby(['season','batter']).apply(lambda x: x[['match_number','over','ball']].drop_duplicates().shape[0]).reset_index(name='balls')
df1=df1.merge(temp,on=['season','batter'],how='left')

df1['strike_rate']=round(df1['runs']/df1['balls']*100,2)

season_group=df.groupby('season')
temp=pd.DataFrame()
for season,group in season_group:
    dismissal_list=group['wicket_player_out'].dropna().tolist()
    outs_per_player=Counter(dismissal_list)
    season_temp=pd.DataFrame()
    for i in outs_per_player:
        tempdf=pd.DataFrame({'season':[season],'batter':[i],'dismissals':[outs_per_player[i]]})
        season_temp=pd.concat([season_temp,tempdf],ignore_index=True)
    temp=pd.concat([temp,season_temp],ignore_index=True)
df1=df1.merge(temp,on=['season','batter'],how='left')
df1['dismissals'] = df1['dismissals'].fillna(0)
df1['average']=round(df1['runs']/df1['dismissals'].replace(0,1),2)

df1.rename(columns={'batter':'player'},inplace=True)

df1.to_csv('datasets/batting_stats.csv',index=False)