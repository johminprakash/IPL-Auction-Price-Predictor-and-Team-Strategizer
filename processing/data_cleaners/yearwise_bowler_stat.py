import pandas as pd

try:
    df=pd.read_csv('datasets\ipl_dataset_csv.csv',low_memory=False)
except FileNotFoundError:
    print("File Not Found")

df1=df[['season','bowler','bowling_team']].drop_duplicates().reset_index().rename(columns={'bowling_team':'team'})

temp=df.groupby(['season','bowler'])['match_number'].nunique().reset_index().rename(columns={'match_number':'innings'})
df1=df1.merge(temp,on=['season','bowler'],how='left')

temp=df.groupby(['season','bowler']).apply(lambda x:x[['match_number','over','ball']].drop_duplicates().shape[0]).reset_index(name='balls')
df1=df1.merge(temp,on=['season','bowler'],how='left')

df1['overs']=(df1['balls']//6).astype(str)+'.'+(df1['balls']%6).astype(str)

temp=df.groupby(['season','bowler'])['runs_conceeded_for_bowlers'].sum().reset_index().rename(columns={'runs_conceeded_for_bowlers':'runs_conceeded'})
df1=df1.merge(temp,on=['season','bowler'],how='left')

tempdropped=df[df['wicket_kind'].notna() & (df['wicket_kind']!='run out')]
temp=tempdropped.groupby(['season','bowler']).size().reset_index(name='wickets')
df1=df1.merge(temp,on=['season','bowler'],how='left')

df1['economy']=round(df1['runs_conceeded']/df1['balls']*6,2)
df1['average']=round(df1['runs_conceeded']/df1['wickets'],2)
df1['strike_rate']=round(df1['balls']/df1['wickets'],2)

df1.rename(columns={'bowler':'player'},inplace=True)

df1.to_csv('datasets/bowling_stats.csv',index=False)

print(df1)