import pandas as pd

try:
    df1=pd.read_csv(r'datasets\batting_stats.csv',low_memory=False)
    df2=pd.read_csv(r'datasets\bowling_stats.csv',low_memory=False)
except FileNotFoundError:
    print("File Not Found")

df=df1.merge(df2,on=['season','player','team'],how='outer')

df.to_csv('datasets/ipl_stats.csv',index=False)

