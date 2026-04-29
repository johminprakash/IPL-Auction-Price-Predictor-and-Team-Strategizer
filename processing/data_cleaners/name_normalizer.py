import pandas as pd

def cols_to_dict1(group):
    return dict(zip(group['Season'].astype(str),group['Team']))

def cols_to_dict2(group):
    return dict(zip(group['season'].astype(str),group['team']))

def name_changer(group,name):
    group['player']=name
    return group

df1=pd.read_csv(r'datasets\auction\auction_sold_csv.csv',low_memory=False)
df2=pd.read_csv(r'datasets\stats\ipl_stats_csv.csv',low_memory=False)
df3=pd.read_csv(r'datasets\auction\retentions_normalized.csv',low_memory=False)

df3=df3.rename(columns={'player':'Player Name','team':'Team','season':'Season'})

df=pd.concat([df1[['Player Name','Team','Season']],df3[['Player Name','Team','Season']]])
players_original=df[['Player Name','Team','Season']].groupby('Player Name').apply(cols_to_dict1).reset_index(name='Dict1')
players=df2[df2['season'].isin(['2016','2017','2018','2019','2020','2021','2022','2023','2024','2025'])].groupby('player').apply(cols_to_dict2).reset_index(name='Dict2')

players.to_csv('players.csv')


d={}
for index,i in players.iterrows():
    count=0
    flag=False
    for index_original,j in players_original.iterrows():
        if(i['player'].split(' ')[-1]==j['Player Name'].split(' ')[-1]):
            if i['player'] in d:
                d[i['player']].append(j['Player Name'])
            else:
                d[i['player']]=[j['Player Name'],]

dfd=pd.DataFrame.from_dict(d,orient='index')
dfd.to_csv('name_match.csv')


matched={}
for i in d:
    matched[i]=[]
    for j in d[i]:
        flag=False
        season1=players[players['player'] == i]['Dict2'].iloc[0].keys()
        season2=players_original[players_original['Player Name'] == j]['Dict1'].iloc[0].keys()
        commonseason=list(set(season1).intersection(set(season2)))
        if commonseason:
            flag=True
            for k in commonseason:
                if players[players['player']==i]['Dict2'].iloc[0][k]!=players_original[players_original['Player Name'] == j]['Dict1'].iloc[0][k]:
                    flag=False
                    break
        if(flag):
            matched[i].append(j)

perfect_matched=[]
unmatched=[]
discrepancy=[]
for i in matched:
    match len(matched[i]):
        case 0:
            unmatched.append(i)
        case 1:
            perfect_matched.append([i,matched[i][0]])
        case _:
            discrepancy.append([i, matched[i]])

pm=pd.DataFrame(perfect_matched,columns=['stats','sold'])
pm.to_csv(r'datasets\normalized\sold-stats.csv',index=False)

um=pd.DataFrame(unmatched,columns=['Name'])
um.to_csv(r'datasets\normalized\sold-stats_unmatched.csv',index=False)

d=pd.DataFrame(discrepancy,columns=['stats','sold'])
d.to_csv('datasets/normalized/sold-stats_discrepancy.csv',index=False)




