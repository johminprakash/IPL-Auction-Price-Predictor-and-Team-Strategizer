import csv,ast

try:
    with open('datasets\ipl_dataset_csv.csv',mode='r',newline='') as csvfile:
        csv_reader=csv.DictReader(csvfile)
        list1= list(csv_reader)

        plist={'2007/08':{},'2009':{},'2009/10':{},'2011':{},'2012':{},'2013':{},'2014':{},'2015':{},'2016':{},'2017':{},'2018':{},'2019':{},'2020/21':{},'2021':{},'2022':{},'2023':{},'2024':{},'2025':{}}

        for i in list1:
            if i['wicket_fielders']!="[]":
                teambowl={i['bowler'],*ast.literal_eval(i['wicket_fielders'])}
            else:
                teambowl={i['bowler']}

            if (i['batting_team'] in plist[i['season']]):
                plist[i['season']][i['batting_team']].update({i['batter'],i['non_striker']})
            else:
                plist[i['season']][i['batting_team']]={i['batter'],i['non_striker']}
            if (i['bowling_team'] in plist[i['season']]):
                plist[i['season']][i['bowling_team']].update(teambowl)
            else:
                plist[i['season']][i['bowling_team']]=(teambowl)
            
        for i,j in plist.items():
            for k,l in j.items():
                for m,n in j.items():
                    if (k!=m):
                        for o in l:
                            for p in n:
                                if o==p:
                                    print(f"Season:{i}\tteam_a:{k}\tteam_b:{m}\tplayer:{o}")

except FileNotFoundError:
    print("File Not Found")                