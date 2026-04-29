import pandas as pd

teams = ["csk","dc","kkr","mi","kxip","rr","rcb","srh"]
columns = ['season','team','player']

rows = []  # collect dictionaries here

for team in teams:
    ip = input(f"Enter players for {team} in season 2019, comma-separated: ")
    out = ip.split(',')
    for player in out:
        rows.append({'season': 2019, 'team': team, 'player': player.strip()})

df = pd.DataFrame(rows, columns=columns)

file_path = 'datasets/retentions1.xlsx'

try:
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
except FileNotFoundError:
    df.to_excel(file_path, index=False)
