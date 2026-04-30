&#x20;                                             ===================================================

&#x20;                                             ||***IPL Auction Price Predictor and Team Strategist***||

&#x20;                                             ===================================================



&#x09;This project aims at building a ML based IPL Price Predictor and AI based suggestion for each team upon giving the required input



**Website:** https://ipl-auction-price-predictor-and-team-strategizer.streamlit.app/



**Overview**



This model begun from a simple IPL ball by ball data from Kaggle and sold and unsold data from IPL's unsold website. These data are processed and normalized using NumPy and Pandas. All the codes, raw files and processed files are given in processing folder.





The ball-by-ball IPL dataset (\~1GB) is not included in this repository due to size constraints.

It was used during preprocessing and feature engineering.



The final working app is given in the separate folder as app.



**Rating Brain:**

The rating is based on 7 attributes:

&#x09;1)experience

&#x09;2)batting intent

&#x09;3)batting consistency

&#x09;4)pace wicket taker

&#x09;5)pace economy

&#x09;6)spin wicket taker

&#x09;7)spin economy

For each attribute top 5% players are choosen as gold standard and marked 100 and other players are rated based on this each season. The team rating are  computed by finding the average of this using suitable computations.



**top4\_engine.py:**

This script learns the winning DNA of teams by using the top 4 finishers in each season and their team rating. The file outputs its data into three files

&#x09;1)*processed\_ml\_master.xls*

&#x09;2)*top4\_profiles.xls*

&#x09;3)*top4\_engine.pkl*



**base\_price\_engine.py:**

This script learns how the price of players are affected from their historical IPL performances and creates a file named '*base\_price\_engine.pkl*' tht contains all ML learnings.



**multipliers\_engine.py:**

The script finds 3 multipliers

&#x09;1) demand\_multiplier

&#x09;2) scarcity\_multiplier

&#x09;3) purse\_multiplier

and changes how they vary bsed on different market situations for each season and creates a file named '*multipliers\_engine.pkl*'

**app.py:**
This script creates graph to show how the 7 attributes had changed for winning DNA over the years. It then uses *Linear Regression* to predict what will be the required value of all attributes for the next year. Then it takes 3 inputs from the website:
1)Auction List CSV

&#x09;2)Retentions CSV

&#x09;3)Purse CSV
It then rates the teams based on '*Retentions CSV*' and find the demands of the team in all 7 attributes. Then it uses the ML model's result to predict the prices of players and show them into a downloadable table format. Atlast it gives AI based suggestions of players with fit\_rating for each team to minimize the gap.





**How to Run:**
Open your terminal in the app folder and run the following commands in order
python top4\_engine.py
python base\_price\_engine.py
python multipliers\_engine.py
streamlit run app.py

