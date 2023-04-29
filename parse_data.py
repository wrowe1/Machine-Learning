import os
import pandas as pd
from bs4 import BeautifulSoup

SCORE_DIR = r'C:\Users\wkrow\Downloads\data\scores'
box_scores = os.listdir(SCORE_DIR) #lists all box scores
box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith(".html")] #gets full file path to box scores

def parse_html(box_score):
	with open(box_score, encoding='Latin-1') as f:
		html = f.read() #reads file into html variable
	soup = BeautifulSoup(html) #creates beautiful soup instance of html to parse
	[s.decompose() for s in soup.select("tr.over_header")] #recleans html for further processing
	[s.decompose() for s in soup.select("tr.thead")] 
	return soup

def read_line_score(soup): 
	line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0] #returns line score to a dataframe
	cols = list(line_score.columns)
	cols[0] = "team" #labels first column as team
	cols[-1] = "total" #labels last column as total points scored
	line_score.columns = cols

	line_score = line_score[["team", "total"]]
	return line_score #returns dataframe (df) of team and total

def read_stats(soup, team, stat):
	df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0] #reads in team html to a df
	df = df.apply(pd.to_numeric, errors="coerce") #converts any string columns to numeric so df can be analyzed numerically
	return df

def read_season_info(soup):
	nav = soup.select("#bottom_nav_container")[0] #searches in nav container html
	hrefs = [a["href"] for a in nav.find_all("a")] #finds and pulls out link in nav container
	season = os.path.basename(hrefs[1]).split("_")[0] #pulls out season number from link
	return season

base_cols = None

games = []

for box_score in box_scores: #for every box score
	soup = parse_html(box_score)
	line_score = read_line_score(soup)
	teams = list(line_score["team"])

	summaries = []
	for team in teams:
		basic = read_stats(soup, team, "basic") #reads basic stats
		advanced = read_stats(soup, team, "advanced") #reads advanced stats
		totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]]) #concatinates advanced df below basic df
		totals.index = totals.index.str.lower() #makes stat names lower case
		maxes = pd.concat([basic.iloc[:-1,:].max(), advanced.iloc[:-1,:].max()]) #concats highest number of an advanced stat df below basic stat df
		maxes.index = maxes.index.str.lower() + "_max" #makes stat names lower case and adds _max to end
		summary = pd.concat([totals, maxes]) #concats totals and maxes

		if base_cols is None: #for first loop through, find all values in box score to look for in other box scores
			base_cols = list(summary.index.drop_duplicates(keep="first")) #remove duplicate stat cols
			base_cols = [b for b in base_cols if "bpm" not in b]

		summary = summary[base_cols] #only select columns we had in first summary
		summaries.append(summary) #append summary df to list, which wil contain both summaries of both teams in a given game

	summary = pd.concat(summaries, axis=1).T #concats summaries to df summary. Transpose make stats on top with each stat a column

	game = pd.concat([summary, line_score], axis=1) #combine summary and line score
	game["home"] = [0, 1] #assigns new stat home with 0 for away team and 1 for home team
	game_opp = game.iloc[::-1].reset_index() #new df that is game df but flipped (first and second row swapped)
	game_opp.columns += "_opp" 

	full_game = pd.concat([game, game_opp], axis=1) #concat opp df to right of game df.
	full_game["season"] = read_season_info(soup) #new column with season number
	full_game["date"] = os.path.basename(box_score)[:8] #gets date from file name of box score
	full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d") #formats date to y-m-d
	full_game["won"] = full_game["total"] > full_game["total_opp"] #new column with true if won else false

	games.append(full_game) #appends full_game df for one game to games df

	if len(games) % 100 == 0: #progress messages for loop every 100 games
		print(f"{len(games)} / {len(box_scores)}") #prints how many box scores completed/total

games_df = pd.concat(games, ignore_index=True) #concatinates games together with games as rows
games_df.to_csv("nba_games.csv") #saves df to a csv