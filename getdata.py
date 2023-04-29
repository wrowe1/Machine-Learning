import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time
import pandas as pd

SEASONS = list(range(2016,2023)) #list of nba seasons to scrape data
DATA_DIR = r'C:\Users\wkrow\Downloads\data'
STANDINGS_DIR = os.path.join(DATA_DIR, "standings") #directory for where to store standings info
SCORES_DIR = os.path.join(DATA_DIR, "scores") #directory to store box scores info

def get_html(url, selector, sleep=5, retries=3): #gets piece of the html from specified url and selector
	html = None
	for i in range(1, retries+1):
		time.sleep(sleep * i) #slows scrape process so don't get banned from basketball-reference, sleeps longer after each timeout
		try:
			with sync_playwright() as p:
				browser = p.chromium.launch() #open source chrome is browser
				page = browser.new_page()
				page.goto(url) #goes to url
				print(page.title())
				html = page.inner_html(selector) #grabs html piece
		except PlaywrightTimeout:
			print(f"Timeout error on {url}") #shows a TimeoutError when there is one
			continue #goes back to top of loop to try again if there is a timeout
		else:
			break #breaks loop if successful
	return html

def scrape_season(season):
	url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html" #grabs url of specified season
	html = get_html(url, "#content .filter") #gets html piece of page

	soup = BeautifulSoup(html, features="html.parser") #creates beautiful soup instance of html to parse
	links = soup.find_all("a")
	href = [l["href"] for l in links] #extracts hrefs from html
	standings_pages = [f"https://basketball-reference.com{l}" for l in href] #turns list of hrefs into list of links

	for url in standings_pages:
		save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1]) #saves to standings directory
		if os.path.exists(save_path): #if already saved, don't save again
			continue

		html = get_html(url, "#all_schedule") #gets full table of box scores from each standings page
		with open(save_path, "w+") as f:
			f.write(html) #writes full table to saved file

for season in SEASONS:
	scrape_season(season) #scrapes standings for each season

standings_files = os.listdir(STANDINGS_DIR)

def scrape_game(standings_file):
	with open(standings_file, 'r') as f:
		html = f.read()
	soup = BeautifulSoup(html, features="html.parser") #creates beautiful soup instance of html to parse
	links = soup.find_all("a") #gets all links in table of box scores
	hrefs = [l.get("href") for l in links] #grabs hrefs from html
	box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l] #filters to only give box score links
	box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores] #turns list of hrefs into list of links
	for url in box_scores:
		save_path = os.path.join(SCORES_DIR, url.split("/")[-1]) #saves to scores directory
		if os.path.exists(save_path): #if already saved, don't save again
			continue
		html = get_html(url, "#content") #grabs html of each box score
		if not html:
			continue
		with open(save_path, "w+", encoding='utf-8') as f:
			f.write(html) #writes box score into saved file

standings_files = [s for s in standings_files if ".html" in s]

for f in standings_files: #scrapes box scores in each standings file
	filepath = os.path.join(STANDINGS_DIR, f)
	scrape_game(filepath)