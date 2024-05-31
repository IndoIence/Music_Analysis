from classes.scraper import GeniusScraper, RedirectPrints
from utils import CONFIG

artists_names = open(CONFIG["to_scrape_file"], "r").read().strip().split("\n")
scraper = GeniusScraper()
ouptut_file = CONFIG["scraper_prints"]
with open(ouptut_file, "a", buffering=1) as f, RedirectPrints(f):
    print("Printing to file :3 \n")
    scraper.scrape(artists_names[759:])
# jędker realista 95
