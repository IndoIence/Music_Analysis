from classes.scraper import GeniusScraper
from utils import CONFIG

artists_names = open(CONFIG['to_scrape_file'], 'r').readlines()
scraper = GeniusScraper()
scraper.scrape(artists_names)