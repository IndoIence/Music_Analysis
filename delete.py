# %%
import pickle
from tqdm import tqdm
import os
from pathlib import Path
import logging
from utils import CONFIG, get_all_urls
from Analysis_imports import scrape_artists_songs
config_G = CONFIG["Genius_scraping"]
logging.basicConfig(
    filename=config_G["GENIUS_LOGFILE"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# %%
scrape_artists_songs(["Tau"])

# %%
