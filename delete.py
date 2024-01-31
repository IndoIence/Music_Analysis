# %%
import pickle
from tqdm import tqdm
import os
from pathlib import Path
import logging
from classes.MyArtist import MyArtist
from utils import CONFIG, get_all_urls, get_artists, get_artist
from Analysis_imports import sanitize_art_name, save_artist_to_pkl
import lyricsgenius

config_G = CONFIG["Genius_scraping"]
token = config_G["GENIUS_BEARER"]


# %%
art_file = 'Malik_Montana.artPkl'
art_path = Path(CONFIG["generated_data_path"])
art = get_artist(art_file, art_path)
words = art.get_40k_words()
print(words)
# %%
