import sys 
sys.path.append("..")
from notebooks.config import GENIUS_BEARER as token
from Analysis_imports import scrape_artist_songs
import lyricsgenius
import pickle
genius = lyricsgenius.Genius(token, verbose=True)

file_name = '../scraped_data/LastFm/all_no_features_LastFM.txt'
artists = open(file_name).read().split('\n')
artists = [a.strip() for a in artists]

stop = len(artists)
start = 3333
scrape_artist_songs(artists[start-1:stop], verbose=True, artist_id=start)