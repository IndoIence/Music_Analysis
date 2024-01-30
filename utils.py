# %%
from typing import List, Tuple
from collections import Counter
from pathlib import Path
from lyricsgenius.genius import Artist, Song
from classes.MyArtist import MyArtist
import spacy_fastlang
import spacy
import pickle
import os
import yaml

config_path = Path(__file__).parent / "config.yml" 
CONFIG = yaml.safe_load(open(config_path, 'r'))
# %%


def get_artists(path: Path = Path(CONFIG["artists_pl_path"]) ):
    art_paths = [f for f in os.listdir(path) if os.path.isfile(f)]
    for art_path in art_paths:
        art: Artist = pickle.load(open(path / art_path))
        yield art

def get_all_lyrics(art: Artist):
    if not art.songs:
        yield ''
    yield '\n'.join((song.lyrics for song in art.songs))

def get_all_urls():
    genius_path = Path(CONFIG["Genius_scraping"]["save_path"])
    urls = []

    for art_path in (f for f in genius_path.iterdir() if f.is_file()):
        with open(art_path, 'rb+') as f:
            art = pickle.load(f)
            urls.append(art.url)
    return sorted(set(urls))


def load_all_artist(genius_path: Path):
    f_names = os.listdir(genius_path)
    for art_file in f_names:
        artist_path = genius_path / art_file
        a = MyArtist(pickle.load(open(artist_path, 'rb')))
        yield a


       

        