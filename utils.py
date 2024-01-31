# %%
from typing import List, Tuple
from collections import Counter
from pathlib import Path
from lyricsgenius.genius import Artist, Song
from classes.MyArtist import MyArtist, MySong
from nltk import word_tokenize

import re
import spacy
import pickle
import os
import yaml

config_path = Path(__file__).parent / "config.yml" 
CONFIG = yaml.safe_load(open(config_path, 'r'))
# %%

def get_artist(name: str, path: Path = Path(CONFIG["artists_pl_path"])):
    return pickle.load(open(path / name, 'rb'))

def get_artists(path: Path = Path(CONFIG["artists_pl_path"]) ):
    art_paths = [f for f in os.listdir(path) if os.path.isfile(path / f)]
    for art_path in art_paths:
        art: Artist = get_artist(art_path, path)
        return art

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


def all_artists_genius(genius_path:Path=Path(CONFIG["Genius_scraping"]["save_path"])):
    f_names = os.listdir(genius_path)
    for art_file in f_names:
        artist_path = genius_path / art_file
        a = MyArtist(pickle.load(open(artist_path, 'rb')))
        yield a     


           


