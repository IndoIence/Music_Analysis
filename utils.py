# %%
from typing import List, Tuple
from pathlib import Path
from classes.MyArtist import MyArtist
import json
import re
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
        art = get_artist(art_path, path)
        yield art

def get_all_urls():
    genius_path = Path(CONFIG["Genius_scraping"]["save_path"])
    urls = []

    for art_path in (f for f in genius_path.iterdir() if f.is_file()):
        with open(art_path, 'rb+') as f:
            art = pickle.load(f)
            urls.append(art.url)
    return sorted(set(urls))

def get_100_biggest():
    to_sort = list(get_artists())
    yield from sorted(to_sort, key=lambda x: x.lyrics_len, reverse=True)[:100][::-1]


def all_artists_genius(genius_path:Path=Path(CONFIG["Genius_scraping"]["save_path"])):
    f_names = os.listdir(genius_path)
    for art_file in f_names:
        artist_path = genius_path / art_file
        a = MyArtist(pickle.load(open(artist_path, 'rb')))
        yield a     

def load_jsonl(p:Path):
    with open(p, 'r') as f:
        while line := f.readline():
            yield json.loads(line)

# for legacy reasons (i am retarded)this is outside this should be in the save_artist_to_pkl
def sanitize_art_name(name:str)-> str:
    return  name.replace(' ', '_').replace(".", "_").replace('/', ' ').replace('?', ' ').strip()
def save_artist_to_file(out_path: Path, artist, artist_name: str, suffix:str = '.artPkl'):
    # before saving check if there are any / or ? in the name -> if so replace them
        if '/' in artist_name or '?' in artist_name:
            artist_name = artist_name.replace('/', ' ').replace('?', ' ')
            # logging.info(
            #     f'Changed name of the artist to {artist_name} because of the slashes/question marks in the name')
        with open(out_path / (artist_name + suffix), 'wb') as f:
            pickle.dump(artist, f)

def get_wsd_data():
    """for all artists return a tuple (file name, list) 
    where list has wsd for each song"""
    path = Path(CONFIG['wsd_outputs'])
    wsd_files = [f for f in os.listdir(path) if os.path.isfile(path / f)]
    for wsd_file in wsd_files:
        result = []
        for song in load_jsonl(path / wsd_file):
            result.append(song['wsd'])
        yield (wsd_file, result)

def get_stopwords():
    path = CONFIG['stopwords']
    return [line.rstrip() for line in open(path)]
        