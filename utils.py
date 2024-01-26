from typing import List, Tuple
from collections import Counter
from pathlib import Path
from lyricsgenius.genius import Artist, Song
import spacy_fastlang
import spacy
import pickle
import os
pl_art_path = Path("scraped_data/artists_pl")
def get_artists(path: Path = pl_art_path ):
    art_paths = [f for f in os.listdir(path) if os.path.isfile(f)]
    for art_path in art_paths:
        art: Artist = pickle.load(open(path / art_path))
        yield art

def get_all_lyrics(art: Artist):
    if not art.songs:
        yield ''
    yield '\n'.join((song.lyrics for song in art.songs))


def load_all_artist(genius_path: Path):
    f_names = os.listdir(genius_path)
    for art_file in f_names:
        artist_path = genius_path / art_file
        a = MyArtist(pickle.load(open(artist_path, 'rb')))
        yield a

class MySong(Song):
    def __init__(self, song:Song):
        self.__dict__ = song.__dict__
        self.language = None
       

class MyArtist(Artist):
    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.language: str = 'xx'
        self.lyrics_len: int = 0
        # self.songs_languages: List[Tuple[str, int]] = Counter([song.language for song in self.songs]).most_common()
        # self.nlp_doc = self._get_nlp_docs(nlp)
    
    def _get_nlp_docs(self, nlp: spacy.language.Language):
        if not hasattr(self, 'songs') or not self.songs:
            return []
        else:
            return [nlp(song.lyrics) for song in self.songs if isinstance(song, Song)]

    def _get_my_songs(self)-> List[MySong]:
        if not hasattr(self, 'songs') or not self.songs:
            return []
        output = []
        for song in self.songs:
            if isinstance(song, MySong):
                output.append(song)
            elif isinstance(song, Song):
                output.append(MySong(song))
        return output
        
    
    def is_polish_artist(self):
        if self.songs_languages:
            return self.songs_languages[0][0] == 'pl'
        else:
            return None
        
def get_artist_language(artist: Artist, ) -> str:
    nlp = spacy.load("pl_core_news_lg")
    nlp.add_pipe('language_detector')
    if not artist.songs():
        return 'xx'
    counter = {}
    for song in artist.songs:
        lang = nlp(song.lyrics)._.language
        if lang in counter:
            counter[lang] += 1
        else:
            counter[lang] = 1
    return max(counter, key=counter.get)