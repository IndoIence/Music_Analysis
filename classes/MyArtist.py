from lyricsgenius.genius import Artist, Song
from classes.MySong import MySong
import spacy
from nltk import word_tokenize
class MyArtist(Artist):
    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.songs_languages = self.get_songs_languages()
        self.language: str = self.songs_languages[0][0]
        self.lyrics_len: int = self.get_lyrics_len()
        # self.nlp_doc = self._get_nlp_docs(nlp)

    def get_songs_languages(self):
        nlp = spacy.load("pl_core_news_lg")
        nlp.add_pipe('language_detector')
        if not self.songs():
            return [('xx', -1)]
        counter = {}
        for song in self.songs:
            lang = nlp(song.lyrics)._.language
            if lang in counter:
                counter[lang] += 1
            else:
                counter[lang] = 1
        return sorted(counter.items(), key=lambda x: x[1])
    
    def get_lyrics_len(self):
        if not self.songs:
            return 0
        count = 0
        for song in self.songs:
            count += self.count_words(song.lyrics)
        self.lyrics_len = count


    def _get_my_songs(self)-> list[MySong]:
        if not hasattr(self, 'songs') or not self.songs:
            return []
        output = []
        for song in self.songs:
            if isinstance(song, MySong):
                output.append(song)
            elif isinstance(song, Song):
                output.append(MySong(song))
        return output
    
    def count_words(text: str):
        return len(word_tokenize(text))