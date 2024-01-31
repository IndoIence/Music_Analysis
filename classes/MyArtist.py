from lyricsgenius.genius import Artist, Song
from classes.MySong import MySong
import spacy
from nltk import word_tokenize
# i don't know if scpacy_fastlang is actually needed
import spacy_fastlang

class MyArtist(Artist):
    @staticmethod
    def _init_nlp():
        nlp = spacy.load("pl_core_news_lg")
        nlp.add_pipe('language_detector')
        return nlp    

    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.songs_languages = self.get_songs_languages()
        self.language: str = self.songs_languages[0][0]
        self.lyrics_len: int = self.get_lyrics_len()
        # self.nlp_doc = self._get_nlp_docs(nlp)

    def get_songs_languages(self):
        nlp = self._init_nlp()
        if not self.songs:
            return [('xx', -1)]
        counter = {}
        for song in self.songs:
            lang = nlp(song.lyrics)._.language
            song.language = lang
            if lang in counter:
                counter[lang] += 1
            else:
                counter[lang] = 1
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    def get_lyrics_len(self, include_features=False):
        if not self.songs:
            return 0
        count = 0
        for song in self.songs:
            # if the song primary artists is the current artist then include it
            if include_features or song.primary_artist.name == self.name:
                count += count_words(song.lyrics)
        return count


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
    
    def get_40k_words(self, limit=40000) -> str:
        result = ''
        count = 0
        for song in self.songs:
            song_cleaned = song.clean_song_text()
            words = word_tokenize(song_cleaned)
            count += len(words)
            if count < limit:
                result += song_cleaned
                continue
            diff = limit - count
            return result + " ".join(words[:diff])

        raise ValueError(f"Artist {self.name} doesn't have {limit} words in avialable songs")

def count_words(text: str):
    return len(word_tokenize(text))