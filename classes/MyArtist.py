from lyricsgenius.genius import Artist, Song
from classes.MySong import MySong
from pathlib import Path
from nltk import word_tokenize
# i don't know if scpacy_fastlang is actually needed


class MyArtist(Artist):
    @staticmethod
    def _init_nlp():
        # is it ok to lazy import like that?
        import spacy
        import spacy_fastlang
        nlp = spacy.load("pl_core_news_lg")
        nlp.add_pipe('language_detector')
        return nlp    

    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.songs_languages = self.get_songs_languages()
        self.language: str = self.songs_languages[0][0]
        self.lyrics_len: int = self.get_lyrics_len()
        # self.nlp_doc = sel    out = ''f._get_nlp_docs(nlp)
    @property
    def name_sanitized(self) -> str:
        return  self.name.replace(' ', '_').replace(".", "_").replace('/', ' ').replace('?', ' ').strip()

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
                count += self.count_words(song.lyrics)
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
    
    def get_limit_songs(self, limit:int=None, prim_art: bool=False, only_art:bool=False) -> list[Song]:
        # returns list of songs just over the limit of words
        # the words are counted ater cleaning
        if not limit:
            limit = float('inf')
        count = 0
        result = []
        for song in self.songs:
            if song.language != 'pl':
                continue
            if only_art and song._body["featured_artists"]:
                continue
            if prim_art and not song._body["primary_artist"]["id"] == self.id:
                continue 
            words = word_tokenize(song.clean_song_lyrics)
            count += len(words)
            result.append(song)
            if count >= limit:
                return result
        return result
    #this should rather be in the song i think
    def count_words(self, text: str) -> int:
        return len(word_tokenize(text))
    
    def save_lyrics(self, save_path: Path = Path(''), filename=None, extension='txt'):
            """
            Overwrites the lyricsgenius.artist.save_lyrics().
            Gives sanitized lyrics without brackets and genius unwanted additions
            """
            if not filename:
                filename = self.name_sanitized
            filename += '.' + extension
            with open(save_path / filename, 'w') as f:
                for i,song in enumerate(self.songs):
                    f.write(str(i+1)+'. '+song.title)
                    f.write('\n\n')
                    f.write(song.clean_song_lyrics)
                    f.write('\n\n')