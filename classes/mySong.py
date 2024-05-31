from lyricsgenius.types import Song
import spacy
import spacy_fastlang
import re


class MySong(Song):
    _nlp = None

    def __init__(self, old_song: Song):
        self.__dict__ = old_song.__dict__.copy()
        self.language = self.get_language()
        self.lyrics_length = self.get_lyrics_length()

    @classmethod
    def from_song(cls, old_song: Song):
        my_song = cls.__new__(cls)
        my_song.__dict__ = old_song.__dict__.copy()
        my_song.language = my_song.get_language()
        return my_song

    @classmethod
    def get_spacy(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load("pl_core_news_sm")
            cls._nlp.add_pipe("language_detector")
        return cls._nlp

    def get_language(self) -> str:
        nlp = self.get_spacy()
        doc = nlp(self.lyrics)
        language = doc._.language if doc._.language_score > 0.8 else "unknown"
        return language

    def get_lyrics_length(self) -> int:
        return len(self.clean_lyrics().split())

    def clean_lyrics(
        self, remove_section_headers=True, remove_newlines=True, lower=True
    ):
        lyrics = self.lyrics.strip()
        if remove_section_headers:
            lyrics = re.sub(r"(\[.*?\])*", "", lyrics)
            lyrics = re.sub("\n{2}", "\n", lyrics)
        # remove a numeral and Embed at the end of the string
        pattern = r"(\d*Embed\s*)$"
        lyrics = re.sub(pattern, "", lyrics)
        if lower:
            lyrics = lyrics.lower()

        return lyrics
