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
    def get_spacy(cls, model_name: str = "pl_core_news_sm", language: bool = True):
        if cls._nlp is None:
            cls._nlp = spacy.load(model_name)
        if language:
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
        if int(self._body["primary_artist"]["id"]) == 114783:
            # really wierd edge case with Taco Hemingway
            # every song has wierd line
            # "See Taco Hemingway LiveGet tickets as low as $75You might also like"
            # remove it
            lyrics = self.lyrics.replace(
                "See Taco Hemingway LiveGet tickets as low as $75You might also like",
                "",
            )
        # Edge case for everyone
        # solving it like that does not seem right but i don't have time
        lyrics = lyrics.replace("You might also like", "")
        if remove_section_headers:
            lyrics = re.sub(r"(\[.*?\])*", "", lyrics)
            lyrics = re.sub("\n{2}", "\n", lyrics)
        # only keep that part of the lyrics that is after the word: "Lyrics"
        # if it is in the first 200 characters
        # then remove everything before the first occurence of it in the string
        pattern = r"Lyrics"
        if pattern in lyrics[:200]:
            lyrics = lyrics[lyrics.find(pattern) + len(pattern) :]
        # remove a numeral and Embed at the end of the string
        pattern = r"(\d*Embed\s*)$"
        lyrics = re.sub(pattern, "", lyrics)
        if lower:
            lyrics = lyrics.lower()

        return lyrics
