from lyricsgenius.genius import Song
import re


class MySong(Song):
    @staticmethod
    def _init_nlp():
        # is it ok to lazy import like that?
        import spacy
        import spacy_fastlang

        nlp = spacy.load("pl_core_news_lg")
        nlp.add_pipe("language_detector")
        return nlp

    def __init__(self, song: Song):
        self.__dict__ = song.__dict__
        self.nlp = None
        self.language = None

    def get_clean_song_lyrics(self, lower=True, linebreaks=False):
        # find the first occurance of the word "Lyrics", and discard what's before that
        lyrics_start = self.lyrics.find("Lyrics") + len("Lyrics")
        lyrics_cleaned = self.lyrics[lyrics_start:]
        if int(self._body["primary_artist"]["id"]) == 114783:
            # really wierd edge case with Taco Hemingway
            # every song has wierd line
            # "See Taco Hemingway LiveGet tickets as low as $75You might also like"
            # remove it
            lyrics_cleaned.replace(
                "See Taco Hemingway LiveGet tickets as low as $75You might also like",
                "",
            )
        # cut out the end of the string (the word Embed and the number or just the word Embed)
        # search for the number on the end and if it exists cut out from it
        if "Embed" == lyrics_cleaned[-5:]:
            lyrics_cleaned = lyrics_cleaned[:-5]
            pattern = r"\s*\d*$"
            lyrics_cleaned = re.sub(pattern, "", lyrics_cleaned)
        # clean english contaminated phrases from genius
        lyrics_cleaned = re.sub(r"You might also like", "", lyrics_cleaned)
        if lower:
            lyrics_cleaned = lyrics_cleaned.lower()
        if not linebreaks:
            lyrics_cleaned = lyrics_cleaned.replace("\n", " ")

        # should ignore anything in the square brackets
        # usualy genius has indication of an artists singing there
        lyrics_cleaned = clean_square_brackets(lyrics_cleaned)
        lyrics_cleaned = lyrics_cleaned.replace("\u200b", "").replace("\u200c", "")
        lyrics_cleaned = lyrics_cleaned.strip()
        return lyrics_cleaned

    def get_language(self) -> list[tuple[str, int]]:
        self.nlp = self._init_nlp() if self.nlp is None else self.nlp
        lang = self.nlp(self.get_clean_song_lyrics())._.language
        return lang


def clean_square_brackets(text):
    pattern = r"\[.*?\]"
    return re.sub(pattern, "", text)
