from lyricsgenius.genius import Song
import re


class MySong(Song):
    _nlp = None

    @staticmethod
    def _init_nlp(model_name: str = "pl_core_news_sm", language: bool = True):
        # is it ok to lazy import like that?
        import spacy
        import spacy_fastlang

        if MySong._nlp is None:
            nlp = spacy.load(model_name)
            MySong._nlp = nlp
        if language and "language_detector" not in MySong._nlp.pipe_names:
            MySong._nlp.add_pipe("language_detector")
        return MySong._nlp

    def __init__(self, song: Song, art_name: str = ""):
        self.__dict__ = song.__dict__
        self.language: str = self._get_language()
        self.word_count: int = self._get_word_count()
        self._sentiment: dict = {}
        self._emotions: dict = {}
        self.artist_name = art_name

    @property
    def sentiment(self):
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value):
        self._sentiment = value

    @property
    def emotions(self):
        return self._emotions

    @emotions.setter
    def emotions(self, value):
        self._emotions = value

    @property
    def date(self):
        return self._body.get(
            "release_date_components", {"year": None, "month": None, "day": None}
        )

    def get_clean_song_lyrics(self, lower=True, linebreaks=False):
        # find the first occurance of the word "Lyrics", and discard what's before that
        lyrics_start = self.lyrics.find("Lyrics") + len("Lyrics")
        lyrics_cleaned = self.lyrics[lyrics_start:]
        if int(self._body["primary_artist"]["id"]) == 114783:
            # really wierd edge case with Taco Hemingway
            # every song has wierd line
            # "See Taco Hemingway LiveGet tickets as low as $75You might also like"
            # remove it
            lyrics_cleaned = lyrics_cleaned.replace(
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
        pattern = r"\[.*?\]"
        lyrics_cleaned = re.sub(pattern, "", lyrics_cleaned)
        lyrics_cleaned = lyrics_cleaned.replace("\u200b", "").replace("\u200c", "")
        lyrics_cleaned = lyrics_cleaned.strip()
        return lyrics_cleaned

    def _get_language(self) -> str:
        nlp = self._init_nlp()
        doc = nlp(self.get_clean_song_lyrics())
        lang = doc._.language if doc._.language_score > 0.8 else "xx"
        return lang

    def _get_word_count(self) -> int:
        return len(re.findall(r"\w+", self.get_clean_song_lyrics()))
