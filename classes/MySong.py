from lyricsgenius.genius import Song
import re


class MySong(Song):
    def __init__(self, song: Song):
        self.__dict__ = song.__dict__
        self.language = None

    def get_clean_song_lyrics(self, lower=True, linebreaks=False):
        # find the first occurance of the word "Lyrics", and discard what's before that
        lyrics_start = self.lyrics.find("Lyrics") + len("Lyrics")
        lyrics_cleaned = self.lyrics[lyrics_start:]
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


def clean_square_brackets(text):
    pattern = r"\[.*?\]"
    return re.sub(pattern, "", text)
