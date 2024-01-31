from lyricsgenius.genius import Song
import re
class MySong(Song):
    def __init__(self, song:Song):
        self.__dict__ = song.__dict__
        self.language = None

    def clean_song_text(self):
        # find the first occurance of the word "Lyrics", and discard what's before that
        lyrics_start = self.lyrics.find('Lyrics') + len('Lyrics')
        lyrics_cleaned = self.lyrics[lyrics_start:].lower()
        # cut out the end of the string (the word Embed and the number)
        # search for the number on the end and if it exists cut out from it
        if re.search(r'\d+', lyrics_cleaned[::-1]):
            lyrics_end = re.search(r'\d+', lyrics_cleaned[::-1]).span()[1]
        else:
            lyrics_end = 1
        lyrics_cleaned = lyrics_cleaned[:-lyrics_end]
        # should ignore anything in the square brackets
        lyrics_cleaned = clean_brackets(lyrics_cleaned)
        return lyrics_cleaned

def clean_brackets(text):
    pattern = r'\[.*?\]'
    return re.sub(pattern, '', text)