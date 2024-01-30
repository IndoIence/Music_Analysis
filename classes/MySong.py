from lyricsgenius.genius import Song

class MySong(Song):
    def __init__(self, song:Song):
        self.__dict__ = song.__dict__
        self.language = None