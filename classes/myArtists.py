from lyricsgenius.types import Artist
from classes.mySong import MySong
from pathlib import Path
from pickle import dump

# MyArtist class is a wrapper for the Artist class
# It adds functionality to the Artist class
# It is used to save the Artist object to a artPkl file
# TODO: information about albums
# TODO: updating existing artists with new attributes
# #     def update_instance(self):
#         # Add any new attributes that were added after instances were pickled
#         if not hasattr(self, 'new_attribute'):
#             self.new_attribute = "default_value"
#         # Add more attributes as needed


class MyArtist(Artist):
    def __init__(self, old_art: Artist):
        self.__dict__ = old_art.__dict__.copy()
        self.songs: list[MySong] = [MySong(song) for song in old_art.songs]
        self.language = self.get_language()
        self.len_lyrics_all = self.get_lyrics_count()
        self.len_lyrics = self.get_lyrics_count(no_features=True)

    # @classmethod
    # def from_artist(cls, old_art: Artist):
    #     my_artist = cls.__new__(cls)
    #     my_artist.__dict__ = old_art.__dict__.copy()
    #     # my_artist.songs: list[MySong] = []
    #     if hasattr(old_art, "songs"):
    #         my_artist.songs = [MySong(song) for song in old_art.songs]
    #     my_artist.language = my_artist.get_language()
    #     return my_artist

    def get_language(self) -> str:
        counter = {}
        if not self.songs:
            return "unknown"
        for song in self.songs:
            counter[song.language] = counter.get(song.language, 0) + 1
        return max(counter, key=counter.get)

    def get_limit_songs(self, limit: int) -> list[MySong]:
        return self.songs[:limit]

    def to_pickle(self, folder: Path, suffix: str = ".artPkl"):
        artist_name = sanitize_name(self.name)
        artist_name += suffix
        with open(folder / artist_name, "wb") as f:
            dump(self, f)

    def get_lyrics_count(self, no_features=False) -> int:
        songs = [song for song in self.songs]
        # songs = self.songs -> why does this not work?
        if no_features:
            songs = [
                song
                for song in self.songs
                if self.id == song._body["primary_artist"]["id"]
                and song._body["featured_artists"] == []
                and song.language == self.language
            ]
        return sum([song.lyrics_length for song in songs])

    def get_albums(self) -> list[dict]:
        albums = []
        for song in self.songs:
            if (
                "album" in song._body
                and bool(song._body["album"])
                and song._body["album"] not in albums
            ):
                albums.append(song._body["album"])
        return albums


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", " ")
        .replace("?", " ")
        .replace("\u200b", "")
        .replace("\n", "")
        .replace(" ", "_")
        .strip()
    )
