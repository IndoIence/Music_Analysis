from lyricsgenius.types import Artist
from classes.mySong import MySong
from pathlib import Path
from pickle import dump
from copy import copy

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
        self.genius_name = self.name
        # better name for saving files etc.
        self.name = sanitize_name(self.name)

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

    def save_raw_lyrics(self, folder: Path, extension: str = ".txt", no_features=False):
        songs = self.songs_no_features() if no_features else self.songs
        f_name = f"{sanitize_name(self.name)}{extension}"
        with open(folder / f_name, "w") as f:
            for i, song in enumerate(songs):
                text = song.clean_lyrics(remove_section_headers=True)
                f.write(f"\n\n{i}. {song.title}")
                f.write(text)

    def get_limit_songs(
        self,
        limit: int,
        no_features=True,
        filter_out_phrases: list[str] = [],
        strict=True,
        artist_language=True,
    ) -> list[MySong]:
        # return MySong list with sum of word_count limit
        # if no_features is True, only songs where the artist is the main artist are returned
        # if filter_out_phrases is not empty, only songs that do not contain any of the phrases are returned
        # if strict is True, list is returned with the exact limit of words
        # takes into account the songs in the language of the artist
        songs = self.songs_no_features() if no_features else self.songs.copy()
        if artist_language:
            songs = [song for song in songs if song.language == self.language]
        cur_count = 0
        result_songs = []
        # loop through songs and add them to the result_songs list if the limit is not exceeded
        for song in songs:
            result_songs.append(song)
            cur_count += song.lyrics_length
            if cur_count > limit:
                if strict:
                    diff = cur_count - limit
                    new_song = copy(song)
                    # this removes newlines which is not good but whatever
                    shorter_lyrics = song.clean_lyrics().split()[:-diff]
                    new_song.lyrics = " ".join(shorter_lyrics)
                    new_song.lyrics_length = diff
                    result_songs.pop()
                    result_songs.append(new_song)
                    cur_count += diff
                break

        return result_songs if cur_count >= limit else []

    def songs_no_features(self) -> list[MySong]:
        return [
            song
            for song in self.songs
            if self.id == song._body["primary_artist"]["id"]
            and song._body["featured_artists"] == []
        ]

    def get_vocab(self, limit_words=0) -> list[tuple[str, int]]:
        songs = self.get_limit_songs(limit_words)
        # spacy tokenize, then count count the lemmas

        vocab = {}
        for song in songs:
            doc = song.get_spacy()(song.clean_lyrics())
            tokens = [token for token in doc if self.valid_token(token)]
            for token in tokens:
                # debug
                if token.lemma_ == "Might":
                    print(token.text)
                vocab[token.lemma_] = vocab.get(token.lemma_, 0) + 1
        return sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    def valid_token(self, token) -> bool:
        return (
            token.is_alpha
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        )


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", " ")
        .replace("?", " ")
        .replace("\u200b", "")
        .replace("\n", "")
        .replace(" ", "_")
        .strip()
    )
