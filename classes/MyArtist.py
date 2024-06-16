from lyricsgenius.genius import Artist, Song
from classes.MySong import MySong
from pathlib import Path
from copy import deepcopy
from pickle import dump


class MyArtist(Artist):

    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.language: str = self._get_language()
        self._lyrics_len_only_art: int = self.get_lyrics_len(
            only_art=True, include_features=False
        )
        self._lyrics_len_all: int = self.get_lyrics_len(include_features=True)
        self._lyrics_len_prim_art: int = self.get_lyrics_len(include_features=False)
        self.translation_table: dict[int, str] = {
            ord("."): "_",
            ord(" "): "_",
            ord("/"): "",
            ord("?"): "",
            ord("\u200c"): "",
            ord("\u200b"): "",
            ord("("): "",
            ord(")"): "",
        }
        # self.lyrics_len: int = self.lyrics_len
        # self.nlp_doc = sel    out = ''f._get_nlp_docs(nlp)

    @property
    def lyrics_len_only_art(self):
        if not hasattr(self, "_lyrics_len_only_art"):
            self._lyrics_len_only_art = self.get_lyrics_len(
                only_art=True, include_features=False
            )
        return self._lyrics_len_only_art

    @property
    def lyrics_len_all(self):
        if not hasattr(self, "_lyrics_len_all"):
            self._lyrics_len_all = self.get_lyrics_len(include_features=True)
        return self._lyrics_len_all

    @property
    def lyrics_len_prim_art(self):
        if not hasattr(self, "_lyrics_len_prim_art"):
            self._lyrics_len_prim_art = self.get_lyrics_len(include_features=False)
        return self._lyrics_len_prim_art

    # i have this property also in the utils but i want to keep the classes self contained
    @property
    def name_sanitized(self) -> str:
        return self.name.translate(self.translation_table).strip()

    # TODO: Is include features even necessary? Isn't it the same as only_art?
    def get_lyrics_len(self, include_features=True, only_art=False) -> int:
        if not self.songs:
            return 0
        count = 0
        for song in self.songs:
            # if the song primary artists is the current artist then include it
            if song.language != self.language:
                continue
            if only_art and song._body["featured_artists"]:
                continue
            if not include_features and song._body["primary_artist"]["id"] != self.id:
                continue
            count += song.word_count
        return count

    def _get_my_songs(self) -> list[MySong]:
        if not hasattr(self, "songs") or not self.songs:
            return []
        output = []
        for song in self.songs:
            if isinstance(song, Song) or isinstance(song, MySong):
                output.append(MySong(song))
            else:
                raise ValueError("song is not a Song or MySong")
        return output

    def _get_language(self) -> str:
        if not self.songs:
            return "xx"
        counter: dict[str, int] = {}
        for song in self.songs:
            counter[song.language] = counter.get(song.language, 0) + 1
        if not counter:
            return "xx"
        # I don't like this as this doesn't inform me if multiple languages have the same count
        return max(counter.items(), key=lambda x: x[1])[0]

    def get_limit_songs(
        # i don't know how to achieve a infinite int with type hints
        self,
        limit: int | float = float("inf"),
        prim_art: bool = True,
        only_art: bool = False,
        strict: bool = False,
    ) -> list[MySong]:
        """
        returns list of songs with the limit of words
        if strict == True the output can be none
        otherwise finishes the song just over the given limit
        the words are counted ater cleaning the song text from genius bullshit
        """
        if strict and self.lyrics_len_only_art < limit:
            return []
        count = 0
        result = []
        for song in self.songs:
            # skip non-polish songs
            if song.language != "pl":
                continue
            if only_art and song._body["featured_artists"]:
                continue
            if prim_art and song._body["primary_artist"]["id"] != self.id:
                continue
            word_count = song.word_count
            count += word_count
            result.append(song)
            if count >= limit:
                diff = int(count - limit)
                if strict and diff > 0:
                    # get just the right amount of words from the last song
                    words = " ".join(
                        song.get_clean_song_lyrics(
                            lower=False, linebreaks=True
                        ).split()[:-diff]
                    )
                    new_song = deepcopy(song)
                    new_song.lyrics = " ".join(words)
                break
        return result

    def save_lyrics(self, save_path: Path = Path(""), filename="", extension="txt"):  # type: ignore
        """
        Overwrites the lyricsgenius.artist.save_lyrics().
        Gives sanitized lyrics without brackets and genius unwanted additions
        """
        if not filename:
            filename = self.name_sanitized
        filename += "." + extension
        with open(save_path / filename, "w") as f:
            for i, song in enumerate(self.songs):
                f.write(str(i + 1) + ". " + song.title)
                f.write("\n\n")
                f.write(song.get_clean_song_lyrics())
                f.write("\n\n")

    def to_pickle(
        self,
        out_path: Path,
        file_name: str = "",
        suffix: str = ".artPkl",
    ):
        if file_name == "":
            file_name = self.name_sanitized
        # before saving check if there are any / or ? in the name -> if so replace them
        if "/" in file_name or "?" in file_name:
            file_name = file_name.replace("/", " ").replace("?", " ")
        with open(out_path / (file_name + suffix), "wb") as f:
            dump(self, f)

    def __str__(self):
        """Return a string representation of the Artist object."""
        msg = f"{self.name}, {self.num_songs} songs, {self.lyrics_len_only_art} words no features {self.lyrics_len_all} all words"
        msg = msg[:-1] if self.num_songs == 1 else msg
        return msg
