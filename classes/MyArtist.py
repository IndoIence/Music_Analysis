from lyricsgenius.genius import Artist, Song
from classes.MySong import MySong
from pathlib import Path
from nltk import word_tokenize
from copy import deepcopy
from pickle import dump

# i don't know if scpacy_fastlang is actually needed


class MyArtist(Artist):
    @staticmethod
    def _init_nlp():
        # is it ok to lazy import like that?
        import spacy
        import spacy_fastlang

        nlp = spacy.load("pl_core_news_lg")
        nlp.add_pipe("language_detector")
        return nlp

    def __init__(self, artist: Artist):
        self.__dict__ = artist.__dict__
        self.songs = self._get_my_songs()
        self.songs_languages = self.get_songs_languages()
        self.language: str = self.songs_languages[0][0]
        self._lyrics_len_only_art: int = self.lyrics_len_only_art
        self._lyrics_len_all: int = self.lyrics_len_all
        self._lyrics_len_prim_art: int = self.lyrics_len_prim_art
        # self.lyrics_len: int = self.lyrics_len
        # self.nlp_doc = sel    out = ''f._get_nlp_docs(nlp)

    @property
    def lyrics_len_only_art(self):
        if not hasattr(self, "_lyrics_len_only_art"):
            self._lyrics_len_only_art = self.get_lyrics_len(only_art=True)
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
        return (
            self.name.replace(" ", "_")
            .replace(".", "_")
            .replace("/", " ")
            .replace("?", " ")
            .replace("\u200b", "")
            .replace("\u200c", "")
            .strip()
        )

    def get_songs_languages(self) -> list[tuple[str, int]]:
        nlp = self._init_nlp()
        if not self.songs:
            return [("xx", -1)]
        counter: dict[str, int] = {}
        for song in self.songs:
            lang = nlp(song.lyrics)._.language
            song.language = lang
            if lang in counter:
                counter[lang] += 1
            else:
                counter[lang] = 1
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # TODO: Is include features even necessary? Isn't it the same as only_art?
    def get_lyrics_len(self, include_features=True, only_art=False) -> int:
        if not self.songs:
            return 0
        count = 0
        for song in self.songs:
            # if the song primary artists is the current artist then include it
            if song.language != "pl":
                continue
            if only_art and song._body["featured_artists"]:
                continue
            if not include_features and song._body["primary_artist"]["id"] != self.id:
                continue
            count += self.count_words(song.get_clean_song_lyrics())
        return count

    def _get_my_songs(self) -> list[MySong]:
        if not hasattr(self, "songs") or not self.songs:
            return []
        output = []
        for song in self.songs:
            if isinstance(song, MySong):
                output.append(song)
            elif isinstance(song, Song):
                output.append(MySong(song))
        return output

    def get_limit_songs(
        # i don't know how to achieve a infinite int with type hints
        self,
        limit: int | float = float("inf"),
        prim_art: bool = False,
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
            words = word_tokenize(song.get_clean_song_lyrics())
            count += len(words)
            result.append(song)
            if count >= limit:
                diff = int(count - limit)
                if strict and diff > 0:
                    words = words[:-diff]
                    new_song = deepcopy(song)
                    new_song.lyrics = " ".join(words)
                break
        return result

    # this should rather be in the song i think
    def count_words(self, text: str) -> int:
        return len(word_tokenize(text))

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
        if file_name is None:
            file_name = self.name_sanitized
        # before saving check if there are any / or ? in the name -> if so replace them
        if "/" in file_name or "?" in file_name:
            file_name = file_name.replace("/", " ").replace("?", " ")
        with open(out_path / (file_name + suffix), "wb") as f:
            dump(self, f)
