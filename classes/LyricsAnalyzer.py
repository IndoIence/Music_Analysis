from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_biggest_by_lyrics_len, get_all_artists, get_artist, CONFIG
from tqdm import tqdm
import lyricsgenius
import spacy
import os
from classes.MyArtist import MyArtist
import heapq
from typing import Generator, Any
from collections import Counter

nlp = spacy.load("pl_core_news_sm")


def is_valid(token):
    return not (
        token.is_punct
        or token.is_stop
        or token.is_bracket
        or token.pos_ == "SPACE"
        or token.is_digit
        or token.like_num
    )


# TODO: load kurwa counters from file if they exist
class LyricsAnalyzer:
    def __init__(
        self,
        artists: list[MyArtist] = [],
        word_limit=30000,
    ) -> None:
        self.word_limit = word_limit
        self.artists = get_biggest_by_lyrics_len(30) if not artists else artists
        self.counters = self.get_counters(
            word_limit=self.word_limit, artists=self.artists
        )

    def get_counters(
        self, word_limit: int, artists: list[MyArtist]
    ) -> dict[str, Counter]:
        arts = tqdm(artists)
        counters: dict[str, Counter] = {}
        for art in arts:
            songs = art.get_limit_songs(word_limit, strict=True, only_art=True)
            if not songs:
                continue
            words = "\n".join(song.get_clean_song_lyrics(lower=True) for song in songs)
            doc = nlp(words)
            c = Counter((token.lemma_, token.pos_) for token in doc if is_valid(token))
            arts.set_description(f"{art}, words_count: {len(c)}")
            counters[art.name_sanitized] = c
        return counters

    def save_counters(self) -> None:
        # save counter lengths
        vocab_path = Path(CONFIG["vocab_path"])
        with open(vocab_path / f"{self.word_limit}.csv", "w") as f:
            f.write("Artist, Vocab_size\n")
            for name, c in sorted(
                self.counters.items(), key=lambda x: len(x[1]), reverse=True
            ):
                f.write(name + ", " + str(len(c)) + "\n")
        # save counters to a file
        if not (vocab_path / str(self.word_limit)).exists():
            (vocab_path / str(self.word_limit)).mkdir()
        # remove all files in the directory
        for f in (vocab_path / str(self.word_limit)).iterdir():
            if f.is_file():
                f.unlink()
        for art_name, c in self.counters.items():
            out_path = vocab_path / str(self.word_limit) / f"{art_name}.csv"
            with open(out_path, "w") as f:
                f.write("Word, POS, Count\n")
                for word, count in sorted(c.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{word[0]}, {word[1]}, " + str(count) + "\n")


# %%
