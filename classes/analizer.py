from classes.myArtists import MyArtist
from pathlib import Path
from tqdm import tqdm
import spacy


class Analizer:
    def __init__(self, artists: list[MyArtist]):
        self.artists: list[MyArtist] = list(artists)
        self.nlp = spacy.load("pl_core_news_sm")

    def get_vocabs(self, limit_words: int = 20000):
        for artist in self.artists:
            yield (artist, artist.get_vocab(limit_words=limit_words))

    def save_vocabs(self, vocab_dir: Path, limit_words: int = 20000):
        for artist, vocab in self.get_vocabs(limit_words=limit_words):
            inner_dir = vocab_dir / str(limit_words)
            if not inner_dir.exists():
                inner_dir.mkdir(parents=True)
            path = inner_dir / (artist.name + ".csv")
            with open(path, "w") as f:
                for k, v in vocab:
                    f.write(f"{k},{v}\n")
