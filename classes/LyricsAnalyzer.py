import spacy.tokens
import plwn
import spacy

from utils import get_biggest_by_lyrics_len, CONFIG, wsd_predict
from classes.MyArtist import MyArtist
from classes.WordNet import WordNet
from pathlib import Path

from tqdm import tqdm
from typing import Any
from collections import Counter
from spacy.tokens import Doc
from tqdm import tqdm


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


class EmoAnalizer:
    def __init__(self, wsd_model, index_data, faiss_index):
        self.wsd_model = wsd_model
        self.index_data = index_data
        self.faiss_index = faiss_index
        self.wn = WordNet(CONFIG["wordnet_file"])
        self.nlp = spacy.load("pl_core_news_sm")

    def decode_prediction(self, scores, indices, index_data):
        output: list[dict[str, Any]] = []
        for lists in zip(scores, indices):
            for score, index in zip(*lists):
                plwn_id, sense_def = index_data[index]
                sense, definition = sense_def
                output.append(
                    {
                        "plwn_id": plwn_id,
                        "sense": sense,
                        "definition": definition,
                        "score": score,
                    }
                )
        return output

    def get_song_emot_sent(self, song):
        sentiment_count = {}
        emotions_count = {}
        doc = self.nlp(song.get_clean_song_lyrics(lower=False))
        for word, input in tqdm(self.input_from_doc(doc), desc=f"Song: {song.title}"):
            scores, indices, vector = wsd_predict(
                self.faiss_index, self.wsd_model, input
            )
            predictions = self.decode_prediction(scores, indices, self.index_data)
            for pred in predictions:
                try:
                    lexical_unit = self.wn.wn.lexical_unit_by_id(pred["plwn_id"])
                except plwn.exceptions.LexicalUnitNotFound:  # type: ignore
                    continue
                info = self.wn.get_info_from_lex_unit(lexical_unit)
                emotions = info["emotions"]
                sentiment = info["sentiment"]
                if sentiment:
                    sentiment_count[sentiment] = sentiment_count.get(sentiment, 0) + 1
                if emotions:
                    for emotion in emotions:
                        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
        return sentiment_count, emotions_count

    def input_from_doc(self, doc: Doc, window=100):
        """
        Input: doc -> a spacy doc because i need to have a consistent approach
        for the brackets in wsd
        """

        def valid_token(token: spacy.tokens.Token):
            return not (token.is_punct or token.is_bracket or token.is_space)

        for sent in doc.sents:
            for token in sent:
                if not valid_token(token):
                    continue
                # put the [unused0] and [unused1] around the word
                middle = f"[unused0] {token.text} [unused1]"
                start_ind = max(0, token.idx - window)
                start = doc.text[start_ind : token.idx]
                end = doc.text[
                    token.idx + len(token.text) : token.idx + len(token.text) + window
                ]
                yield token.text, start + middle + end
