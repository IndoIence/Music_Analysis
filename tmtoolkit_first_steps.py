# %%
from utils import get_all_artists, CONFIG, sanitize_art_name
from pathlib import Path
from tqdm import tqdm
import tempfile
import spacy
import joblib
import pickle

nlp = spacy.load("pl_core_news_sm")
from tmtoolkit.corpus import Corpus, Document, print_summary, tokens_table


def get_artists_documents(in_dir=Path(CONFIG["artists_pl_path"])):
    def doc_labels(name: str):
        n = 1
        while True:
            yield name + f"_{n}"
            n += 1

    artist_names = CONFIG["artist_names"]
    for a_name in artist_names:
        f_name = sanitize_art_name(a_name)
        with open(in_dir / (f_name + ".artPkl"), "rb") as f:
            art = pickle.load(f)
            d = doc_labels(art.name)
            for song in art.songs:
                yield (next(d), song.clean_song_lyrics)


# %%
# for tf idf i need first all the texts from all artists together on a corpus
c_small = Corpus({k: v for k, v in get_artists_documents()}, spacy_instance=nlp, max_workers=0.1)

# %% learn how to save a corpus to a directory
# just pickle it and load it from pickle
CONFIG["corpus_path"] = "output_data/corpus/"
corpus_path = Path(CONFIG["corpus_path"])
# with open(corpus_path / 'tmtoolkit_top30.pkl', 'wb') as f:
#     pickle.dump(c_small, f)
joblib.dump(c_small, "tmtoolkit_top30.pkl")


# %%

from tmtoolkit.corpus import lemmatize, to_lowercase, dtm
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table

lemmatize(c_small)
to_lowercase(c_small)
mat, doc_labels, vocab = dtm(c_small, return_doc_labels=True, return_vocab=True)
tfidf_mat = tfidf(mat)
top_tokens = sorted_terms_table(tfidf_mat, vocab, doc_labels, top_n=5)
# %%
# how can i join all the songs in my corpus?
art_names = [sanitize_art_name(name) for name in CONFIG["artist_names"]]
join_dict = {name + "_joined": name + "*" for name in art_names}
from tmtoolkit.corpus import corpus_join_documents

corpus_join_documents(c_small, join_dict, match_type="glob")


# %%
def save_corpus(corpus, file_path):
    serializable_data = {
        "tokens": corpus.tokens,
        "doc_labels": corpus.doc_labels,
        "metadata": corpus.metadata,
        # Add other attributes as needed
    }
    joblib.dump(serializable_data, file_path)


def load_corpus(file_path):
    loaded_data = joblib.load(file_path)
    corpus = Corpus()
    corpus.tokens = loaded_data["tokens"]
    corpus.doc_labels = loaded_data["doc_labels"]
    corpus.metadata = loaded_data["metadata"]
    # Assign other attributes as needed
    return corpus


# %%
