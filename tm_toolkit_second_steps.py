# %%
from utils import get_all_artists, CONFIG, sanitize_art_name, get_artist, get_biggest_by_lyrics_len
from pathlib import Path
from tqdm import tqdm
import tempfile
import spacy
import joblib
import pickle

nlp = spacy.load("pl_core_news_sm")
from tmtoolkit.corpus import Corpus, Document, print_summary, tokens_table, doc_tokens

# %%
artist_names = CONFIG["artist_names"]
artists = [get_artist(sanitize_art_name(name)) for name in artist_names]
artists = get_biggest_by_lyrics_len(n=50)
artists = [a for a in artists if a.songs]
texts = {}
for artist in tqdm(artists, "getting lyrics for artist"):
    text = "\n".join(song.clean_song_lyrics for song in artist.get_limit_songs(30000, strict=True, only_art=True))
    if text:
        texts[artist.name_sanitized] = text
# %%
# for tf idf i need first all the texts from all artists together on a corpus
c_small = Corpus(texts, spacy_instance=nlp)
# %%
from tmtoolkit.corpus import lemmatize, to_lowercase, dtm
from tmtoolkit.corpus import filter_clean_tokens
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table

filter_clean_tokens(c_small)
lemmatize(c_small)
to_lowercase(c_small)
mat, doc_labels, vocab = dtm(c_small, return_doc_labels=True, return_vocab=True)
tfidf_mat = tfidf(mat)
top_tokens = sorted_terms_table(tfidf_mat, vocab, doc_labels, top_n=5)
# %%
# c_small['Filipek'][:100]
tokens = doc_tokens(c_small)
for v in tokens.values():
    print(len(v))
# %%
[(a.name, sum(a.songs)) for a in artists if a.songs]

# %%
a = top_tokens.to_html()
a
# %%
