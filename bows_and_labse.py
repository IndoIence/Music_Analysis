# %%
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize, OrdinalEncoder
from sklearn.model_selection import train_test_split
from utils import get_biggest_by_lyrics_len, get_artist, CONFIG
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)

from transformers import BertTokenizer
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


def get_xgboost_data(X_train, X_test, y_train, y_test):
    enc = OrdinalEncoder()
    y_train_enc = enc.fit_transform(y_train.values.reshape(-1, 1))
    y_test_enc = enc.transform(y_test.values.reshape(-1, 1))
    return X_train, X_test, y_train_enc, y_test_enc


# %%
def get_data(artists):
    data_list = []
    for artist in artists:
        songs = artist.get_limit_songs(4e4, only_art=True)
        for song in songs:
            data_list.append(
                {
                    "texts": song.get_clean_song_lyrics(),
                    "artists": artist.name_sanitized,
                }
            )
    return data_list


# get labse vectors should take text and split those texts into batches that fit into the model
def split_into_segments(
    text: str, tokenizer: BertTokenizer, max_tokens_per_segment=256
):
    max_tokens_per_segment -= 2  # reserve 2 tokens for special tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    segments = [
        tokens[i : i + max_tokens_per_segment]
        for i in range(0, len(tokens), max_tokens_per_segment)
    ]
    for segment in segments:
        segment.insert(0, tokenizer.cls_token_id)
        segment.append(tokenizer.sep_token_id)
    text_segments = [tokenizer.decode(segment) for segment in segments]
    return text_segments


def get_labse_vectors(model: SentenceTransformer, texts):
    # split texts into batches that can fit into the model
    seq_len = 256
    segments = [split_into_segments(text, model.tokenizer, seq_len) for text in texts]
    # encode the segments
    encoded = [model.encode(segment) for segment in segments]
    averaged = [np.mean(enc, axis=0) for enc in encoded]
    normalized = normalize(averaged)
    return normalized


# %%

artists = [get_artist(name) for name in CONFIG["top10"]]
data = pd.DataFrame(get_data(artists))

# %%
model_labse = SentenceTransformer("sentence-transformers/LaBSE")
vectorizer = CountVectorizer()

X_train, X_test, y_train, y_test = train_test_split(
    data["texts"], data["artists"], test_size=0.2, random_state=42
)
# %%
train_vect = vectorizer.fit_transform(X_train)
test_vect = vectorizer.transform(X_test)
# %%
train_labse = get_labse_vectors(model_labse, X_train.values)
test_labse = get_labse_vectors(model_labse, X_test.values)
# normalize the labse vectors
train_labse_norm = normalize(train_labse)
test_labse_norm = normalize(test_labse)
# %%
train_comb = np.hstack([train_labse_norm, normalize(train_vect.toarray())])
test_comb = np.hstack([test_labse_norm, normalize(test_vect.toarray())])

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

names_to_labels = {name: i for i, name in enumerate(np.unique(y_train))}
# %%

xgboost_train_labels = [names_to_labels[i] for i in y_train.values]
xgboost_test_labels = [names_to_labels[i] for i in y_test.values]
# %%
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = MultinomialNB()
clf4 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
)
clf5 = XGBClassifier()
clfs = [clf1, clf2, clf3, clf4, clf5]
test_vecs = [
    test_vect,
    test_comb,
    test_vect,
    test_vect,
    test_comb,
]
train_vecs = [train_vect, train_comb, train_vect, train_vect, train_comb]
ys_test = [y_test, y_test, y_test, xgboost_test_labels, xgboost_test_labels]
ys_train = [y_train, y_train, y_train, xgboost_train_labels, xgboost_train_labels]
titles = [
    "RandomForest",
    "RandomForest combined",
    "MultinomialNB",
    "XGBoost",
    "XGBoost combined",
]
# Train the classifiers
for clf, train_vec, labels in zip(clfs, train_vecs, ys_train):
    clf.fit(train_vec, labels)

# clf1.fit(train_vect, y_train)
# clf2.fit(train_comb, y_train)
# clf3.fit(train_vect, y_train)
# clf4.fit(train_vect, xgboost_train_labels)

# %%
outfile = Path(CONFIG["plots_path"])

for i, (clf, test_vec, y) in enumerate(zip(clfs, test_vecs, ys_test)):
    fix, ax = plt.subplots(figsize=(6, 6))
    y_pred = clf.predict(test_vec)
    cm = confusion_matrix(y, y_pred, labels=clf.classes_)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
        ax=ax,
    )
    f1 = f1_score(y, y_pred, average="weighted")
    ax.set_title(titles[i] + f" F1: {f1:.2f}")
    plt.savefig(outfile / (titles[i] + ".png"), format="png")

# %%
