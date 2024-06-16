import pytorch_lightning as pl
import numpy as np
import seaborn as sns
import pandas as pd

# import torch
# import torch.nn.functional as F

from utils import get_artist, CONFIG
from classes.MyArtist import MyArtist

from typing import Iterable
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from transformers import BertTokenizerFast


class LyricsDataModule(pl.LightningDataModule):
    def __init__(self, artists: Iterable[MyArtist], word_limit=4e4):
        self.artists = artists
        self.word_limit = word_limit

    def prepare_data(self) -> pd.DataFrame:
        data_list = []
        for artist in self.artists:
            songs = artist.get_limit_songs(self.word_limit, only_art=True)
            for song in songs:
                data_list.append(
                    {
                        "texts": song.get_clean_song_lyrics(),
                        "artists": artist.name_sanitized,
                    }
                )
        return pd.DataFrame(data_list)

    def setup(self, stage=None):
        data = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(
            data["texts"], data["artists"], test_size=0.2, random_state=42
        )
        self.train_dataset = TextDataset(X_train.values, y_train.values)
        self.val_dataset = TextDataset(X_val.values, y_val.values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TextClassificationModel(pl.LightningModule):
    def __init__(self, vectorizer, classifier, model):
        super(TextClassificationModel, self).__init__()
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.model: SentenceTransformer = model
        self.max_tokens_per_segment = model.tokenizer.model_max_length

    def forward(self, texts):
        labse_vectors = self.get_labse_vectors(texts)
        vectorized_texts = self.vectorizer.transform(texts)
        combined = np.hstack((labse_vectors, vectorized_texts.toarray()))
        return self.classifier.predict(combined)

    def get_labse_vectors(self, texts: Iterable[str]):
        text_segments = [
            self.split_into_segments(text, self.model.tokenizer) for text in texts
        ]
        encoded = [self.model.encode(segments) for segments in text_segments]
        averaged = [np.mean(enc, axis=0) for enc in encoded]
        return averaged

    def split_into_segments(self, text: str, tokenizer: BertTokenizerFast):
        """Split text into segments that fit into the model
        first we need to decode the text into tokens
        then we reverse the tokens into segments of correct length"""
        len_no_special = (
            self.max_tokens_per_segment - 2
        )  # reserve 2 tokens for special tokens
        tokens = tokenizer.encode(text, add_special_tokens=False, padding=True)
        segments = [
            tokens[i : i + len_no_special]
            for i in range(0, len(tokens), len_no_special)
        ]
        for segment in segments:
            segment.insert(0, tokenizer.cls_token_id)  # type: ignore
            segment.append(tokenizer.sep_token_id)  # type: ignore
        text_segments = [tokenizer.decode(segment) for segment in segments]
        return text_segments

    # def training_step(self, batch, batch_idx):
    #     texts, labels = batch
    #     vectors = self.vectorizer.transform(texts)
    #     # label encoder is it necessary?
    #     outputs = self.forward(vectors)
    #     loss = F.cross_entropy(outputs, labels)
    #     self.log("train_loss", loss)
    #     return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    artists = [get_artist(a_name) for a_name in CONFIG["top10"]]
    dm = LyricsDataModule(artists, word_limit=4e4)
    dm.setup()
    vectorizer = CountVectorizer()
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    train_vetorized = vectorizer.fit_transform(dm.train_dataset.texts)
    classifier.fit(train_vetorized, dm.train_dataset.labels)
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    model = TextClassificationModel(vectorizer, classifier, labse_model)
    outputs = model.forward(dm.val_dataset.texts)

    cm = confusion_matrix(dm.val_dataset.labels, outputs, labels=classifier.classes_)
    fix, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classifier.classes_,
        yticklabels=classifier.classes_,
        ax=ax,
    )
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig(("temp" + ".png"), format="png")
