import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from utils import get_biggest_arts, get_artist, CONFIG
from classes.MyArtist import MyArtist
from typing import Iterator


class LyricsDataModule(pl.LightningDataModule):
    def __init__(self, artists: Iterator[MyArtist], word_limit=4e4):
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
    def __init__(self, model_name, n_classes):
        super(TextClassificationModel, self).__init__()
        self.model_labse = SentenceTransformer(model_name)
        self.vectorizer = CountVectorizer()
        self.classifier = torch.nn.Linear(
            768 + len(self.vectorizer.get_feature_names_out()), n_classes
        )
        self.label_encoder = LabelEncoder()

    def forward(self, texts):
        labse_embeddings = self.model_labse.encode(texts, convert_to_tensor=True)
        bow_embeddings = torch.tensor(
            self.vectorizer.transform(texts).toarray(), dtype=torch.float32
        )
        combined_embeddings = torch.cat((labse_embeddings, bow_embeddings), dim=1)
        return self.classifier(combined_embeddings)

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        labels = self.label_encoder.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = self.forward(texts)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        labels = self.label_encoder.transform(labels)
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = self.forward(texts)
        val_loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def prepare_data(self, artists: Iterator[MyArtist], word_limit=4e4) -> pd.DataFrame:
    data_list = []
    for artist in artists:
        songs = artist.get_limit_songs(word_limit, only_art=True)
        print(len(songs))
        for song in songs:
            data_list.append(
                {
                    "texts": song.get_clean_song_lyrics(),
                    "artists": artist.name_sanitized,
                }
            )
    return pd.DataFrame(data_list)


class TextDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_train, X_val, y_train, y_val = train_test_split(
            self.data["texts"], self.data["artists"], test_size=0.2, random_state=42
        )
        self.train_dataset = TextDataset(X_train.values, y_train.values)
        self.val_dataset = TextDataset(X_val.values, y_val.values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def get_data(artists: Iterator[MyArtist]):
    data_list = []
    for artist in artists:
        songs = artist.get_limit_songs(4e4, only_art=True)
        print(len(songs))
        for song in songs:
            data_list.append(
                {
                    "texts": song.get_clean_song_lyrics(),
                    "artists": artist.name_sanitized,
                }
            )
    return data_list


# Assuming the data preparation part
artists = [get_artist(name) for name in CONFIG["top10"]]
data = pd.DataFrame(get_data(artists))

# Initialize Data Module and Model
data_module = TextDataModule(data)
model = TextClassificationModel(
    "sentence-transformers/LaBSE", n_classes=len(data["artists"].unique())
)

# Initialize Trainer and fit model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
