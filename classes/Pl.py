import pytorch_lightning as pl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
from classes.MySong import MySong
from classes.MyArtist import MyArtist

from tqdm import tqdm
from typing import Iterable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from transformers import BertTokenizerFast
from sentence_transformers import SentenceTransformer


# class GoodDataModule(pl.LightningDataModule):
#     # datamodule that gives us songs with
#     def __init__(self, songs: list[MySong], batch_size = 32) -> None:
#         self.batch_size = batch_size


#     def setup(self):
#         self.train_dataset = DataframeDataset(train_df)
#         self.test_dataset = DataframeDataset(test_df)
#         self.val_dataset = DataframeDataset(val_df)

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
#         )

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class LyricsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        artists: Iterable[MyArtist],
        model,
        transforms=[],
        word_limit=4e4,
        only_art=True,
        labels_mode: str = "artist",
        vectorizer=CountVectorizer(),
        batch_size=32,
    ):
        super().__init__()
        self.artists = artists
        self.word_limit = word_limit
        self.only_art = only_art
        self.labse_model = model
        self.transforms = transforms
        self.labels_encoder = LabelEncoder()
        self.max_tokens_per_segment = model.tokenizer.model_max_length if model else 0
        self.labels_mode = labels_mode
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def prepare_data(
        self,
    ) -> pd.DataFrame:
        assert self.labels_mode in [
            "artist",
            "year",
        ], "label must be either 'artist' or 'year'"
        data_list = []
        for artist in self.artists:
            songs = artist.get_limit_songs(self.word_limit, only_art=self.only_art)
            for song in tqdm(songs, desc=f"Processing {artist.name}"):
                clean_lyrics = song.get_clean_song_lyrics()
                if not clean_lyrics:
                    continue
                data_point = {}
                if "labse" in self.transforms:
                    labse_vector = self.get_labse_vector(clean_lyrics)
                    data_point["labse_vector"] = labse_vector
                data_point["artist"] = artist.name_sanitized
                data_point["text"] = clean_lyrics
                data_list.append(data_point)
        df = pd.DataFrame(data_list)
        if self.labels_mode == "artist":
            df["label"] = self.labels_encoder.fit_transform(df["artist"])
        elif self.labels_mode == "year":
            df["label"] = df["year"]
        return df

    def get_labse_vector(self, text: str):
        text_segments = self.split_into_segments(text, self.labse_model.tokenizer)
        encoded = [self.labse_model.encode(segment) for segment in text_segments]
        averaged = np.mean(encoded, axis=0)
        return torch.tensor(averaged)

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

    def setup(self, stage=None):
        df = self.prepare_data()
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
        )
        helper = self.vectorizer.fit_transform(train_df["text"].values)
        self.tf_idf = TfidfTransformer().fit(helper)
        self.train_dataset = DataframeDataset(
            train_df,
            self.vectorizer,
            self.transforms,
            self.tf_idf,
        )
        self.test_dataset = DataframeDataset(
            test_df,
            self.vectorizer,
            self.transforms,
            self.tf_idf,
        )
        self.val_dataset = DataframeDataset(
            val_df,
            self.vectorizer,
            self.transforms,
            self.tf_idf,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class DataframeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer, transform=None, tfidf=None):
        self.dataframe = dataframe
        self.transform = transform
        self.vectorizer = vectorizer
        self.tfidf = tfidf

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx]
        sample = dict(features)
        vector = self.vectorizer.transform([sample["text"]])
        if self.tfidf:
            vector = self.tfidf.transform(vector)
        sample["vector"] = torch.tensor(vector.toarray().reshape(-1)).float()

        # TODO: concatenate the labse vector with the vector
        if self.transform and "labse_vector" in sample:
            sample["vector"] = torch.cat((sample["vector"], sample["labse_vector"]))
        return sample


class LabseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        songs: Iterable[MySong],
        labse_model=SentenceTransformer("LaBSE"),
        labels_mode: str = "artist",
        batch_size: int = 64,
    ):
        super().__init__()
        self.songs = songs
        self.labse_model = labse_model
        self.labels_encoder = LabelEncoder()
        self.max_tokens_per_segment = self.labse_model.tokenizer.model_max_length
        self.labels_mode = labels_mode
        self.batch_size = batch_size

    def prepare(
        self,
    ) -> pd.DataFrame:
        data_list = []
        for song in tqdm(self.songs, desc="Processing labse vectors of songs"):
            clean_lyrics = song.get_clean_song_lyrics()
            if not clean_lyrics:
                continue
            data_point = {}
            labse_vector = self.get_labse_vector(clean_lyrics)
            data_point["labse_vector"] = labse_vector
            data_point["artist"] = song.artist_name
            data_point["text"] = clean_lyrics
            data_list.append(data_point)
        df = pd.DataFrame(data_list)
        if self.labels_mode == "artist":
            df["label"] = self.labels_encoder.fit_transform(df["artist"])
        return df

    def get_labse_vector(self, text: str):
        text_segments = self.split_into_segments(text, self.labse_model.tokenizer)
        encoded = [self.labse_model.encode(segment) for segment in text_segments]
        averaged = np.mean(encoded, axis=0)
        return torch.tensor(averaged)

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

    def setup(self, stage=None):
        df = self.prepare()
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
        )
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(train_df["text"].values)
        self.train_dataset = DataframeDataset(
            train_df,
            self.vectorizer,
        )
        self.test_dataset = DataframeDataset(
            test_df,
            self.vectorizer,
        )
        self.val_dataset = DataframeDataset(
            val_df,
            self.vectorizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class ClassifyLabseModel(pl.LightningModule):
    def __init__(self, output_dim):
        super(ClassifyLabseModel, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=output_dim
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        vectors, labels = batch["labse_vector"], batch["label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        self.accuracy(outputs, labels)
        self.log("test_accuracy", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["labse_vector"], batch["label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        self.accuracy(outputs, labels)
        self.log("validation_accuracy", self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        vectors, labels = batch["labse_vector"], batch["label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.accuracy(outputs, labels)
        self.log("test_accuracy", self.accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
