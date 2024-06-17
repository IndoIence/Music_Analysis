import pytorch_lightning as pl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from classes.MyArtist import MyArtist

from tqdm import tqdm
from typing import Iterable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from transformers import BertTokenizerFast


class LyricsDataModule(pl.LightningDataModule):
    def __init__(self, artists: Iterable[MyArtist], model, transforms, word_limit=4e4):
        super().__init__()
        self.artists = artists
        self.word_limit = word_limit
        self.labse_model = model
        self.transforms = transforms
        self.labels_encoder = LabelEncoder()
        self.max_tokens_per_segment = model.tokenizer.model_max_length if model else 0

    def prepare_data(self) -> pd.DataFrame:
        data_list = []
        for artist in self.artists:
            songs = artist.get_limit_songs(self.word_limit, only_art=True)
            for song in tqdm(songs, desc=f"Processing {artist.name}"):
                clean_lyrics = song.get_clean_song_lyrics()
                if "labse" in self.transforms:
                    labse_vector = self.get_labse_vector(clean_lyrics)
                data_list.append(
                    {
                        "artist": artist.name_sanitized,
                        "text": clean_lyrics,
                        "labse_vector": labse_vector,
                    }
                )
        df = pd.DataFrame(data_list)
        df["encoded_label"] = self.labels_encoder.fit_transform(df["artist"])
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
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(train_df["text"].values)
        self.train_dataset = DataframeDataset(
            train_df,
            self.vectorizer,
            self.transforms,
        )
        self.test_dataset = DataframeDataset(
            test_df,
            self.vectorizer,
            self.transforms,
        )
        self.val_dataset = DataframeDataset(
            val_df,
            self.vectorizer,
            self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)


class DataframeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # should i drop the artist name? Or is it helpful to get it for plotting and stuff
        features = row.drop(["artist"])
        sample = dict(features)
        vector = torch.tensor(
            self.vectorizer.transform([sample["text"]]).toarray().reshape(-1)
        ).float()
        sample["vector"] = vector
        # TODO: concatenate the labse vector with the vector
        if self.transform and "labse_vector" in sample:
            sample["vector"] = torch.cat((sample["vector"], sample["labse_vector"]))
        return sample


class TextClassificationModel(pl.LightningModule):
    def __init__(self, output_dim, input_dim):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim**0.5))
        self.fc2 = nn.Linear(int(input_dim**0.5), 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["encoded_label"]
        # label encoder is it necessary?
        # should i get the
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["encoded_label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
