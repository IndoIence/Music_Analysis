# %%
import pytorch_lightning as pl
import numpy as np
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_artist, CONFIG
from classes.MyArtist import MyArtist

from typing import Iterable
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from transformers import BertTokenizerFast


class LyricsDataModule(pl.LightningDataModule):
    def __init__(self, artists: Iterable[MyArtist], model, transforms, word_limit=4e4):
        super().__init__()
        self.artists = artists
        self.word_limit = word_limit
        self.model = model
        self.transforms = transforms

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
        df = pd.DataFrame(data_list)
        df["encoded_labels"] = LabelEncoder().fit_transform(df["artists"])
        return df

    def setup(self, stage=None):
        data = self.prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(
            data["texts"], data["encoded_labels"], test_size=0.2, random_state=42
        )
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(X_train.values)
        self.train_dataset = LabseDataset(
            X_train.values, y_train.values, self.model, self.vectorizer, self.transforms
        )
        self.val_dataset = LabseDataset(
            X_val.values, y_val.values, self.model, self.vectorizer, self.transforms
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)


class LabseDataset(Dataset):
    def __init__(self, texts, labels, model, vectorizer, transform=None):
        self.texts = texts
        self.labels = labels
        self.labse_model = model
        self.max_tokens_per_segment = model.tokenizer.model_max_length if model else 0
        self.transform = transform
        self.vectorizer = vectorizer

    def get_sentiment(self, text):
        # TODO: implement sentiment analysis
        return self.labse_model(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        vector = torch.tensor(
            self.vectorizer.transform([text]).toarray().reshape(-1)
        ).float()
        sample = {
            "text": text,
            "vector": vector,
            "label": label,
        }
        # TODO: concatenate the labse vector with the vector
        if self.transform:
            labse_vector = self.get_labse_vector(text)
            sample["labse_vector"] = labse_vector
            sample["vector"] = np.concatenate([vector, labse_vector], axis=1)
        return sample

    def get_labse_vector(self, text: str):
        text_segments = self.split_into_segments(text, self.labse_model.tokenizer)
        encoded = [self.labse_model.encode(segment) for segment in text_segments]
        averaged = np.mean(encoded, axis=0)
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


class TextClassificationModel(pl.LightningModule):
    def __init__(self, output_dim, input_dim):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["label"]
        # label encoder is it necessary?
        # should i get the
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# %%
if __name__ == "__main__":
    # %%
    artists = [get_artist(a_name) for a_name in CONFIG["top10"]]
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    dm = LyricsDataModule(artists, word_limit=4e4, model=labse_model, transforms=None)
    dm.setup()
    input_dim = dm.train_dataset[0]["vector"].shape[0]
    output_dim = len(np.unique(dm.train_dataset.labels))
    t = pl.Trainer(max_epochs=10)
    model = TextClassificationModel(output_dim, input_dim)
    # %%
    t.fit(model, dm)
    # %%
    t.validate(model, dm)
    # %%
    val_loader = dm.val_dataloader()
    outs = []
    model.eval()
    for batch in val_loader:
        vectors, labels = batch["vector"], batch["label"]
        output_val = model(vectors)
        output_val = dm.val_dataset.labels_encoder.inverse_transform(
            output_val.detach().numpy().reshape(-1, 10)
        )
        outs.extend(output_val)

    # %%

    cm = confusion_matrix(
        dm.val_dataset.labels, outs, labels=dm.val_dataset.labels_encoder.categories_[0]
    )
    # %%
    fix, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=dm.val_dataset.labels_encoder.categories_[0],  # type: ignore
        yticklabels=dm.val_dataset.labels_encoder.categories_[0],  # type: ignore
        ax=ax,
    )
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig(("temp" + ".png"), format="png")
# %%
