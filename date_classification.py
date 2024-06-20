 # %%
from utils import get_artist, CONFIG, get_all_artists, data_years
import pandas as pd
import matplotlib.pyplot as plt
import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import BertTokenizerFast
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
# %%
excluded_terms = ['- recenzja', 'remix)', 'mixtape', 'remix]']
non_dupe_songs = []
for a in get_all_artists():
    songs = [song for song in a.songs if not any(term.lower() in song.title.lower() for term in excluded_terms)]
    ids = set()
    for song in songs:
        if song.title in ids or any(term.lower() in song.title.lower() for term in excluded_terms):
            continue
        ids.add(song.id)
        non_dupe_songs.append(song)
# %%
df = data_years(non_dupe_songs)
# %%
del non_dupe_songs
# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

# drop the songs before 1990
df = df[df["year"] >= 1991]
# drop empty songs
df = df[df.text != ""]
labse_model = SentenceTransformer("sentence-transformers/LaBSE")
df["labse_vector"] = list(map(lambda x: get_labse_vector(x, labse_model), df["text"]))


# %%


class YearRegressionModel(pl.LightningModule):
    def __init__(self, input_dim):
        super(YearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim**0.5))
        self.fc2 = nn.Linear(int(input_dim**0.5), 128)
        self.fc3 = nn.Linear(128, 1)
        self.loss_fn = nn.MSELoss()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        vectors, labels = batch["counted"], batch["year"]
        outputs = self.forward(vectors)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["counted"], batch["year"]
        outputs = self.forward(vectors)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    


def get_labse_vector(text, labse_model, max_tokens_per_segment = 256):

    def split_into_segments(text: str, tokenizer: BertTokenizerFast):
        len_no_special = (
            max_tokens_per_segment - 2
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
    text_segments = split_into_segments(text, labse_model.tokenizer)
    encoded = [labse_model.encode(segment) for segment in text_segments]
    averaged = np.mean(encoded, axis=0)
    return torch.tensor(averaged)
class DfDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer):
        self.dataframe = dataframe
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        sample = dict(row)
        sample["counted"] = torch.tensor(
            self.vectorizer.transform([row["text"]]).toarray().reshape(-1)
        ).float()
        return sample
        
class DfDataModule(pl.LightningDataModule):
    # TODO: should add transforms here
    def __init__(self, df, batch_size=64, transorms=None):
        self.df = df
        self.batch_size = batch_size
        self.vectorizer = CountVectorizer(min_df=2)
        self.transforms = transorms

    def setup(self, stage=None):
        df_train, df_test = train_test_split(self.df, test_size=0.2, stratify=self.df["year"], random_state=42)
        self.vectorizer.fit(df_train["text"].values)
        self.train_dataset = DfDataset(df_train, vectorizer=self.vectorizer)
        self.test_dataset = DfDataset(df_test, vectorizer=self.vectorizer)

    def apply_transforms(self, x):
        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)
        return x
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# %%
dm = DfDataModule(df)
dm.setup()
# %%
input_dim = dm.train_dataset[0]["counted"].shape[0]
model = YearRegressionModel(input_dim)


# %%
t = pl.Trainer(max_epochs=10)
t.fit(model, dm.train_dataloader())
# %%
dm.train_dataset[51]

# %%
# print(len(df))
df = df[df['text'] != ""]
# %%
# %%
