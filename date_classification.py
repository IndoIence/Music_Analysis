# %%
from utils import get_artist, CONFIG, get_all_artists, data_years
import pandas as pd
import matplotlib.pyplot as plt
import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import BertTokenizerFast
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from classes.MySong import MySong
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


# %%
def get_df(songs: list[MySong]):
    data = []
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    for song in tqdm(songs, desc="Processing songs"):
        data.append(
            {
                "text": song.get_clean_song_lyrics(),
                "artist": song.artist_name,
                "year": song.date["year"],
                "labse_vector": (
                    song.labse_vector
                    if hasattr(song, "labse_vector")
                    else get_labse_vector(song.get_clean_song_lyrics(), labse_model)
                ),
                "title": song.title,
            }
        )
        # TODO: put vectorized text here maybe? No can't because tocenizer should be only from train

    df = pd.DataFrame(data)
    df = df[df.text != ""]
    df = df[df["year"].between(1989, 2024)]
    df["year"] = split_into_year_buckets(df["year"])
    return df


# # drop empty songs
# # %%
# labse_model = SentenceTransformer("sentence-transformers/LaBSE")
# df["labse_vector"] = list(map(lambda x: get_labse_vector(x, labse_model), df["text"]))
def get_all_songs_with_years():
    excluded_terms = ["- recenzja", "remix)", "mixtape", "remix]"]
    non_dupe_songs = []
    ids = set()
    for a in get_all_artists():
        for song in a.songs:
            if song.title in ids or any(
                term.lower() in song.title.lower() for term in excluded_terms
            ):
                continue
            ids.add(song.id)
            non_dupe_songs.append(song)

    # df = data_years(non_dupe_songs)
    # # drop the songs before 1991
    # df = df[df["year"] >= 1991]
    # df = df[df.text != ""]
    return non_dupe_songs


def split_into_year_buckets(l: list[float]) -> list[int]:
    buckets = []
    for year in l:
        if year < 2011:
            buckets.append(0)
        elif year < 2016:
            buckets.append(1)
        elif year < 2020:
            buckets.append(2)
        else:
            buckets.append(3)
    return buckets


# %%
import torchmetrics


class YearClassifcationModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(YearClassifcationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim**0.5))
        self.fc2 = nn.Linear(int(input_dim**0.5), 128)
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
        vectors, labels = batch["counted"], batch["year"]
        outputs = self.forward(vectors)
        outputs = outputs.type(torch.float64)
        with torch.autocast("cuda"):
            loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        self.accuracy(outputs, labels)
        self.log("train_accuracy", self.accuracy, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["counted"], batch["year"]
        outputs = self.forward(vectors)
        outputs = outputs.type(torch.float64)
        with torch.autocast("cuda"):
            loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        self.accuracy(outputs, labels)
        self.log("val_accuracy", self.accuracy, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def get_labse_vector(text, labse_model, max_tokens_per_segment=256):

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
        sample["counted"] = torch.cat((sample["counted"], sample["labse_vector"]))
        return sample


class DfDataModule(pl.LightningDataModule):
    # TODO: should add transforms here
    def __init__(self, df, batch_size=64, transorms=None):
        self.df = df
        self.batch_size = batch_size
        self.vectorizer = CountVectorizer(min_df=2)
        self.transforms = transorms

    def setup(self, stage=None):
        df_train, df_test = train_test_split(
            self.df, test_size=0.2, stratify=self.df["year"], random_state=42
        )
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


songs_list = get_all_songs_with_years()
df = get_df(songs_list)
dm = DfDataModule(df)
dm.setup()
input_dim = len(dm.test_dataset[0]["counted"])
model = YearClassifcationModel(input_dim, 4)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
    ),
    ModelCheckpoint(monitor="val_loss", mode="min"),
    TQDMProgressBar(refresh_rate=10),
]
t = pl.Trainer(max_epochs=7, callbacks=callbacks)
# %%

t.fit(model, dm.train_dataloader(), dm.test_dataloader())
# %%
t.validate(model, dm.test_dataloader())
valid_loader = dm.test_dataloader()
# %%
correct = 0
total = 0
outputs = []
labels = []
for batch in dm.test_dataloader():
    vectors, labels = batch["counted"], batch["year"]
    outputs = model.forward(vectors)
    outputs = torch.argmax(outputs, dim=1)
# %%
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

label2id = {"before_2011": 0, "2011-2015": 1, "2016-2019": 2, "2020-2024": 3}
id2label = {i: label for label, i in label2id.items()}
label_names = list(label2id.keys())

outputs = []
labels = []

for batch in dm.test_dataloader():
    vector = batch["counted"]
    label = batch["year"]
    out = model(vector)
    out = torch.argmax(out, dim=1)
    outputs.extend(out)
    labels.extend(label)
# %%
cm = confusion_matrix(labels, outputs)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


fix, ax = plt.subplots(figsize=(8, 7))

# get percentages instead of counts
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_names,  # type: ignore
    yticklabels=label_names,  # type: ignore
    ax=ax,
)


sec_x_ax = ax.secondary_xaxis("top")
sec_x_ax.set_xlabel("Predicted", fontsize=12, labelpad=15)
sec_x_ax.set_xticks([])
sec_y_ax = ax.secondary_yaxis("right")
sec_y_ax.set_ylabel("True", fontsize=12, labelpad=5)
sec_y_ax.set_yticks([])


# %%
