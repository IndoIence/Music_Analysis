from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torchmetrics.classification
from utils import CONFIG
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from classes.MySong import MySong
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


def save_confusion_matrix(
    model, dm, vector_key="vector", fname="temp", dir_name="confusion_matrix"
):

    def get_confussion_matrix():
        outputs = []
        labels = []
        for batch in dm.val_dataloader():
            vector = batch[vector_key]
            label = batch["label"]
            out = model(vector)
            out = torch.argmax(out, dim=1)
            outputs.extend(out)
            labels.extend(label)

        labels_to_names = dm.labels_encoder.inverse_transform(labels)
        outputs = dm.labels_encoder.inverse_transform(outputs)
        cm = confusion_matrix(
            labels_to_names, outputs, labels=dm.labels_encoder.classes_
        )
        return cm

    cm = get_confussion_matrix()
    fix, ax = plt.subplots(figsize=(8, 7))
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=dm.labels_encoder.classes_,  # type: ignore
        yticklabels=dm.labels_encoder.classes_,  # type: ignore
        ax=ax,
    )
    plots_dir = Path(CONFIG["plots_path"]) / dir_name
    plots_dir.mkdir(exist_ok=True)
    f = plots_dir / (fname + ".png")
    plt.savefig(f, format="png", bbox_inches="tight")


# def get_labse_vector(text, labse_model, max_tokens_per_segment=256):

#     def split_into_segments(text: str, tokenizer):
#         len_no_special = (
#             max_tokens_per_segment - 2
#         )  # reserve 2 tokens for special tokens
#         tokens = tokenizer.encode(text, add_special_tokens=False, padding=True)
#         segments = [
#             tokens[i : i + len_no_special]
#             for i in range(0, len(tokens), len_no_special)
#         ]
#         for segment in segments:
#             segment.insert(0, tokenizer.cls_token_id)  # type: ignore
#             segment.append(tokenizer.sep_token_id)  # type: ignore
#         text_segments = [tokenizer.decode(segment) for segment in segments]

#         return text_segments

#     text_segments = split_into_segments(text, labse_model.tokenizer)
#     encoded = [labse_model.encode(segment) for segment in text_segments]
#     averaged = np.mean(encoded, axis=0)
#     return torch.tensor(averaged)


class DataframeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vectorizer,
        transform=None,
        tfidf=None,
        vector_key="vector",
        label_key="artist",
    ):
        self.dataframe = dataframe
        self.transform = transform
        self.vectorizer = vectorizer
        self.tfidf = tfidf
        self.vector_key = vector_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx]
        sample = dict(features)
        vector = self.vectorizer.transform([sample["text"]])
        sample["count_vector"] = torch.tensor(vector.toarray().reshape(-1)).float()
        if self.tfidf:
            vector = self.tfidf.transform(vector)
        sample["tf_idf_vector"] = torch.tensor(vector.toarray().reshape(-1)).float()

        if self.transform and "labse_vector" in sample:
            sample[self.vector_key] = torch.cat(
                (sample[self.label_key], torch.from_numpy(sample["labse_vector"]))
            )
        return {"vector": sample[self.vector_key], "label": sample[self.label_key]}


class LabseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        label_key="artist",
        vector_key="vector",
        transforms=None,
    ):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.label_key = label_key
        self.vector_key = vector_key
        self.transforms = transforms
        self.labels_encoder = LabelEncoder().fit(self.df[self.label_key])
        self.df["label"] = self.labels_encoder.transform(self.df[self.label_key])

    def setup(self, stage=None):
        train_df, test_df = train_test_split(
            self.df,
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
        helper = self.vectorizer.fit_transform(train_df["text"].values)
        self.tf_idf = TfidfTransformer().fit(helper)
        # here kurwa changing the labels or something
        self.df["labels"] = self.labels_encoder.fit(train_df["artist"])
        self.train_dataset = DataframeDataset(
            dataframe=train_df,
            vectorizer=self.vectorizer,
            transform=self.transforms,
            tfidf=self.tf_idf,
            label_key=self.label_key,
            vector_key=self.vector_key,
        )
        self.test_dataset = DataframeDataset(
            test_df,
            vectorizer=self.vectorizer,
            transform=self.transforms,
            tfidf=self.tf_idf,
            label_key=self.label_key,
            vector_key=self.vector_key,
        )
        self.val_dataset = DataframeDataset(
            val_df,
            vectorizer=self.vectorizer,
            transform=self.transforms,
            tfidf=self.tf_idf,
            label_key=self.label_key,
            vector_key=self.vector_key,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


def get_labse_vector(
    text,
    labse_model,
    max_tokens_per_segment=256,
):
    def split_into_segments(text: str, tokenizer):
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


class TextClassificationModel(pl.LightningModule):
    def __init__(self, output_dim, input_dim):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim**0.5))
        self.fc2 = nn.Linear(int(input_dim**0.5), 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=output_dim
        )
        self.f1_score = torchmetrics.classification.MulticlassF1Score(
            num_classes=output_dim
        )

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
        self.accuracy(outputs, labels)
        self.log("training_accuracy", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["label"]
        outputs = self.forward(vectors)
        # metrics
        loss = F.cross_entropy(outputs, labels)
        self.accuracy(outputs, labels)
        self.f1_score(outputs, labels)
        self.log("val_loss", loss)
        self.log("validation_accuracy", self.accuracy, on_epoch=True, on_step=True)
        self.log("f1 score", self.f1_score, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        vectors, labels = batch["vector"], batch["label"]
        outputs = self.forward(vectors)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss)
        self.accuracy(outputs, labels)
        self.log("test_accuracy", self.accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
