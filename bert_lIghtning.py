import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch import nn
import torch
from torchmetrics.classification import Accuracy
from transformers import DistilBertModel
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer
from torch import Tensor
from classes.MyArtist import MyArtist
from sklearn.model_selection import train_test_split


class BertFinetuning(pl.LightningModule):
    def __init__(self, n_classes, model_name):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # here i need pooling or sth
        output = self.classifier(output)
        output = torch.sigmoid(output)
        loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "label": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def on_train_epoch_end(self, outputs):
        predictions = torch.cat([output["predictions"] for output in outputs])
        labels = torch.cat([output["label"] for output in outputs])
        self.log(
            "train_accuracy",
            self.accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
        )


class BertInputsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "label": row["label"],
        }


class BertDataModule(pl.LightningDataModule):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def setup(self, stage=None):
        train, test = train_test_split(self.df, test_size=0.2)
        self.train_dataset = BertInputsDataset(train)
        self.test_dataset = BertInputsDataset(test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=4)


def songs_from_artists(
    arts: list[MyArtist],
    tokenizer,
    label2id: dict,
    song_limit: int = 300,
    mode="features",
):
    assert mode in ["features", "solo"], "mode must be either 'features' or 'solo'"
    data = []
    for art in arts:
        songs = art.songs if mode == "features" else art.solo_songs
        for song in songs[:song_limit]:
            clean_lyrics = song.get_clean_song_lyrics(lower=False)
            if clean_lyrics == "":
                continue
            input_ids, attention_mask = transform_text(clean_lyrics, tokenizer)
            for one_input, one_mask in zip(input_ids, attention_mask):
                data.append(
                    {
                        "label": label2id[song.artist_name],
                        "input_ids": one_input,
                        "attention_mask": one_mask,
                    }
                )
    return data


def chunks_from_artists(
    arts,
    tokenizer,
    label2id: dict,
    song_limit: int = 300,
):
    inputs = []
    attentions = []
    labels = []
    for art in arts:
        for song in art.solo_songs[:song_limit]:
            input_ids, attention_mask = transform_text(song.lyrics, tokenizer)
            for one_input, one_mask in zip(input_ids, attention_mask):
                inputs.append(one_input)
                attentions.append(one_mask)
                labels.append(label2id[song.artist_name])
    return inputs, attentions, labels


def tokenize(text, tokenizer: DistilBertTokenizer) -> tuple[Tensor, Tensor]:
    result = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    return result["input_ids"][0], result["attention_mask"][0]


def split_overlapping(
    tensor: Tensor, chunk_size: int = 510, stride: int = 400, min_chunk_len=100
) -> list[Tensor]:
    chunks = [tensor[i : i + chunk_size] for i in range(0, tensor.shape[0], stride)]
    if len(chunks) > 1:
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_len]
    return chunks


def add_special_tokens(input_chunks: list[Tensor], mask_chunks: list[Tensor]):
    for i in range(len(input_chunks)):
        input_chunks[i] = torch.cat(
            [torch.tensor([101]), input_chunks[i], torch.tensor([102])]
        )
        mask_chunks[i] = torch.cat(
            [torch.tensor([1]), mask_chunks[i], torch.tensor([1])]
        )


def add_padding(
    input_chunks: list[Tensor], mask_chunks: list[Tensor], tokenizer
) -> None:
    for i in range(len(input_chunks)):
        pad_len = 512 - input_chunks[i].shape[0]
        input_chunks[i] = torch.cat(
            [input_chunks[i], torch.tensor([tokenizer.pad_token_id] * pad_len)]
        )
        mask_chunks[i] = torch.cat([mask_chunks[i], torch.tensor([0] * pad_len)])


def stack_chunks(
    input_chunks: list[Tensor], mask_chunks: list[Tensor]
) -> tuple[Tensor, Tensor]:
    return torch.stack(input_chunks).long(), torch.stack(mask_chunks).int()


def transform_text(
    text: str,
    tokenizer: DistilBertTokenizer,
    chunk_size: int = 510,
    stride: int = 400,
    min_chunk_len=100,
):
    id_long, mask_long = tokenize(text, tokenizer)
    id_chunks = split_overlapping(id_long, chunk_size, stride, min_chunk_len)
    mask_chunks = split_overlapping(mask_long, chunk_size, stride, min_chunk_len)

    add_special_tokens(id_chunks, mask_chunks)
    add_padding(id_chunks, mask_chunks, tokenizer)
    input_ids, attention_mask = stack_chunks(id_chunks, mask_chunks)
    return input_ids, attention_mask
