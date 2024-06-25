# %%
from utils import get_artist, CONFIG, get_all_artists, data_years
from classes.Pl import LabseDataModule, ClassifyLabseModel
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
from pathlib import Path
from utils_pl import save_confusion_matrix, get_confussion_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%

top10 = [get_artist(name) for name in CONFIG["top10"]]

songs = [song for a in top10 for song in a.solo_songs[:130]]
songs_more = [song for a in top10 for song in a.songs[:300]]

dm = LabseDataModule(songs)
# dm = LabseDataModule(songs_more)


# %%
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5),
    ModelCheckpoint(monitor="val_loss", mode="min"),
]
model = ClassifyLabseModel(output_dim=len(top10))
t = Trainer(max_epochs=40, log_every_n_steps=1, callbacks=callbacks)
# %%
t.fit(model, dm)

# %%
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_confussion_matrix(model, dm, vector_key="vector"):
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
    cm = confusion_matrix(labels_to_names, outputs, labels=dm.labels_encoder.classes_)
    return cm, dm.labels_encoder.classes_


cm, labels = get_confussion_matrix(model, dm, vector_key="labse_vector")
p = Path(CONFIG["plots_path"]) / "labse"
save_confusion_matrix(cm, labels=labels, dir_path=p)

# %%
