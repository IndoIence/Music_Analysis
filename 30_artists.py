# %%
from utils import get_artist, CONFIG, get_biggest_arts
from utils_pl import (
    save_confusion_matrix,
    TextClassificationModel,
    get_confussion_matrix,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

import os
from pathlib import Path
from utils_pl import LabseDataModule
import torchmetrics
from classes.MySong import MySong
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from utils_pl import get_labse_vector
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def get_df(songs: list[MySong]):
    data = []
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    for song in tqdm(songs, desc="Processing songs"):
        data.append(
            {
                "text": song.get_clean_song_lyrics(),
                "artist": song.artist_name,
                "year": song.date["year"],
                "labse_vector": get_labse_vector(
                    song.get_clean_song_lyrics(), labse_model
                ),
                "title": song.title,
            }
        )
        # TODO: put vectorized text here maybe? No can't because tocenizer should be only from train

    df = pd.DataFrame(data)
    df["label"] = LabelEncoder().fit_transform(df["artist"])
    return df


def do(df, vector_key, transform, max_epochs, arts):
    dm = LabseDataModule(
        df, label_key="label", vector_key=vector_key, transforms=transform
    )
    dm.setup()
    input_dim = dm.train_dataset[0]["vector"].shape[0]
    output_dim = len(arts)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
        ),
        ModelCheckpoint(monitor="val_loss", mode="min"),
    ]
    model = TextClassificationModel(output_dim, input_dim)
    t = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        deterministic=True,
    )
    t.fit(model, dm)
    test_output = t.test(model, dataloaders=dm.test_dataloader())
    name = str(len(arts)) + "_" + vector_key
    f_name = name + "_labse" if "labse" in transform else name + "_no_labse"
    path = Path(t.checkpoint_callback.best_model_path).parent.parent
    with open(path / "model_info.txt", "w") as f:
        for key, val in test_output[0].items():
            f.write(f"{key}: {val}\n")

    return model, dm, path


# %%
if __name__ == "__main__":
    # %%
    max_epochs = 20
    seed_everything(41, workers=True)
    # arts10 = get_biggest_arts(10, only_art=True, mode="songs")
    arts30 = get_biggest_arts(30, only_art=True, mode="songs")
    # arts_list = [arts10, arts10, arts30, arts30]
    # song_limits = [180, 360, 100, 150]
    # vector_keys = ["tf_idf_vector", "count_vector"]
    # song_list = ""
    # for arts, limit in zip(arts_list, song_limits):
    #     # alternate between solo_songs and songs
    #     song_list = "songs" if song_list == "solo_songs" else "solo_songs"
    #     songs = [song for a in arts for song in getattr(a, song_list)[:limit]]
    #     df = get_df(songs)
    #     df = df[df["text"] != ""]
    #     for vector_key in vector_keys:
    #         for transform in [[], ["labse"]]:
    #             do(df, vector_key, transform, max_epochs, arts)
    df = get_df([song for a in arts30 for song in a.songs[:180]])
    df = df[df["text"] != ""]
    # %%
    vector_keys = "count_vector"
    model, dm, path = do(df, vector_keys, [], max_epochs, arts30)
    # %%
    cm, labels = get_confussion_matrix(model, dm)
    save_confusion_matrix(cm, labels, path, fname="confusion_matrix.png")

    # %%
    needed = ["Peja", "Pikers", "Slums_Attack", "Taco_Hemingway", "Young_Igi"]
    # get the indexes of the needed artists
    vec_func = np.vectorize(lambda x: x in needed)
    indexes = np.where(vec_func(labels))[0]
    cm_subet = cm[np.ix_(indexes, indexes)]
    print(cm_subet)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        cm_subet,
        annot=True,
        # fmt=".2f", inst so don't need it
        cmap="Blues",
        xticklabels=needed,
        yticklabels=needed,
        ax=ax,
    )
    sec_x_ax = ax.secondary_xaxis("top")
    sec_x_ax.set_xlabel("Predicted", fontsize=12, labelpad=15)
    sec_x_ax.set_xticks([])
    sec_y_ax = ax.secondary_yaxis("right")
    sec_y_ax.set_ylabel("True", fontsize=12, labelpad=5)
    sec_y_ax.set_yticks([])
# %%
