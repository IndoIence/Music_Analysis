# %%
from utils import get_artist, CONFIG, get_biggest_arts
import pytorch_lightning as pl
from utils_pl import save_confusion_matrix, TextClassificationModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
from pathlib import Path
from utils_pl import save_confusion_matrix, LabseDataModule
import torchmetrics
from classes.MySong import MySong
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from utils_pl import get_labse_vector
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            }
        )
        # TODO: put vectorized text here maybe? No can't because tocenizer should be only from train

    df = pd.DataFrame(data)
    df["label"] = LabelEncoder().fit_transform(df["artist"])
    return df


# %%
if __name__ == "__main__":
    # %%
    max_epochs = 30
    arts = get_biggest_arts(30, only_art=True, mode="songs")
    songs = [song for a in arts for song in a.solo_songs[:150]]
    df = get_df(songs)
    name = "tfidf_30_artists"
    conf_matrix = torchmetrics.classification.ConfusionMatrix(
        task="multiclass",
        num_classes=len(arts),
    )
    print(len(songs))
    for transform in [[], ["labse"]]:
        dm = LabseDataModule(
            df, label_key="label", vector_key="count_vector", transforms=transform
        )
        dm.setup()
        input_dim = dm.train_dataset[0]["vector"].shape[0]
        output_dim = len(arts)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ]
        model = TextClassificationModel(output_dim, input_dim)
        t = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=1, callbacks=callbacks)
        t.fit(model, dm)
        validate_output = t.validate(model, dataloaders=dm.val_dataloader())
        f_name = name + "_labse" if "labse" in transform else name + "_no_labse"
        save_confusion_matrix(model, dm, fname=f_name, dir_name="tfidf_confusion")
        path = Path(t.checkpoint_callback.best_model_path).parent.parent
        with open(path / "model_info.txt", "w") as f:
            for key, val in validate_output[0].items():
                f.write(f"{key}: {val}\n")

# %%
