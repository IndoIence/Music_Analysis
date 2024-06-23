# %%
from classes.Pl import LyricsDataModule, TextClassificationModel
from utils import get_artist, CONFIG, get_biggest_by_lyrics_len
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
from utils_pl import save_confusion_matrix
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def get_arts_list(no_sokol=True):
    if no_sokol:
        bigs = get_biggest_by_lyrics_len(11, only_art=False)
        bigs.pop(9)
    else:
        bigs = get_biggest_by_lyrics_len(10, only_art=False)
    top10 = [get_artist(a_name) for a_name in CONFIG["top10"]]
    # bigs_only_art = [a.get_limit_songs(4e4, only_art=True) for a in bigs]

    # top_10_only_art = [a.get_limit_songs(4e4, only_art=True) for a in top10]
    # top_10_features = [a.get_limit_songs(8e4, only_art=False) for a in top10]
    return bigs, bigs, top10, top10


# %%
if __name__ == "__main__":
    # %%
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    max_epochs = 15
    arts_lists = get_arts_list(no_sokol=True)
    word_limits = [4e4, 8e4, 4e4, 8e4]
    names = ["biggest", "biggest_features", "top10", "top10_features"]
    only_arts = [True, False, True, False]
    for transform in [[], ["labse"]]:
        for arts_list, word_limit, name, only_art in zip(
            arts_lists, word_limits, names, only_arts
        ):
            print(arts_list, word_limit, name, only_art)
            dm = LyricsDataModule(
                arts_list,
                labse_model,
                word_limit=word_limit,
                only_art=only_art,
                transforms=transform,
            )
            dm.setup()

            input_dim = dm.train_dataset[0]["vector"].shape[0]
            output_dim = len(arts_list)
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5),
                ModelCheckpoint(monitor="val_loss", mode="min"),
            ]
            model = TextClassificationModel(len(arts_list), input_dim)
            t = pl.Trainer(
                max_epochs=max_epochs, log_every_n_steps=1, callbacks=callbacks
            )
            t.fit(model, dm)
            t.validate(model, dataloaders=dm.val_dataloader())
            f_name = name + "_labse" if "labse" in transform else name + "_no_labse"
            save_confusion_matrix(model, dm, fname=f_name, dir_name="tfidf_confusion")

# %%
