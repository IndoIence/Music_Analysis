from classes.Pl import LyricsDataModule, TextClassificationModel
from utils import get_artist, CONFIG, get_biggest_by_lyrics_len
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def model_confusion_matrix(model, dm, fname='temp'):
        outputs = []
        labels = []
        val_loader = dm.test_dataset
        for batch in val_loader:
            vector = batch["vector"]
            label = batch["encoded_label"]
            out = model(vector)
            out = torch.argmax(out)
            outputs.append(out)
            labels.append(label)
        

        # %%
        labels_to_names = dm.labels_encoder.inverse_transform(labels)
        outputs = dm.labels_encoder.inverse_transform(outputs)
        cm = confusion_matrix(labels_to_names, outputs, labels=dm.labels_encoder.classes_)
        # %%
        fix, ax = plt.subplots(figsize=(7, 6))
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
        title = "Confusion Matrix" if fname == "temp" else "fname"
        plt.title(title)
        plt.show()
        plots_dir = Path(CONFIG["plots_path"])
        f = plots_dir / (fname + ".png")
        plt.savefig(f, format="png")

def get_three_arts_list():
    bigs = get_biggest_by_lyrics_len(10, only_art=False)
    top10 = [get_artist(a_name) for a_name in CONFIG["top10"]]
    # bigs_only_art = [a.get_limit_songs(4e4, only_art=True) for a in bigs]
    
    # top_10_only_art = [a.get_limit_songs(4e4, only_art=True) for a in top10]
    # top_10_features = [a.get_limit_songs(8e4, only_art=False) for a in top10]
    return bigs, top10, top10

# %%
if __name__ == "__main__":
    # %%
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    arts_lists = get_three_arts_list()
    word_limits = [4e4, 4e4, 8e4]
    names = ["biggest", "top10", "top10_features"]
    only_arts = [True, True, False]
    for transform in [["labse"],[]]:
        for arts_list, word_limit, name, only_art in zip(arts_lists, word_limits, names, only_arts):
            print(arts_list, word_limit, name, only_art)
            dm = LyricsDataModule(arts_list,labse_model, word_limit=word_limit, only_art=only_art, transforms=transform)
            dm.setup()
            input_dim = dm.train_dataset[0]["vector"].shape[0]
            output_dim = len(arts_list)
            t = pl.Trainer(max_epochs=10)
            model = TextClassificationModel(output_dim, input_dim)
            t.fit(
                model,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )
            t.validate(model, dataloaders=dm.test_dataloader())
            f_name = name + "_labse" if "labse" in transform else name + "_no_labse"
            model_confusion_matrix(model, dm, f_name)
    
    # artists = [get_artist(a_name) for a_name in CONFIG["top10"]]
    
    # dm = LyricsDataModule(artists, word_limit=4e4, model=labse_model)
    # dm.setup()
    # input_dim = dm.train_dataset[0]["vector"].shape[0]
    # output_dim = len(artists)
    # t = pl.Trainer(max_epochs=10)
    # model = TextClassificationModel(output_dim, input_dim)
    # # %%
    # t.fit(
    #     model,
    #     train_dataloaders=dm.train_dataloader(),
    #     val_dataloaders=dm.val_dataloader(),
    # )
    # # %%
    # t.validate(model, dataloaders=dm.test_dataloader())
    # # %%
    # model_confusion_matrix(model, dm, "no_labse")

    # # 2nd model with labse:
    # dm = LyricsDataModule(artists, word_limit=4e4, model=labse_model, transforms=["labse"])
    # dm.setup()
    # input_dim = dm.train_dataset[0]["vector"].shape[0]
    # output_dim = len(artists)
    # t = pl.Trainer(max_epochs=10)
    # model = TextClassificationModel(output_dim, input_dim)
    # # %%
    # t.fit(
    #     model,
    #     train_dataloaders=dm.train_dataloader(),
    #     val_dataloaders=dm.val_dataloader(),
    # )
    # # %%
    # t.validate(model, dataloaders=dm.test_dataloader())
    # model_confusion_matrix(model, dm, "labse")  
# %%
