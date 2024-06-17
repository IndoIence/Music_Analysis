from classes.Pl import LyricsDataModule, TextClassificationModel
from utils import get_artist, CONFIG
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
if __name__ == "__main__":
    # %%
    artists = [get_artist(a_name) for a_name in CONFIG["top10"]]
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    # %%
    dm = LyricsDataModule(artists, word_limit=4e4, model=labse_model, transforms=False)
    dm.setup()
    input_dim = dm.train_dataset[0]["vector"].shape[0]
    output_dim = len(artists)
    t = pl.Trainer(max_epochs=10)
    model = TextClassificationModel(output_dim, input_dim)
    # %%
    t.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    # %%
    t.validate(model, dataloaders=dm.test_dataloader())
    # %%
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
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig(("temp" + ".png"), format="png")
# %%
