from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from utils import CONFIG
import numpy as np
import seaborn as sns
import torch


def save_confusion_matrix(
    model, dm, vector_key="vector", fname="temp", dir_name="confusion_matrix"
):

    def get_confussion_matrix():
        outputs = []
        labels = []
        for input in dm.val_dataset:
            vector = input[vector_key]
            label = input["label"]
            out = model(vector)
            out = torch.argmax(out)
            outputs.append(out)
            labels.append(label)

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
