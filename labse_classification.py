# %%
from utils import get_artist, CONFIG, get_all_artists, data_years
from classes.Pl import LabseDataModule, ClassifyLabseModel
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from utils_pl import save_confusion_matrix
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%

top10 = [get_artist(name) for name in CONFIG["top10"]]

songs = [song for a in top10 for song in a.solo_songs[:130]]
songs_more = [song for a in top10 for song in a.songs[:300]]

dm = LabseDataModule(songs)
dm2 = LabseDataModule(songs_more)


# %%
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5),
    ModelCheckpoint(monitor="val_loss", mode="min"),
]
model = ClassifyLabseModel(output_dim=len(top10))
t = Trainer(max_epochs=40, log_every_n_steps=1, callbacks=callbacks)
# %%
t.fit(model, dm)
t.validate(model, dm.val_dataloader())

# %%
save_confusion_matrix(model, dm, vector_key="labse_vector", fname="labse_top10")
# %%

# %%
model2 = ClassifyLabseModel(len(top10))
t = Trainer(max_epochs=40, log_every_n_steps=1, callbacks=callbacks)
t.fit(model2, dm2)
t.validate(model2, dm2)

# %%
