# %%
from utils import CONFIG, sanitize_art_name
from pathlib import Path
import pickle
import os

# %%
in_dir = Path(CONFIG["artists_pl_path"])
out_dir = Path(CONFIG["websty_path"])
[os.remove(f) for f in out_dir.iterdir() if f.is_file]

artist_names = CONFIG["artist_names"]
for a_name in artist_names:
    f_name = sanitize_art_name(a_name)
    with open(in_dir / (f_name + '.artPkl'), 'rb') as f:
        artist = pickle.load(f)
    out = ''
    songs = artist.get_limit_songs(limit = 30000, only_art=True)
    for i, song in enumerate(songs):
        out += str(i) + '\n' + song.clean_song_text() + '\n'
    open(out_dir / (f_name + '.txt'), 'w').write(out)

# %%

import zipfile
out_dir = Path(CONFIG["websty_path"])

zip_name = 'websty.zip'
with zipfile.ZipFile(Path(CONFIG["websty_path"]) / zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for ff in (f for f in out_dir.iterdir() if f.is_file() and f.suffix == '.txt'):
        zf.write(ff, ff.name)


