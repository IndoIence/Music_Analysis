# %%
from utils import CONFIG, sanitize_art_name
from pathlib import Path
from classes.MyArtist import MyArtist
import pickle
import os
import zipfile


# %%
def get_websty_input(one_artist_file=False, words_limit=10000):
    in_dir = Path(CONFIG["artists_pl_path"])
    out_dir = Path(CONFIG["websty_path"])
    # remove all files in the out_dir
    [os.remove(f) for f in out_dir.iterdir() if f.is_file]

    artist_names = CONFIG["artist_names"]
    for a_name in artist_names:
        f_name = sanitize_art_name(a_name)
        with open(in_dir / (f_name + ".artPkl"), "rb") as f:
            artist: MyArtist = pickle.load(f)
        songs = artist.get_limit_songs(limit=words_limit, only_art=True)
        # save to file based on the format
        if one_artist_file:
            out = ""
            for i, song in enumerate(songs):
                out += str(i) + "\n" + song.get_clean_song_lyrics() + "\n"
            open(out_dir / (f_name + ".txt"), "w").write(out)
        else:
            for song in songs:
                open(
                    out_dir / (f_name + "_" + sanitize_art_name(song.title) + ".txt"),
                    "w",
                ).write(song.get_clean_song_lyrics())


# %%


def zip_websty():
    out_dir = Path(CONFIG["websty_path"]).parent

    zip_name = "websty.zip"
    with zipfile.ZipFile(
        Path(CONFIG["websty_path"]) / zip_name, "w", zipfile.ZIP_DEFLATED
    ) as zf:
        for ff in (f for f in out_dir.iterdir() if f.is_file() and f.suffix == ".txt"):
            zf.write(ff, ff.name)


get_websty_input(False, words_limit=20000)
zip_websty()
