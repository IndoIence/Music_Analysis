from pathlib import Path
import yaml
import pickle

CONFIG = yaml.safe_load(open("config.yaml", "r"))
GENIUS_CONFIG = CONFIG["genius"]


def load_artist(
    artist_name: str,
    folder: Path = Path(GENIUS_CONFIG["save_path"]),
    suffix: str = ".artPkl",
):
    artist_name = sanitize_name(artist_name)
    artist_name += suffix
    with open(folder / artist_name, "rb") as f:
        return pickle.load(f)


def load_artists(
    artist_names: list[str] = [],
    folder: Path = Path(GENIUS_CONFIG["save_path"]),
    suffix: str = ".artPkl",
):
    # if no artists are given load all artists in the folder
    if not artist_names:
        artist_names = [f_name for f_name in folder.iterdir() if f_name.is_file()]
    for artist_name in artist_names:
        yield load_artist(artist_name, folder, suffix)


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", " ")
        .replace("?", " ")
        .replace("\u200b", "")
        .replace("\n", "")
        .replace(" ", "_")
        .strip()
    )
