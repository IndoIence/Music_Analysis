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
    artists: list[str],
    folder: Path = Path(GENIUS_CONFIG["save_path"]),
    suffix: str = ".artPkl",
):
    for artist_name in artists:
        yield load_artist(artist_name, folder, suffix)


def sanitize_name(name: str) -> str:
    return name.replace("/", " ").replace("?", " ").replace("\u200b", "").strip()
