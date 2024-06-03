from pathlib import Path
import yaml
import pickle
import heapq
from classes.myArtists import MyArtist

CONFIG = yaml.safe_load(open("config.yaml", "r"))
GENIUS_CONFIG = CONFIG["genius"]


def load_artist(
    artist_name: str,
    folder: Path = Path(CONFIG["polish_artist_path"]),
    suffix: str = ".artPkl",
):
    artist_name = sanitize_name(artist_name)
    artist_name += suffix
    with open(folder / artist_name, "rb") as f:
        art: MyArtist = pickle.load(f)
        return art


def load_artists(
    artist_names: list[str] = [],
    folder: Path = Path(CONFIG["polish_artist_path"]),
    suffix: str = ".artPkl",
):
    # if no artists are given load all artists in the folder
    if not artist_names:
        artist_names = [f_name.stem for f_name in folder.iterdir() if f_name.is_file()]
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


def get_biggest_artists(n: int, folder: Path = Path(CONFIG["polish_artist_path"])):
    # get n biggest artists by their total lyric count in a heap
    # it the count is the same artist with the lower index is chosen (no comparison of artists)
    artists = []
    for i, artist in enumerate(load_artists(folder=folder)):
        lyric_count = artist.len_lyrics
        if len(artists) < n:
            heapq.heappush(artists, (lyric_count, i, artist))
        elif lyric_count > artists[0][0]:
            heapq.heappushpop(artists, (lyric_count, i, artist))
    result = [artist for _, _, artist in artists]
    result.sort(key=lambda x: x.len_lyrics, reverse=True)
    return result
