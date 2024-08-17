# %%
from pathlib import Path
from classes.MyArtist import MyArtist
from classes.MySong import MySong
import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import heapq
import faiss
import os
from transformers import pipeline
from datasets import load_dataset
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Iterable
from unidecode import unidecode
import yaml
from tqdm import tqdm

config_path = Path(__file__).parent / ".config.yaml"
CONFIG = yaml.safe_load(open(config_path))
# %%


def get_artist(
    name: str, path: Path = Path(CONFIG["artists_pl_path"]), ext: str = ".artPkl"
) -> MyArtist:
    if ext and ext[0] != ".":  # give dot to extension if it's not there
        ext = "." + ext
    name = sanitize_art_name(name)
    p = path / (name + ext)
    return pickle.load(open(p, "rb"))


def get_all_artists(path: Path = Path(CONFIG["artists_pl_path"])):
    art_paths = [f for f in path.iterdir() if f.is_file()]
    art_paths.sort()
    for art_path in tqdm(art_paths):
        filename, ext = art_path.stem, art_path.suffix
        art = get_artist(filename, path, ext)
        yield art


def get_all_urls():
    genius_path = Path(CONFIG["Genius_scraping"]["save_path"])
    urls = []

    for art_path in (f for f in genius_path.iterdir() if f.is_file()):
        with open(art_path, "rb+") as f:
            art = pickle.load(f)
            urls.append(art.url)
    return sorted(set(urls))


def get_biggest_arts(n: int = 50, only_art=True, mode="lyr") -> list[MyArtist]:
    heap: list[tuple[int, int, MyArtist]] = []
    assert mode in [
        "lyr",
        "songs",
    ], "mode must be either 'lyr' or 'songs'"
    for helper, artist in tqdm(
        enumerate(get_all_artists()), "sorting artists by lyrics length"
    ):
        if mode == "lyr":
            l = artist.lyrics_len_only_art if only_art else artist.lyrics_len_all
        elif mode == "songs":
            l = len(artist.solo_songs) if only_art else len(artist.songs)
        if len(heap) < n:
            heapq.heappush(heap, (l, helper, artist))
        else:
            heapq.heappushpop(heap, (l, helper, artist))
    return [artist for _, _, artist in sorted(heap, reverse=True)]


def all_artists_genius(
    genius_path: Path = Path(CONFIG["Genius_scraping"]["save_path"]),
):
    f_names = genius_path.iterdir()
    for art_file in f_names:
        artist_path = genius_path / art_file
        a = MyArtist(pickle.load(open(artist_path, "rb")))
        yield a


def recalc_my_artists(path: Path = Path(CONFIG["artists_pl_path"])):
    """remove all artists from the path and save them again"""
    # since artists are saved duirng iteration first getting the list of all files
    for f in tqdm(list(path.iterdir())):
        if not f.is_file():
            continue
        new_art = MyArtist(pickle.load(open(f, "rb")))
        f.unlink()
        new_art.to_pickle(path)


def load_jsonl(p: Path):
    with open(p) as f:
        while line := f.readline():
            yield json.loads(line)


def sanitize_art_name(name: str) -> str:
    translation_table: dict[int, str] = {
        ord("."): "_",
        ord(" "): "_",
        ord("/"): "",
        ord("?"): "",
        ord("\u200c"): "",
        ord("\u200b"): "",
        ord("("): "",
        ord(")"): "",
    }
    return name.translate(translation_table).strip()


def get_websty_input(
    words_limit: int,
    artists: Iterable[MyArtist] = [],
    out_dir=Path(CONFIG["websty_path"]),
):
    txt_dir = out_dir / "txts"
    txt_dir.mkdir(parents=True, exist_ok=True)
    # remove all files in the txt_dir
    [f.unlink() for f in txt_dir.iterdir() if f.is_file]

    for art in tqdm(artists):
        f_name = unidecode(art.name_sanitized) + ".txt"
        songs = art.get_limit_songs(limit=words_limit, only_art=True, strict=True)
        # save to file based on the format

        out = ""
        for i, song in enumerate(songs):
            out += str(i) + "\n" + song.get_clean_song_lyrics(linebreaks=True) + "\n"
        open(txt_dir / f_name, "w").write(out)

    zip_websty(txt_dir)


def zip_websty(dir: Path):
    """
    Zip all the txt files in the out_dir and save the into a parent direcotry
    """
    zip_name = "websty.zip"
    with ZipFile(dir.parent / zip_name, "w", ZIP_DEFLATED) as zf:
        for ff in (f for f in dir.iterdir() if f.is_file()):
            zf.write(ff, ff.name)


# def to_pkl(
#         artist: MyArtist,
#         artist_name: str,
#         suffix:str = '.artPkl',
#         out_path: Path=Path(CONFIG['artists_pl_path']),
#         ):
#     # before saving check if there are any / or ? in the name -> if so replace them
#         if '/' in artist_name or '?' in artist_name:
#             artist_name = artist_name.replace('/', ' ').replace('?', ' ')
#             # logging.info(
#             #     f'Changed name of the artist to {artist_name} because of the slashes/question marks in the name')
#         with open(out_path / (artist_name + suffix), 'wb') as f:
#             pickle.dump(artist, f)


def get_wsd_data():
    """for all artists return a tuple (file name, list)
    where list has wsd for each song"""
    path = Path(CONFIG["wsd"]["outputs"]).parent / "wsd"
    wsd_files = [f for f in path.iterdir() if f.is_file()]
    for wsd_file in wsd_files:
        result = []
        for song in load_jsonl(wsd_file):
            result.append(song["wsd"])
        yield (wsd_file, result)


def get_stopwords():
    path = CONFIG["stopwords"]
    return [line.rstrip() for line in open(path)]


def get_wsd_model(gpu_nr: int):
    """load and return wsd model, faiss index and faiss dataset
    good idea to load the index to the gpu now"""

    index_name = CONFIG["wsd"]["linking_index_name"]
    model_name = CONFIG["wsd"]["model_name"]
    auth_token = os.environ.get("CLARIN_KNEXT", "")
    faiss_index_path = CONFIG["wsd"]["faiss_index_path"]
    model = pipeline("feature-extraction", model=model_name, use_auth_token=auth_token)
    ds = load_dataset(index_name, use_auth_token=auth_token)["train"]  # type: ignore
    index_data = {
        idx: (e_id, e_text)
        for idx, (e_id, e_text) in enumerate(zip(ds["entities"], ds["texts"]))  # type: ignore
    }
    faiss_index = faiss.read_index(faiss_index_path, faiss.IO_FLAG_MMAP)
    res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(res, gpu_nr, faiss_index)
    return model, index_data, faiss_index


def wsd_predict(faiss_index, model, text: str, top_k: int = 3):
    # takes only the [CLS] embedding (for now)
    query = model(text, return_tensors="pt")[0][0].numpy().reshape(1, -1)
    scores, indices = faiss_index.search(query, top_k)
    scores, indices = scores.tolist(), indices.tolist()
    return scores, indices, query


# def prepare_data(
#     artists: Iterable[MyArtist],
#     word_limit,
#     only_art,
#     labse=False,
#     sentiment=False,
#     label="artist",
# ) -> pd.DataFrame:
#     # i want just two columns: vectors and labels
#     assert (label in ["artist", "date"], "label must be either 'artist' or 'date'")
#     basic_vectorizer = CountVectorizer()
#     data_list = []
#     for artist in artists:
#         songs = artist.get_limit_songs(word_limit, only_art=only_art)
#         for song in tqdm(songs, desc=f"Processing {artist.name}"):
#             clean_lyrics = song.get_clean_song_lyrics()
#             data_point = {}
#             if labse:
#                 labse_vector = get_labse_vector(clean_lyrics)
#                 data_point["labse_vector"] = labse_vector
#             if sentiment:
#                 ...
#             data_point["artist"] = artist.name_sanitized
#             data_point["text"] = clean_lyrics
#             data_point["year"] = song.date["year"]

#             data_list.append(data_point)
#     df = pd.DataFrame(data_list)
#     if label == "artist":
#         df["encoded_label"] = basic_vectorizer.fit_transform(df["artist"])
#     elif label == "date":
#         df["encoded_label"] = df["year"]
#     if

#     return df


def data_years(songs: Iterable[MySong]) -> pd.DataFrame:
    data_points = []
    for song in songs:
        data_point = {}
        if song.date is None:
            continue
        data_point["year"] = song.date["year"]
        data_point["text"] = song.get_clean_song_lyrics()
        data_point["artist"] = song.artist_name
        data_point["title"] = song.title
        data_points.append(data_point)
    df = pd.DataFrame(data_points)
    df = df.astype({"year": "float32"})
    return df


def split_into_year_buckets(l: list[float]) -> list[int]:
    buckets = []
    for year in l:
        if year < 2011:
            buckets.append(0)
        elif year < 2016:
            buckets.append(1)
        elif year < 2020:
            buckets.append(2)
        else:
            buckets.append(3)
    return buckets
