from transformers import pipeline
import os
import pickle
import faiss
from datasets import load_dataset
import json
from tqdm import tqdm
from pathlib import Path
import spacy
from spacy.tokens import Doc

import logging
import typing

if typing.TYPE_CHECKING:
    from classes.MyArtist import MyArtist
from utils import CONFIG, sanitize_art_name, get_artist

logging.basicConfig(
    filename=CONFIG["Logging"]["wsd"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


nlp = spacy.load("pl_core_news_sm")
auth_token: str | None = os.environ.get("CLARIN_KNEXT")

sample_text = (
    "Imiona zmieniać można, numery też, ale fakt to fakt"
    "Tańczą przy nas szprychy, wiesz? Super ekstra [unused0] szprychy [unused1]"
    "Możesz marzyć o takich dziewczynach - ciebie nie może być przy tym"
)


def load_model(model_name: str = "clarin-knext/wsd-encoder"):
    model = pipeline("feature-extraction", model=model_name, use_auth_token=auth_token)
    return model


def load_index(index_name: str = "clarin-knext/wsd-linking-index"):
    ds = load_dataset(index_name, use_auth_token=auth_token)["train"]
    index_data = {
        idx: (e_id, e_text)
        for idx, (e_id, e_text) in enumerate(zip(ds["entities"], ds["texts"]))
    }
    faiss_index = faiss.read_index("./ignore/encoder.faissindex", faiss.IO_FLAG_MMAP)
    return index_data, faiss_index


def predict(index, text: str = sample_text, top_k: int = 3, raw=True):
    index_data, faiss_index = index
    # takes only the [CLS] embedding (for now)
    query = model(text, return_tensors="pt")[0][0].numpy().reshape(1, -1)
    scores, indices = faiss_index.search(query, top_k)
    scores, indices = scores.tolist(), indices.tolist()
    if raw:
        return scores, indices, query
    results = "\n".join(
        [
            f"{index_data[result[0]]}: {result[1]}"
            for output in zip(indices, scores)
            for result in zip(*output)
        ]
    )
    return results


def input_from_doc(doc: Doc, window=100):
    """
    Input: doc -> a spacy doc because i need to have a consistent approach
    for the brackets in wsd
    """
    for sent in doc.sents:
        for token in sent:
            if token.is_punct or token.is_bracket or token.is_stop:
                continue
            # put the [unused0] and [unused1] around the word
            middle = f"[unused0] {token.text} [unused1]"
            start_ind = max(0, token.idx - window)
            start = doc.text[start_ind : token.idx]
            end = doc.text[
                token.idx + len(token.text) : token.idx + len(token.text) + window
            ]
            yield token.text, start + middle + end


def load_artists():
    for art_name in CONFIG["wsd"]["artists"]:
        art_name = sanitize_art_name(art_name)
        yield get_artist(art_name)


def get_song_wsd(lyrics: str, index_data):
    text = song.get_clean_song_lyrics()
    doc = nlp(text)
    song_dict = {
        "name": song.title,
        "text": text,
        "wsd": [],
    }

    for word, context in tqdm(input_from_doc(doc), song.title):
        scores, indices, vector = predict(index, context)
        suggestions = {}
        # transform 2d lists (scores, indices) to human readable outputs
        for lists in zip(scores, indices):
            for score, cur_ind in zip(*lists):
                plwn_id, sense_def = index_data[cur_ind]
                sense, definition = sense_def
                suggestions[plwn_id] = {
                    "sense": sense,
                    "definition": definition,
                    "score": score,
                }
        song_dict["wsd"].append(
            {
                "word": word,
                "vector": vector.tolist(),  # type: ignore
                "suggestions": suggestions,
            }
        )
    return song_dict


if __name__ == "__main__":
    """
    in CONFIG file:
        list of artists
        word limit for artist
    for every artist get songs (set the limit for words)
    for every song make a spacy doc
    for every doc get get tokens
    for every token get the context around it
    pass that context to make a prediction
    save all predictions from a song to a dictionary
    save that dict to a jsonl file
    """
    word_limit = CONFIG["wsd"]["word_limit"]
    out_dir = CONFIG["wsd"]["outputs"]
    model = load_model()
    # loading to gpu
    index_data, index = load_index()
    gpu_nr = 0
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_nr, index)
    index = (index_data, gpu_index)

    for artist in load_artists():
        logging.info(f"Start wsd for : {artist.name}")
        out_fname = sanitize_art_name(artist.name) + ".jsonl"
        # clear the file first before writing to it
        open(Path(out_dir) / out_fname, "w")
        limit_songs = artist.get_limit_songs(word_limit, only_art=True)

        for song in tqdm(limit_songs, f"{artist.name} songs"):
            song_dict = get_song_wsd(song.get_clean_song_lyrics(), index_data)
            with open(Path(out_dir) / out_fname, "a") as f:
                try:
                    json.dumps(
                        song_dict,
                        ensure_ascii=False,
                        # indent=4,
                        # separators=(",", ":"),
                    )
                    f.write("\n")
                except:
                    logging.error(f"Saving to file failed for : {artist.name}")
