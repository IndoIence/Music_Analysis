from transformers import pipeline
import os
import pickle
import faiss
import datasets
import json
from tqdm import tqdm
from pathlib import Path
import spacy
import logging
from utils import get_100_biggest, get_artists, CONFIG, sanitize_art_name

logging.basicConfig(
    filename=CONFIG["Logging"]['wsd'],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


nlp = spacy.load("pl_core_news_lg")
auth_token: str = os.environ.get("CLARIN_KNEXT")

sample_text = (
    "Imiona zmieniać można, numery też, ale fakt to fakt"
    "Tańczą przy nas szprychy, wiesz? Super ekstra [unused0] szprychy [unused1]"
    "Możesz marzyć o takich dziewczynach - ciebie nie może być przy tym")

def load_model(model_name: str = "clarin-knext/wsd-encoder"):
    model = pipeline("feature-extraction", model=model_name, use_auth_token=auth_token)
    return model

def load_index(index_name: str = "clarin-knext/wsd-linking-index"):
    ds = datasets.load_dataset(index_name, use_auth_token=auth_token)['train']
    index_data = {
        idx: (e_id, e_text) for idx, (e_id, e_text) in
        enumerate(zip(ds['entities'], ds['texts']))
    }
    faiss_index = faiss.read_index("./encoder.faissindex", faiss.IO_FLAG_MMAP)
    return index_data, faiss_index

def predict(index, text: str = sample_text, top_k: int=3, raw=True):
    index_data, faiss_index = index
    # takes only the [CLS] embedding (for now)
    query = model(text, return_tensors='pt')[0][0].numpy().reshape(1, -1)

    scores, indices = faiss_index.search(query, top_k)
    scores, indices = scores.tolist(), indices.tolist()
    if raw:
        return scores, indices
    results = "\n".join([
        f"{index_data[result[0]]}: {result[1]}"
        for output in zip(indices, scores)
        for result in zip(*output)
    ])
    return results

def input_from_doc(doc, window=100):
    for sent in doc.sents:
        for token in sent:
            if token.is_alpha and not token.is_punct and not token.is_bracket:
                # put the [unused0] and [unused1] around the word
                middle = f"[unused0] {token.text} [unused1]"
                start_ind = max(0, token.idx-window)
                start = doc.text[start_ind:token.idx]
                end = doc.text[token.idx+len(token.text):token.idx+len(token.text)+window]
                yield token.text, start + middle + end

if __name__ == "__main__":
    out_dir = CONFIG["wsd_outputs"]
    in_dir = Path(CONFIG['artists_pl_path'])
    artist_names = CONFIG["artist_names"]
    model = load_model()
    # loading to gpu
    index_data, index = load_index()
    gpu_nr = 0
    res= faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_nr, index)
    index = (index_data, gpu_index)
    # actual search

    for a_name in artist_names:
        f_name = sanitize_art_name(a_name)
        with open(in_dir / (f_name + '.artPkl'), 'rb') as f:
            artist = pickle.load(f)
        logging.info(f"Start wsd for : {artist.name}")
        f_name = sanitize_art_name(artist.name) + '.jsonl'
        with open(Path(out_dir) / f_name, 'w') as f:
            for song in tqdm(artist.get_limit_songs(5000, only_art=True)):
                doc = nlp(song.lyrics)
                d = {"name": song.title,
                     "text": song.lyrics,
                     "wsd": []}
                for word, i in tqdm(input_from_doc(doc)):
                    scores, indices = predict(index, i)
                    suggestions = {}
                    for lists in zip(scores, indices):
                        for s,i in zip(*lists):
                            plwn_id, sense_def = index_data[i]
                            sense, definition = sense_def
                            suggestions[plwn_id]: {
                                "sense": s,
                                "definition": definition,
                                "score": i,
                            }
                    d["wsd"].append({"word":word, "suggestions": suggestions})
            try:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
            except:
                logging.error(f"Saving to file failed for : {artist.name}")


"""

"""