# %%
# import pickle
# from tqdm import tqdm
# import os
# from pathlib import Path
# import logging
# from classes.MyArtist import MyArtist
from utils import CONFIG, get_all_urls, get_artists, get_artist, get_100_biggest
import spacy
import faiss
from wsd import load_index
# nlp = spacy.load("pl_core_news_lg")
#config_G = CONFIG["Genius_scraping"]
#token = config_G["GENIUS_BEARER"]


# %%
a = [a.name for a in get_100_biggest()]
# %%
# załaduj jedną piosenkę i popatrz na sturkturę gościnnych tracków
# czy jest opcja brać tylko jednego 
# jest opcja żeby nie było feat'ów
# trzeba by wtedy zaglądać do body -> _body i featured artists
# jeśli jest więcej niż jeden wtedy jest tzw lipa

# nie powinienem brać całego konteksu tylko jakąś część -> powiedzmy 30 słow z każdej strony