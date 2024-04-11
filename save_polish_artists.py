# %%
from utils import MyArtist, CONFIG
from pathlib import Path
import os
import spacy
import spacy_fastlang
import pickle
import logging
from tqdm import tqdm
logging.basicConfig( 
    filename=Path('logs/polish_non_polish.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
nlp = spacy.load("pl_core_news_lg")
nlp.add_pipe('language_detector')
# %%
file_path = Path('scraped_data/Genius/')
pl_path = file_path.parent / 'artists_pl'
npl_path = file_path.parent / 'artists_not_pl'
artists = [f for f in os.listdir(file_path) if os.path.isfile(file_path / f)]
# skipping
# skip_some = artists.index('Gojas(Gola Gianni)')
artists = artists[:1]
# delete this
# %%

# %%

# load all artist
# load them as myArtsist
# load thier songs as my songs
# get language of thieir songs and save them to the corresponding folder
def old_Art_to_MyArt(artists):
    for artist_path in tqdm(artists):
        logging.info(f"{file_path / artist_path}")
        print('printing cur art name: ', artist_path)
        print(f"{file_path / artist_path}")
        g_art = pickle.load(open(file_path / artist_path, 'rb'))
        my_art = MyArtist(g_art)
        counter = {}
        for song in my_art.songs:
            lang = nlp(song.lyrics)._.language
            if lang in counter:
                counter[lang] += 1
            else:
                counter[lang] = 1
            song.language = lang
        my_art.language =  max(counter, key=counter.get)
        if my_art.language == 'pl':
            with open(pl_path / artist_path, 'wb') as f:
                pickle.dump(my_art, f)
        else:
            with open(npl_path / artist_path, 'wb') as f:
                pickle.dump(my_art, f)
# %%
# Removing duplicates in polish artists and saving them to a txt file with all of them
def old_pl_art_url_to_txt():
    pl_artists = [f for f in os.listdir(pl_path) if os.path.isfile(pl_path / f)]
    all_urls = set()
    to_remove = []
    for i, art_path in enumerate(pl_artists[1:]):
        print(i, art_path)
        my_art = pickle.load(open(pl_path / art_path, 'rb'))
        art_url = my_art.url
        if art_url in all_urls:
            to_remove.append(art_path)
            logging.info(f'removing: {art_path}: duplicate of : {art_url}')
        all_urls.add(art_url)
    for r_art in to_remove:
        os.remove(pl_path / r_art)
# %%
# uses all artists saved in genius directory and saves their output to 
# 
def artists_url_to_txt(path = Path(CONFIG['Genius_scraping']['save_path'])):
    all_urls = set()
    for art_file in os.listdir(path):
        if not os.path.isfile(path / art_file):
            continue
        art = pickle.load(open(path / art_file,'rb'))
        if art.url in all_urls:
            logging.info(f'Found duplicate {art_file} of {art.url}')
        else:
            all_urls.add(art.url)
    with open(CONFIG['all_artists_urls_path'], 'w') as f:
        f.write('# here I want to have ALL genius url that have been scraped (polish or not) \n')
        for url in sorted(all_urls):
            f.write(url+ '\n')

        


# %%
