# %%
import pickle
import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from Analysis_imports import clean_song_text
file_path = Path('scraped_data/Genius/MyArtists')
pl_artists_path = Path('scraped_data/artists_pl')
genius_path = file_path.parent
artists_files = os.listdir(file_path)
from pympler.asizeof import asizeof

# i need to rename to a consistent file artPkl
# %%



# %%
