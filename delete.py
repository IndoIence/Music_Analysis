# %%
from utils import load_artist, load_artists, CONFIG, sanitize_name
from classes.myArtists import MyArtist
from classes.scraper import GeniusScraper
from pathlib import Path
import pickle

# need to get information about albums of the artist
# %%
genius_path = Path(CONFIG["polish_artist_path"])
art: MyArtist = load_artist("Tede", folder=genius_path, suffix=".artPkl")
print(art.songs)


# %%
albums = art.get_albums()
for album in albums:
    if "release_date_for_display" in album:
        print(album["release_date_for_display"], album["full_title"])
    # else:
    #     print(album)
# %%
for album in albums:
    print(album)

# %%
