# %%
from classes.LyricsAnalyzer import EmoAnalizer
from utils import get_wsd_model, get_artist, get_all_artists, CONFIG
from pathlib import Path

e = EmoAnalizer(*get_wsd_model(0))

# %%
out_file = Path(CONFIG["artists_pl_path"])
top30 = (get_artist(name) for name in CONFIG["top30"])
for artist in top30:
    print(artist.name)
    for song in artist.songs:
        if hasattr(song, "_emotions") and hasattr(song, "_sentiment"):
            continue
        sentiment_count, emotions_count = e.get_song_emot_sent(song)
        song.emotions = emotions_count
        song.sentiment = sentiment_count
        artist.to_pickle(out_file)
print("----------------------------------")
for artist in get_all_artists():
    print(artist.name)
    for song in artist.songs:
        if hasattr(song, "_emotions") and hasattr(song, "_sentiment"):
            continue
        sentiment_count, emotions_count = e.get_song_emot_sent(song)
        song.emotions = emotions_count
        song.sentiment = sentiment_count
        artist.to_pickle(out_file)