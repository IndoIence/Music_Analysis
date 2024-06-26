# %%
from classes.LyricsAnalyzer import EmoAnalizer
from utils import get_wsd_model, get_artist, get_all_artists, CONFIG, get_biggest_arts
from pathlib import Path

e = EmoAnalizer(*get_wsd_model(0))

# %%
out_file = Path(CONFIG["artists_pl_path"])
top10 = get_biggest_arts(10)
for artist in top10:
    print(artist.name)
    for song in artist.songs[:100]:
        if hasattr(song, "_emotions") and hasattr(song, "_sentiment"):
            continue
        sentiment_count, emotions_count = e.get_song_emot_sent(song)
        song.emotions = emotions_count
        song.sentiment = sentiment_count
        artist.to_pickle(out_file)
print("----------------------------------")
for artist in get_biggest_arts(30)[::-1]:
    print(artist.name)
    for song in artist.songs:
        if hasattr(song, "_emotions") and hasattr(song, "_sentiment"):
            continue
        sentiment_count, emotions_count = e.get_song_emot_sent(song)
        song.emotions = emotions_count
        song.sentiment = sentiment_count
        artist.to_pickle(out_file)
