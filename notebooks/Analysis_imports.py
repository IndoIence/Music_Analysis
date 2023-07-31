import re
import nltk
import lyricsgenius
import pickle
from API_KEYS import GENIUS_BEARER as token
from itertools import chain
def clean_song_text(lyrics):
    def clean_brackets(text):
        pattern = r'\[.*?\]'
        return re.sub(pattern,'', text)
    # find the first occurance of the word "Lyrics", and discard what's before that
    lyrics_start = lyrics.find('Lyrics') + len('Lyrics')
    lyrics = lyrics[lyrics_start:].lower()
    # cut out the end of the string (the word Embed and the number)
    # search for the number on the end and if it exists cut out from it
    if re.search(r'\d+', lyrics[::-1]):
        lyrics_end = re.search(r'\d+', lyrics[::-1]).span()[1]
    else:
        lyrics_end = 1
    lyrics = lyrics[:-lyrics_end]
    # should ignore anything in the square brackets
    lyrics = clean_brackets(lyrics)
    # remove interpunction
    tokenizer = nltk.WhitespaceTokenizer()
    lyrics = tokenizer.tokenize(lyrics)
    # lyrics.replace(r'.|,|!|?','')
    # # split on newlines and on spaces
    # lyrics = re.split(r' |\n', lyrics)
    # # remove empty strings
    # lyrics = list(filter(None, lyrics))
    return lyrics

def scrape_artist_songs(artist_list, max_songs=150, verbose=False, save_small_artists=True):
    genius = lyricsgenius.Genius(token, verbose=verbose)
    scrape_folder_prefix = '../scraped_data/Genius/'
    for artist_name in artist_list:
        artist = genius.search_artist(artist_name, max_songs=max_songs)
        # when the song search is empty it probably means that there are no songs in genius to scrape
        if not artist:
            continue
        albums = genius.search_albums(artist_name)
        album_names = [album['result']['name'] for album in albums['sections'][0]['hits']]
        # when the artist doesn't have 3 albums  or 15 songs don't consider them
        if len(album_names) < 3:
            print(f'{artist_name} has only {len(album_names)} albums')
            continue
        nr_of_songs = len(artist.songs)
        if nr_of_songs < 15:
            print(f'artist has only {nr_of_songs} songs')
            continue
        # Check the number of words in the songs. If there are below 30k save to the small artists
        songs_lyrics = [clean_song_text(song.lyrics) for song in artist.songs]
        word_count = count_words_in_lyrics(songs_lyrics)
        if word_count > 30000:
            with open(scrape_folder_prefix + artist_name, 'wb') as f:
                pickle.dump(artist, f)
        else:
            with open(scrape_folder_prefix + 'small_artists/' + artist_name, 'wb') as f:
                pickle.dump(artist, f)

def count_words_in_lyrics(songs):
    # better memory usage than len(list(chain(*songs))) i guess
    word_count = sum(1 for elem in chain(*songs))
    return word_count
