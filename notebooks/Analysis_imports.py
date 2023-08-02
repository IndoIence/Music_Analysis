import logging
import pickle
import re
from itertools import chain

import lyricsgenius
import nltk

from config import GENIUS_BEARER as token
from config import LOGFILE, GENIUS_TIMEOUT, GENIUS_SLEEP_TIME

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_song_text(lyrics):
    def clean_brackets(text):
        pattern = r'\[.*?\]'
        return re.sub(pattern, '', text)

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


def names_very_different(lastfm_name, genius_name):
    # if no substring longer than 3 is from the original name, then that name is different
    # For the name Tau Genius returns 2 Chainz (???)
    # want to see if in many cases name received from genius is completely different
    # if the names are short don't consider
    if len(lastfm_name) < 3 or len(genius_name) < 3:
        return False
    for i in range(len(lastfm_name) - 3):
        if lastfm_name[i:i + 3].lower() in genius_name.lower():
            return False
    return True


### TODO: try except for timeouts and other errors
### TODO: often genius outputs a well known english artist ie (Ariana Grande, Chief Keef) find a way to mitigate those
### TODO: handling timout exceptions and HTTPERRORs
##
def scrape_artist_songs(artist_list, max_songs=2000, verbose=False, save_small_artists=True):
    genius = lyricsgenius.Genius(token, verbose=verbose, timeout=GENIUS_TIMEOUT, sleep_time=GENIUS_SLEEP_TIME)
    scrape_folder_prefix = '../scraped_data/Genius/'
    for artist_name in artist_list:
        logging.info(f"Beginning search for {artist_name}")
        try:
            artist = genius.search_artist(artist_name, max_songs=max_songs)
        except TimeoutError as err:
            logging.exception(err)
        except Exception as err:
            logging.exception(err)

        # when the song search is empty it probably means that there are no songs in genius to scrape
        if not artist:
            logging.info(f"Nothing found for: {artist_name}")
            continue
        if names_very_different(artist_name, artist.name):
            # changing name here is a bad practice and can be problematic in the future, but it is late and i am lazy to rewrite more
            last_fm_name = artist_name
            artist_name = artist_name + '(' + artist.name + ')'
            print(f'Name from lastFM: {last_fm_name},\nName from Genius {artist.name}')
            logging.warning(f'Name from lastFM: {artist_name}, Name from Genius {artist.name}')
        albums = genius.search_albums(artist_name)
        album_names = [album['result']['name'] for album in albums['sections'][0]['hits']]

        # when the artist doesn't have 3 albums  or 15 songs don't consider them
        if len(album_names) < 3:
            print(f'{artist_name} has only {len(album_names)} albums')
            logging.info(f'{artist_name} has only {len(album_names)} albums')
        nr_of_songs = len(artist.songs)
        if nr_of_songs < 15:
            print(f'artist has only {nr_of_songs} songs')
            logging.info(f'artist has only {nr_of_songs} songs')
        # Check the number of words in the songs. If there are below 30k save to the small artists
        songs_lyrics = [clean_song_text(song.lyrics) for song in artist.songs]
        word_count = count_words_in_lyrics(songs_lyrics)

        # before saving check if there are any / or ? in the name -> if so replace them
        if '/' in artist_name or '?' in artist_name:
            artist_name = artist_name.replace('/', ' ').replace('?', ' ')
            logging.info(
                f'Changed name of the artist to {artist_name} because of the slashes/question marks in the name')
        if word_count > 30000:
            with open(scrape_folder_prefix + artist_name, 'wb') as f:
                pickle.dump(artist, f)
        else:
            with open(scrape_folder_prefix + 'small_artists/' + artist_name, 'wb') as f:
                pickle.dump(artist, f)
        logging.info(f"Ending search for {artist_name}")


def count_words_in_lyrics(songs):
    # better memory usage than len(list(chain(*songs))) i guess
    word_count = sum(1 for elem in chain(*songs))
    return word_count
