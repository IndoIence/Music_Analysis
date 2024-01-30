import logging
import pickle
import re
from itertools import chain
from pathlib import Path
import lyricsgenius
import nltk
from tqdm import tqdm
from utils import CONFIG, MyArtist
config_G = CONFIG["Genius_scraping"]

logging.basicConfig(
    filename=config_G["GENIUS_LOGFILE"],
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
    # remove interpunctionbrack
    tokenizer = nltk.WhitespaceTokenizer()
    lyrics = tokenizer.tokenize(lyrics)
    # lyrics.replace(r'.|,|!|?','')
    # # split on newlines and on spaces
    # lyrics = re.split(r' |\n', lyrics)
    # # remove empty strings
    # lyrics = list(filter(None, lyrics))
    print(lyrics)
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

### TODO: often genius outputs a well known english artist ie (Ariana Grande, Chief Keef) find a way to mitigate those
### TODO: For the name Tau Genius returns 2 Chainz (???)
def scrape_artists_songs(artist_list: list[str]):
    urls = open(config_G["GENIUS_URLS_PATH"], 'r').read().strip().split()
    for artist_id, artist_name in enumerate(tqdm(artist_list)):
        artist = scrape_artist_songs(artist_name)
        # check if already saved
        if artist.url in urls:
            logging.info(f"Artist url {artist.url} already saved")
            continue
        out_path = Path(config_G["save_path"])
         # save to the general file
        # in the general directory the file names are without .artPkl
        # -> for now only MyArtists have that
        save_artist_to_pkl(out_path, artist, artist_name)
        my_artist = MyArtist(artist)
        my_artist.get_artist_language()      

        # save to specific pl / non pl directories
        if my_artist.language == "pl":
            out_path2 = Path(CONFIG['artists_pl_path'])
        else:
            out_path2 = Path(CONFIG['artists_non_pl_path'])
        file_name = sanitize_art_name(my_artist.name)
        save_artist_to_pkl(out_path2, artist, file_name)
            
def sanitize_art_name(name:str)-> str:
    name.replace(' ', '_').replace(".", "_")
    return name + '.artPkl'
        



def count_words_in_lyrics(songs):
    # better memory usage than len(list(chain(*songs))) i guess
    word_count = sum(1 for elem in chain(*songs))
    return word_count

def scrape_artist_songs(artist_name: str):
    genius = lyricsgenius.Genius(access_token=config_G["GENIUS_BEARER"],
                                 verbose=config_G["verbose"],
                                 timeout=config_G["GENIUS_TIMEOUT"],
                                 sleep_time=config_G["GENIUS_SLEEP_TIME"])
    logging.info(f"Beginning search for {artist_name}")
    try:
        artist = genius.search_artist(artist_name, 
                                      max_songs=config_G['max_songs'], 
                                      include_features=config_G['features'])
    except TimeoutError as err:
        logging.error(f'timeout for {artist_name}', err)
        return
    except Exception as e:
        logging.error(f'Different error for {artist_name}', e)
        return
    # when the song search is empty it probably means that there are no songs in genius to scrape
    if not artist:
        logging.info(f"Nothing found for: {artist_name}")
        return
    if names_very_different(artist_name, artist.name):
        # changing name here is a bad practice and can be problematic in the future, but it is late and i am lazy to rewrite more
        artist_name = artist_name + '(' + artist.name + ')'
        message = f'Difference in names: Name from lastFM: {artist_name},\nName from Genius {artist.name}'
        print(message)
        logging.warning(message)
    albums = genius.search_albums(artist_name)
    album_names = [album['result']['name'] for album in albums['sections'][0]['hits']]

    # when the artist doesn't have 3 album or 15 songs
    if len(album_names) < 3:
        print(f'{artist_name} has only {len(album_names)} albums')
        logging.info(f'{artist_name} has only {len(album_names)} albums')
    nr_of_songs = len(artist.songs)
    if nr_of_songs < 15:
        print(f'artist has only {nr_of_songs} songs')
        logging.info(f'artist has only {nr_of_songs} songs')
        # Check the number of words in the songs. If there are below 30k save to the small artists
        #songs_lyrics = [clean_song_text(song.lyrics) for song in artist.songs]
        #word_count = count_words_in_lyrics(songs_lyrics)
    return artist


def save_artist_to_pkl(out_path: Path, artist, artist_name: str):
    # before saving check if there are any / or ? in the name -> if so replace them
        if '/' in artist_name or '?' in artist_name:
            artist_name = artist_name.replace('/', ' ').replace('?', ' ')
            logging.info(
                f'Changed name of the artist to {artist_name} because of the slashes/question marks in the name')
        with open(out_path / artist_name, 'wb') as f:
            pickle.dump(artist, f)
        logging.info("Ending search for: {}".format(artist_name))