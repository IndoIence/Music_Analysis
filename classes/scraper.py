import lyricsgenius
import pickle
import logging
from utils import GENIUS_CONFIG, sanitize_name
from pathlib import Path
from requests import Timeout
import sys

# TODO: add handling of 0 width space in the name
# TODO: move token to env variable


class RedirectPrints:
    def __init__(self, new_stdout, new_stderr=None):
        if new_stderr is None:
            new_stderr = new_stdout
        self.new_stdout = new_stdout
        self.new_stderr = new_stderr
        self.old_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.new_stdout
        sys.stderr = self.new_stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self.new_stdout.flush()  # Ensure the buffer is flushed
        self.new_stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class GeniusScraper:
    def __init__(self, token=GENIUS_CONFIG["TOKEN"]):
        self.genius = lyricsgenius.Genius(
            token,
            verbose=True,
            retries=GENIUS_CONFIG["retries"],
            timeout=GENIUS_CONFIG["timeout"],
            sleep_time=GENIUS_CONFIG["sleep_time"],
        )
        self.urls = self.load_urls()
        # self.genius.remove_section_headers = True
        # self.genius.skip_non_songs = True
        # self.genius.excluded_terms = ["(Remix)", "(Live)"]

        logging.basicConfig(
            filename=GENIUS_CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        # try:
        #     raise Timeout("Halko\nTesting timeout error messages")
        # except Exception as e:
        #     logging.error(e)

    def load_urls(self, path: Path = Path(GENIUS_CONFIG["urls_file"])):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def save_urls(self, path: Path = Path(GENIUS_CONFIG["urls_file"])):
        with open(path, "w") as f:
            for url in sorted(self.urls):
                f.write(url.lower() + "\n")

    def scrape(self, artists_list: list[str], max_songs: int = 2000):
        for artist_name in artists_list:
            artist = self.scrape_artist(artist_name, max_songs)
            if not artist:
                logging.info(f"Nothing found for: {artist_name}")
                continue
            if names_very_different(artist_name, artist.name):
                message = f"Difference in names: Name from lastFM: {artist_name},\nName from Genius {artist.name}"
                print(message)
                logging.warning(message)
            self.save_artist(artist, artist_name)
            logging.info(f"Scraped {artist}")
            self.urls.append(artist.url)
            self.save_urls()

    def save_artist(
        self,
        artist,
        artist_name: str = "",
        folder: Path = Path(GENIUS_CONFIG["save_path"]),
        suffix: str = ".pkl",
    ):
        if artist_name == "":
            artist_name = artist.name
        if names_very_different(artist_name, artist.name):
            artist_name = sanitize_name(artist_name + "(" + artist.name + ")")
        else:
            artist_name = sanitize_name(artist.name)
        artist_name += suffix
        with open(folder / artist_name, "wb") as f:
            pickle.dump(artist, f)

    def scrape_artist(self, artist_name: list[str], max_songs: int = 2000):
        # first try to find the artist in the urls
        logging.info(f"Beginning search for {artist_name}")
        artist = None
        try:
            artist = self.genius.search_artist(artist_name, max_songs=0)
        except TimeoutError as err:
            message = f"timeout for {artist_name}"
            logging.error(f"{message} {err}")
            # TODO: save to timeout file the name of unprocessed artist
        except Exception as e:
            message = f"Error for {artist_name}"
            logging.exception(f"{message} {e}")

        if artist is not None and artist.url.lower() in self.urls:
            logging.info(f"Artist {artist_name} already scraped. URL: {artist.url}")
            return None

        try:
            artist = self.genius.search_artist(artist_name, max_songs=max_songs)
        except TimeoutError as err:
            message = f"timeout for {artist_name}"
            logging.error(f"{message} {err}")
            # TODO: save to timeout file the name of unprocessed artist
        except Exception as e:
            message = f"Error for {artist_name}"
            logging.error(f"{message} {err}")
        return artist


def names_very_different(lastfm_name, genius_name):
    # if no substring longer than 3 is from the original name, then that name is different
    # For the name Tau Genius returns 2 Chainz (???)
    # want to see if in many cases name received from genius is completely different
    # if the names are short don't consider
    if len(lastfm_name) < 3 or len(genius_name) < 3:
        return False
    for i in range(len(lastfm_name) - 3):
        if lastfm_name[i : i + 3].lower() in genius_name.lower():
            return False
    return True
