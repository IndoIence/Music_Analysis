import requests
import time
import logging
from config import LAST_FM_API_KEY, LAST_FM_USER_AGENT, LOGFILE_LASTFM
from dataclasses import dataclass

@dataclass
class Artysta:
    name: str
    url: str
    def __key(self):
         return self.name
    def __hash__(self):
        return hash(self.__key())
    def __eq__(self, other):
        if isinstance(other, Artysta):
            return self.__key() == other.__key()



url = 'http://ws.audioscrobbler.com/2.0/'
headers = {
    'user-agent': LAST_FM_USER_AGENT
}
def get_tag_artists(tag, API_KEY=LAST_FM_API_KEY, max_pages=30, page=1):
    all_artists = []
    params = {
        'tag': tag,
        'api_key': API_KEY,
        'method': 'tag.getTopArtists',
        'format': 'json',
        'page': page,
    }
    while True:

        if page % 10 == 0:
            print(f'Searching page nr {page}')
        try:
            response = requests.get(url=url, headers=headers, params=params)
        except Exception as e:
            logging.error(e)
            continue
        data = response.json()

        if 'error' in data:
            print(f"Error: {data['message']}")
            logging.error(f"Error for {params}", data['message'])
            continue
        else:
            artists_batch = [artist for artist in data['topartists']['artist']]
            all_artists.extend(artists_batch)

            if data['topartists']['@attr']['page'] == data['topartists']['@attr']['totalPages']:
                print('Reached max page')
                print('Number of pages searched', data['topartists']['@attr']['totalPages'])
                break
            if page == max_pages:
                break

            page += 1
            params['page'] = page
        # wait for the next request
        time.sleep(0.25)
    return all_artists
