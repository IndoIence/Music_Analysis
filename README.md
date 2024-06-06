Scraping LastFM for names of polish artist:
Artist from LastFM:
* with "Polish Hip Hop" tag and equivalent ("Polski rap" etc.)
* with both Polish and Hip Hop (Rap) tags and equivalent


Scraping lyrics from Genius using thier LastFM names:
* utilizing lyricsgenius
* language detection of description of an artist to filter out non polish ones
* splitting artist with above and below 30k total words
TODO:
Tf-idf analysis
Cosine Similarity
t-SNE

Naming scheme for artist:
pickled into artist_name.artPkl

Current pipeline for adding a new artist:
scrape_new_artits -> checks if has been seen in the urls and adds if it is new
checks the language of the artists and saves two copies. One in directory for all artists and one in direcotry for polish or nonpolish artists
* polish artists all lyrics are exported to the korpusomat directory as .txt

LyricsAnalyzier
functionality for tfids and soon more
