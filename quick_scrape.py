# %%
from classes.scraper import GeniusScraper, RedirectPrints
from utils import CONFIG
from pathlib import Path

artists_names = [
    f_name.stem.replace("_", " ")
    for f_name in Path(CONFIG["korpusomat_files_path"]).iterdir()
    if f_name.is_file()
]
# %%
scraper = GeniusScraper()
ouptut_file = CONFIG["scraper_prints"]
with open(ouptut_file, "a", buffering=1) as f, RedirectPrints(f):
    print("Printing to file :3 \n")
    scraper.scrape(artists_names[236:])
    scraper.process_artists()


# %%
