# # %%
# from utils import CONFIG, get_artist
# from classes.LyricsAnalyzer import LyricsAnalyzer
# from tqdm import tqdm
# from pathlib import Path

# # import spacy
# from collections import Counter

# # nlp = spacy.load("pl_core_news_sm")
# la = LyricsAnalyzer()
# counters = la.counters
# #
# print(counters)

# # %%
# la.save_counters(la.counters, 30000)
# # %%
# # for every counter sum the vals and plot it in a bar plot
# # %%
# for k,c in counters.items():
#     print(k)
#     print(sum(c.values()))
#     print(len(c))
# %%

# %%
import plwn.exceptions
from utils import get_artist, CONFIG, get_wsd_data
from classes.WordNet import WordNet
import plwn


def get_hypernym(lex_unit):
    try:
        return next(
            iter(
                rel_obj
                for rel_type, rel_obj in lex_unit.related_pairs()
                if rel_type.name == "hiponimia"
            )
        )
    except Exception:
        pass


# %%
wn = WordNet("./default_model")

# %%


# %%
# data = load_data(Path('./data/preproc'))
data = get_wsd_data()
header = [
    "filename",
    "lemma",
    "pos",
    "verb_aspect",
    "sentiment",
    "emotions",
    "valuations",
    "synset",
    "domain",
    "1st-hypernym",
    "2nd-hypernym",
    "3rd-hypernym",
    "register",
]
with open("wordnet-analysis.tsv", "w") as ofile:
    ofile.write("\t".join(header))
    ofile.write("\n")
    for fname, songs in data:
        for song in songs:
            for word in song:
                suggestions = word["suggestions"]
                # there are up to three suggestions (can there be 0 if they are too far away?)
                # choosing one with the highest score
                if not suggestions:
                    continue
                uuid = next(iter(suggestions))
                try:
                    sense = wn.wn.lexical_unit_by_id(uuid)
                except plwn.exceptions.LexicalUnitNotFound:
                    continue
                hypernym_1st = get_hypernym(sense)
                hypernym_2nd = None
                hypernym_3rd = None
                if hypernym_1st:
                    hypernym_2nd = get_hypernym(hypernym_1st)
                if hypernym_2nd:
                    hypernym_3rd = get_hypernym(hypernym_2nd)

                aspect = sense.verb_aspect if sense.verb_aspect else "null"
                hypernym_1st = "null" if not hypernym_1st else hypernym_1st.short_str()
                hypernym_2nd = "null" if not hypernym_2nd else hypernym_2nd.short_str()
                hypernym_3rd = "null" if not hypernym_3rd else hypernym_3rd.short_str()

                sentiment = (
                    sense.emotion_markedness.name
                    if sense.emotion_markedness
                    else "null"
                )
                emotions = (
                    "|".join(
                        emotion.name
                        for emotion in sense.emotion_names
                        if sense.emotion_names
                    )
                    if sense.emotion_names
                    else "null"
                )
                valuations = (
                    "|".join(
                        value.name
                        for value in sense.emotion_valuations
                        if sense.emotion_valuations
                    )
                    if sense.emotion_valuations
                    else "null"
                )
                lemma = sense.lemma
                pos = sense.pos.short_value
                # num_senses = len(wn.get_senses(lemma, pos))
                ofile.write(
                    f"{fname}\t{lemma}\t{pos}\t{aspect}\t{sentiment}\t{emotions}\t{valuations}\t{sense.synset.short_str()}\t{sense.domain.name}\t{hypernym_1st}\t{hypernym_2nd}\t{hypernym_3rd}\t{sense.usage_notes}\n"
                )


# %%


example_lemma = "pies"
example_pos = "NOUN"
# %%

# %%
# if in already know the synset id i can use this method
# but at first i only have the lemma and lemma id
# how can i get synset id from lemma id?
# i can get lexical unit by id
# then from that get synset and from that synset all synonyms or sth
example_id = "d6edc687-aac4-11ed-aae5-0242ac130002"
sense = wn.wn.lexical_unit_by_id(example_id)
print(sense)
# synset = unit.synset.lexical_units
hypernyms = wn.get_hypernym(sense)
print(hypernyms)


# if in already know the synset id i can use this method
# but at first i only have the lemma and lemma id
# how can i get synset id from lemma id?
# i can get lexical unit by id
# then from that get synset and from that synset all synonyms or sth
example_id = "d6edc687-aac4-11ed-aae5-0242ac130002"
sense = wn.wn.lexical_unit_by_id(example_id)
print(sense)
# synset = unit.synset.lexical_units
hypernyms = wn.get_hypernym(sense)
print(hypernyms)


# %%
