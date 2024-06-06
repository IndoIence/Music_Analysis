# %%
# markdown seemingly unused parts
import os
from pathlib import Path
from collections import defaultdict
import plwn

import spacy

nlp = spacy.load("pl_core_news_sm")


# %%
def load_data(datadir):
    data = []
    for fpath in datadir.iterdir():
        if not fpath.suffix == ".txt":
            continue
        with open(fpath) as f:
            text = " ".join(text.strip() for text in f.readlines())
        data.append((fpath.stem, nlp(text)))
    return data


# %% [markdown]
# data = load_data(Path('./data/raw'))

# %% [markdown]
# from transformers import MT5ForConditionalGeneration, T5Tokenizer
#
# def load_model(modelpath):
#     model = MT5ForConditionalGeneration.from_pretrained(modelpath)
#     tokenizer = T5Tokenizer.from_pretrained(modelpath)
#     return model, tokenizer

# %% [markdown]
# model, tokenizer = load_model("google/mt5-small")

# %%


# plwn.download()
# wn = plwn.load("./default_model")

# %%
# lex = wn.lexical_unit("pies", plwn.PoS.noun_pl, 2)
# print(lex)


# %%
class WordNet:

    pos_mapping = {
        "VERB": "CZASOWNIK",
        "NOUN": "RZECZOWNIK",
        "ADJ": "PRZYMIOTNIK",
        "ADV": "PRZYSŁÓWEK",
    }

    def __init__(self, wn_path):
        self.wn = plwn.load(wn_path)
        self.senses = self.wn.lexical_units()

        self.index_by_lemma = defaultdict(set)

        for lexicalunit in self.wn.lexical_units():
            self.index_by_lemma[
                (lexicalunit.lemma, lexicalunit.pos.short_value.upper())
            ].add(lexicalunit)

    def get_senses(self, lemma=None, pos=None):
        if lemma and pos:
            # adding spacy name handling
            plwn_pos = self.pos_mapping.get(pos, pos)
            return self.index_by_lemma[(lemma, plwn_pos)]
        return self.senses

    def get_sense_by_id(self, lemma, synid):
        try:
            synset = self.wn.synset_by_id(synid)
        except plwn.exceptions.SynsetNotFound:
            return
        try:
            return next(
                iter(
                    (synset, lexicalunit)
                    for lexicalunit in synset.lexical_units
                    if lexicalunit.lemma == lemma
                )
            )
        except StopIteration:
            return

    def get_hyperonyms(self, synset):
        return synset.hyperonyms


# %%
# wordnet = WordNet('./data/plwn-15012022.db')
wordnet = WordNet("./default_model")

# %% [markdown]
# wn = plwn.load('./data/plwn-15012022.db')


# %%
def polysemy_stats(wordnet, data):
    stats = {}
    for fname, doc in data:
        tokens = (token for sent in doc.sents for token in sent)
        for token in tokens:
            lemma, pos = token.lemma_, token.pos_
            senses = wordnet.get_senses(lemma, pos)
            stats[(lemma, pos)] = len(senses)
    return stats


# %%
# import one of the pl artits
# concat all the songs and put them to a doc

import os
from utils import CONFIG, get_artist
import pickle

artist = get_artist("Tede")
# %%
all_songs = "\n".join(
    song.lyrics for song in artist.get_limit_songs(only_art=True, limit=2e4)
)
cur_doc = nlp(all_songs)

# %%
data = [("dummy_name", cur_doc)]
stats_with_zeros = polysemy_stats(wordnet, data)
stats = {k: v for k, v in stats_with_zeros.items() if v}


# %%
sorted_stats = sorted(stats.items(), key=lambda i: i[1], reverse=True)
sorted_stats

# %%
type(None)

# %%
polysemy_stats(wordnet, data)
# ujednoznacznić i policzyć w ilu znaczeniach dane słowo się pojawia w korpusie

"""
dokument                    słowo           domena                                      znaczenie   lemat   ile_sensów_w_korpusie   abstrakcyjny_hiperonim  aspekty wydźwięk    emocje  wartości_fundamentalne
OPS_PSL_25.01.2018_O_M_m	zlecenie	    zdarzenia
OPS_PSL_25.01.2018_O_M_m	Sądu	        grupy ludzi i rzeczy
OPS_PSL_25.01.2018_O_M_m	Rejonowego	    przymiotniki relacyjne (rzeczownikowe)
OPS_PSL_25.01.2018_O_M_m	Wydziału	    związek miedzy ludźmi, rzeczami lub ideami
OPS_PSL_25.01.2018_O_M_m	Karnego	        przymiotniki jakościowe
OPS_PSL_25.01.2018_O_M_m	dnia	        czas i stosunki czasowe
OPS_PSL_25.01.2018_O_M_m	obserwacji	    związane z myśleniem
OPS_PSL_25.01.2018_O_M_m	sądowo	        przymiotniki relacyjne (rzeczownikowe)
OPS_PSL_25.01.2018_O_M_m	psychiatrycznej	przymiotniki relacyjne (rzeczownikowe)
OPS_PSL_25.01.2018_O_M_m	terminie	    czas i stosunki czasowe
OPS_PSL_25.01.2018_O_M_m	Psychiatrii	    miejsca i umiejscowienie
OPS_PSL_25.01.2018_O_M_m	Sądowej	        przymiotniki relacyjne (rzeczownikowe)
OPS_PSL_25.01.2018_O_M_m	Instytutu	    związek miedzy ludźmi, rzeczami lub ideami
"""

# rejestry: wyciągnąć z nowego dumpa


# %%
def disamb_fwns(wordnet, data):
    for fname, doc in data:
        tokens = (token for sent in doc.sents for token in sent)
        for token in tokens:
            lemma, pos = token.lemma_, token.pos_
            senses = wordnet.get_senses(lemma, pos)

            if not senses:
                continue

            fwns = sorted(senses, key=lambda sense: sense.variant)[0]
            yield fname, token, fwns.domain.value


# %%
with open("domains.txt", "w") as ofile:
    for fname, token, domain in disamb_fwns(wordnet, data):
        ofile.write(f"{fname} {token} {domain}\n")

# %% [markdown]
# ### *2nd Approach*
# - Use disambiguation toolkit and apply it to CCL data
# - Generate new stats based on disambiguation results

# %%
# from xml.dom.minidom import parse


# def load_data(datadir):
#     data = []
#     for fpath in datadir.iterdir():
#         if not fpath.suffix == ".xml":
#             continue
#         if ".wsd" not in fpath.stem:
#             continue
#         yield (fpath.stem, parse(str(fpath)))


# %% [markdown]
# def load_data(datadir: Path):
#     for fpath in datadir.iterdir():
#         if fpath.suffix !=


# %%
def get_pos(ctag):
    "From NKJP Tagset to regular tagset"
    ctag = ctag.split(":")[0]

    nouns = {"subst", "depr", "brev"}
    verbs = {
        "fin",
        "bedzie",
        "praet",
        "impt",
        "inf",
        "pcon",
        "pant",
        "imps",
        "winien",
        "pred",
        "pact",
        "ppas",
        "pred",
        "ger",
    }
    adjs = {"adj", "adja", "adjp", "adjc"}
    advs = {"adv"}

    if ctag in nouns:
        return "2"
    elif ctag in adjs:
        return "4"
    elif ctag in verbs:
        return "1"
    elif ctag in advs:
        return "3"

    return None


# %%
def get_lemma(base, props):
    """Get token's lemma. The token might be a part of multiword expression,
    so we have to check token's properties.
    """
    try:
        return next(
            iter(
                p.firstChild.nodeValue
                for p in props
                if p.getAttribute("key") == "mwe_base"
            )
        )
    except StopIteration:
        pass
    try:
        return base.firstChild.nodeValue
    except Exception:
        pass


# %%
def get_sense(props):
    try:
        return next(
            iter(
                p.firstChild.nodeValue
                for p in props
                if p.getAttribute("key") == "sense:ukb:syns_id"
            )
        )
    except Exception:
        pass


# %%
def preproc(fpath, fdom):

    for token in fdom.getElementsByTagName("tok"):
        lex = token.getElementsByTagName("lex")[0]

        base, ctag = lex.childNodes
        pos = get_pos(ctag.firstChild.nodeValue)

        props = list(token.getElementsByTagName("prop"))

        lemma = get_lemma(base, props)
        sense = get_sense(props)

        if sense:
            yield (lemma, pos, sense)


# %%
def get_hypernym(synset):
    try:
        return next(
            iter(
                rel_obj
                for rel_type, rel_obj in synset.related_pairs()
                if rel_type.name == "hiponimia"
            )
        )
    except Exception:
        pass


# %%
def decode_pos(pos):
    if pos == "2":
        return "noun"
    elif pos == "4":
        return "adj"
    elif pos == "1":
        return "verb"
    elif pos == "3":
        return "adv"


# %%
sense = next(iter(wordnet.get_senses("dom", "NOUN")))
sense.usage_notes

# %%
data = load_data(Path("./data/preproc"))
header = [
    "filename",
    "lemma",
    "pos",
    "verb_aspect",
    "num_senses",
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
    for fname, fdom in data:
        for row in preproc(fname, fdom):
            lemma, pos, sense = row

            pos = decode_pos(pos)
            num_senses = len(wordnet.get_senses(lemma, pos.upper()))

            try:
                synset, sense = wordnet.get_sense_by_id(lemma, sense)
            except Exception:
                continue

            hypernym_1st = get_hypernym(synset)
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
                sense.emotion_markedness.name if sense.emotion_markedness else "null"
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

            ofile.write(
                f"{fname}\t{lemma}\t{pos}\t{aspect}\t{num_senses}\t{sentiment}\t{emotions}\t{valuations}\t{synset.short_str()}\t{sense.domain.value}\t{hypernym_1st}\t{hypernym_2nd}\t{hypernym_3rd}\t{sense.usage_notes}\n"
            )


# %%
import pandas as pd

# %%
data = pd.read_csv("./wordnet-analysis.tsv", sep="\t")

# %%
data.columns

# %%
data["domain"].value_counts()

# %%
domain_stats = data.groupby(["filename", "domain"]).size().unstack(fill_value=0)

# %%
domain_stats

# %%
sentiment_stats = data.groupby(["filename", "sentiment"]).size().unstack(fill_value=0)

# %%
sentiment_stats

# %%
import plwn

wn = plwn.load("default_model")
# wn = plwn.load('./data/plwn-15012022.db')

# %%
from collections import defaultdict

lexicalunits = defaultdict(list)
for lu in wn.lexical_units():
    lexicalunits[lu.pos.name].append(lu)

# %%
lu = lexicalunits["verb"][0]

# %%
lu
