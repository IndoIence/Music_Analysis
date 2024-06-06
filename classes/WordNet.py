import plwn
from collections import defaultdict
from plwn.storages.sqlite import PLWordNet


class WordNet:

    pos_mapping = {
        "VERB": "CZASOWNIK",
        "NOUN": "RZECZOWNIK",
        "ADJ": "PRZYMIOTNIK",
        "ADV": "PRZYSŁÓWEK",
    }

    def __init__(self, wn_path):
        self.wn: PLWordNet = plwn.load(wn_path)
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
        except plwn.exceptions.SynsetNotFound:  # type: ignore
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

    def get_hypernym(self, lex_unit):
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
