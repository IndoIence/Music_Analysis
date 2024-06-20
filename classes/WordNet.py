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

    def get_info_from_lex_unit(self, sense, str_format=False):

        hypernym_1st = self.get_hypernym(sense)
        hypernym_2nd = None
        hypernym_3rd = None
        if hypernym_1st:
            hypernym_2nd = self.get_hypernym(hypernym_1st)
        if hypernym_2nd:
            hypernym_3rd = self.get_hypernym(hypernym_2nd)

        hypernym_1st = None if not hypernym_1st else hypernym_1st.short_str()
        hypernym_2nd = None if not hypernym_2nd else hypernym_2nd.short_str()
        hypernym_3rd = None if not hypernym_3rd else hypernym_3rd.short_str()

        aspect = sense.verb_aspect.name if sense.verb_aspect else None
        sentiment = sense.emotion_markedness.name if sense.emotion_markedness else None
        emotions = [
            emotion.name for emotion in sense.emotion_names if sense.emotion_names
        ]

        valuations = [
            value.name for value in sense.emotion_valuations if sense.emotion_valuations
        ]

        lemma = sense.lemma
        pos = sense.pos.short_value
        if str_format:
            emotions = "|".join(emotions)
            valuations = "|".join(valuations)
        return {
            "lemma": lemma,
            "pos": pos,
            "verb_aspect": aspect,
            "sentiment": sentiment,
            "emotions": emotions,
            "valuations": valuations,
            "synset": sense.synset.short_str(),
            "domain": sense.domain.name,
            "hypernym_1st": hypernym_1st,
            "hypernym_2nd": hypernym_2nd,
            "hypernym_3rd": hypernym_3rd,
            # "register": sense.usage_notes, kurwa to daje mi tuple a nie stringa
        }
