# %%
# get data from wsd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_wsd_data, get_stopwords


def get_wsd_vals(no_stopwords=False, tresholds: tuple[int, int] | None = None):
    """wsd data shrunk down to tuple with (Artist_name, list(best_scores), list(words))"""
    if no_stopwords:
        stopwords = set(get_stopwords())
    all_scores: list[tuple[str, list[float], list[str]]] = []
    for a_name, songs in get_wsd_data():
        # print(a_name)
        scores: list[float] = []
        words: list[str] = []
        for wsd in songs:
            for word in wsd:
                if no_stopwords and word["word"].lower() in stopwords:
                    continue
                vals = [float(word["suggestions"][plwn_id]["score"]) for plwn_id in word["suggestions"]]
                m = max(vals)
                if tresholds and (tresholds[0] > m or m >= tresholds[1]):
                    continue
                scores.append(m)
                words.append(word["word"])
        all_scores.append((a_name, scores, words))
    return all_scores


with_stop = get_wsd_vals()
no_stop = get_wsd_vals(no_stopwords=True)


# %%
for (a_name, scores, _), (a_name1, scores1, _) in zip(with_stop, no_stop):
    plt.hist(scores, bins=10)
    plt.hist(scores1, bins=10)
    plt.title(a_name)
    plt.legend(["stop", "no_stop"])
    plt.show()
    # %%
for a_name, scores, _ in no_stop:
    plt.hist(scores, bins=10)
    plt.title(a_name)
    plt.show()
# %%

# %%
small_score = get_wsd_vals(tresholds=(-80, -40))
big_score = get_wsd_vals(tresholds=(40, 60), no_stopwords=True)
for a_name, scores, _ in small_score:
    plt.hist(scores, bins=10)
    plt.title(a_name)
    plt.show()
# %%
# Peja duża pewność
artist_ind = 1
peja_name, scores, words = big_score[artist_ind]
print(f"{a_name} pewne")
for s, w, _ in zip(scores, words, range(20)):
    print(s, w)

peja_name, scores, words = small_score[artist_ind]
print(f"{a_name} niepewne")
for s, w, _ in zip(scores, words, range(20)):
    print(s, w)
# %%
# the same shit with removed stopwords

for a_name, scores, _ in no_stop:
    plt.hist(scores, bins=10)
    plt.title(a_name)
    plt.show()
# %%
print(no_stop[0])
# %%
avg_stop = np.average(with_stop[1][1])
std_stop = np.std(with_stop[1][1])
avg_no_stop = np.average(no_stop[1][1])
std_no_stop = np.std(no_stop[1][1])
# %%
print(len(with_stop[1][1]), len(no_stop[1][1]))
print(avg_no_stop, std_no_stop)
print(avg_stop, std_stop)

# %%
