# %%
from utils import get_biggest_by_lyrics_len, get_artist
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import numpy as np
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt


# %%
data_list = []
biggest_artists = get_biggest_by_lyrics_len(10)
# %%
for art in biggest_artists:
    print(art.name)
    print(art.lyrics_len_only_art)
for artist in biggest_artists:
    songs = artist.get_limit_songs(3e4, only_art=True)
    for song in songs:
        data_list.append({"texts": song.get_clean_song_lyrics(), "artist": artist.name})

data = pd.DataFrame(data_list)
# %%
X_train, X_test, y_train, y_test = train_test_split(
    data["texts"], data["artist"], test_size=0.2, random_state=42
)

# %%
# Initialize the vectorizer
# vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer()

# Fit and transform the training data, transform the test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Initialize the classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vec, y_train)


# Make predictions on the test data
y_pred = classifier.predict(X_test_vec)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
cm_count_not_norm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
cm_count = (
    cm_count_not_norm.astype("float") / cm_count_not_norm.sum(axis=1)[:, np.newaxis]
)

sns.heatmap(
    cm_count,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classifier.classes_,
    yticklabels=classifier.classes_,
    ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.show()

# %%
