import os
from numpy.lib.function_base import extract
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize
from nltk.corpus import stopwords
# from nltk.util import everygrams

import numpy as np

# nlp = spacy.load("en_core_web_sm")
# import nltk
# nltk.download("punkt")
# nltk.download("stopwords")

# Process and save:

docs = []

for i in range(1, 21):
  print(i)
  with open(os.path.join("./data", f"art{i}.txt")) as f:
    doc = f.read()
    tokens = list(filter(
        lambda l: l.isalpha() and l not in stopwords.words("english"),
        map(lambda l: l.lower(), word_tokenize(doc))))
    docs.append(" ".join(tokens))

with open("out-astro", "w") as fout:
  for doc in docs:
    fout.write(doc)
    fout.write("\n")

# Load:

# docs = list(map(lambda l: l.strip(), open("out-astro", "r").readlines()))

# TF-IDF based terms extraction

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 4))

tfidf = tfidf_vectorizer.fit_transform(docs)
feature_names = np.array(tfidf_vectorizer.get_feature_names())

THRESHOLD = 0.01

termScores = [(feature_names[col], tfidf[row, col])
              for row, col in zip(*tfidf.nonzero())]
termScoresFiltered = list(filter(
    lambda l: l[1] > THRESHOLD and len(l[0]) > 1,
    termScores))

extracted_terms = set(map(lambda l: l[0], termScoresFiltered))

with open("astro-terms", "w") as fout:
  for term in extracted_terms:
    fout.write(f"{term}\n")
