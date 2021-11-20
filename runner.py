import os
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import everygrams

import numpy as np

# nlp = spacy.load("en_core_web_sm")
# nltk.download("punkt")
# nltk.download("stopwords")

# astronomy topic
docs = []

for i in range(1, 21):
  with open(os.path.join("./data", f"art{i}.txt")) as f:
    doc = f.read()
    tokens = list(filter(
        lambda l: l.isalpha() and l not in stopwords.words("english"),
        map(lambda l: l.lower(), word_tokenize(doc))))
    docs.append(" ".join(tokens))

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
len(set(termScoresFiltered)), len(termScoresFiltered)

extracted_terms = set(map(lambda l: l[0], termScoresFiltered))

print(extracted_terms)

# prepare docs for classification
flattened = sorted(set([term for doc in docs for term in doc.split(" ")]))

tok2idx = {tok: idx for idx, tok in enumerate(flattened)}
tok2idx["<UNKNOWN>"] = len(tok2idx)
idx2tok = {v: k for k, v in tok2idx.items()}

# convert docs
docs_converted = [[tok2idx[tok] for tok in doc.split()] for doc in docs]

# convert extracted terms
extracted_converted = {tuple([tok2idx[tok] for tok in ngram.split()])
                       for ngram in extracted_terms}

# annotated = []
#
# for doc in docs_converted:
#     for extracted in extracted_converted:
#         if extracted
