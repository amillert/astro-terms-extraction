import os
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

# docs = []
# 
# for i in range(1, 21):
#   print(i)
#   with open(os.path.join("./data", f"art{i}.txt")) as f:
#     doc = f.read()
#     tokens = list(filter(
#         lambda l: l.isalpha() and l not in stopwords.words("english"),
#         map(lambda l: l.lower(), word_tokenize(doc))))
#     docs.append(" ".join(tokens))
# 
# with open("out-astro", "w") as fout:
#   for doc in docs:
#     fout.write(doc)
#     fout.write("\n")

# Load:

docs = list(map(lambda l: l.strip(), open("out-astro", "r").readlines()))

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

from pprint import pprint

pprint(termScores)
print(extracted_terms)

exit(12)

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