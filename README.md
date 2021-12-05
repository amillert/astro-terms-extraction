## Astrophysics domain term extraction

### Source files:
- runner.py for documents preprocessing and tf-idf-based terms candidates extraction
- rule_based.py for rule-based terms candidates filtering
- lstm_BIO_tag.py for IOB tagging
- lstm_BIO_train.py for training IOB tagger model (used in `lstm_BIO_tag`)

### Dependencies:
- numpy
- nltk (with `nltk.download("stopwords")`)
- pytorch
- transformers