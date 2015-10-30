from code import interact
import string
from collections import defaultdict
from ast import literal_eval
from itertools import izip
import nltk.data
from gensim.models import word2vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
punctuation = set(string.punctuation)

def yield_data(fname):
    for l in open(fname):
        yield literal_eval(l)

def make_wordlist(sentance):
    r = ''.join([c for c in sentance if not c in punctuation])
    word_list = r.split()
    return word_list

def make_sentance_list(review):
    global tokenizer
    raw_sentances = tokenizer.tokenize(review.strip())
    sentances = [make_wordlist(sentance) 
                 for sentance in raw_sentances
                 if len(sentance) > 0]
    return sentances

def list_review_sentances(data):
    sentances = []
    for d in data:
        review_text = d['reviewText']
        sentances.extend(make_sentance_list(review_text))
	return sentances

num_features = 500
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print "Training model..."
sentances = list_review_sentances(yield_data('train.json'))
print sentances[0]
model = word2vec.Word2Vec(sentances, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=True)
model_name = "500features_40minwords_10context"
model.save(model_name)
