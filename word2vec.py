from code import interact
import yaml
from collections import deque
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
replace_punctuation = string.maketrans(string.punctuation,
                                       ' '*len(string.punctuation))


def yield_data(fname):
    for l in open(fname):
        yield literal_eval(l)

def make_wordlist(sentance):
    sans_punct = sentance.translate(replace_punctuation)
    word_list = sans_punct.split()
    return word_list

def make_sentance_list(review):
    global tokenizer
    raw_sentances = tokenizer.tokenize(review.strip())
    sentances = [make_wordlist(sentance) 
                 for sentance in raw_sentances
                 if len(sentance) > 0]
    return sentances

def list_review_sentances(fname):
    sentances = deque()
    count = 0
    for data in yield_data(fname):
        count += 1
        if count % 10000 == 0: print count/10000
        review_text = data['reviewText']
        new_sentances = make_sentance_list(review_text)
        sentances.extend(new_sentances)
    return sentances

num_features = 500
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print "Building sentance list..."
sentances = list_review_sentances('train.json')
with open('sentances.yml', 'w') as infile:
    yaml.dump(sentances, infile)
print "Training model..."
model = word2vec.Word2Vec(sentances, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=True)
model_name = "500features_40minwords_10context"
model.save(model_name)
