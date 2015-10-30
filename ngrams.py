from code import interact
import string
from collections import defaultdict
from ast import literal_eval
from itertools import izip

def yield_data(fname):
    for l in open(fname):
        yield literal_eval(l)

word_dict = defaultdict(int)
bigram_dict = defaultdict(int)
trigram_dict = defaultdict(int)

def find_ngrams(fname):
    i = 0
    punctuation = set(string.punctuation)
    for datum in yield_data(fname):
        i += 1
        if i % 10000 == 0:
            print i / 10000
        r = ''.join([c for c in datum['reviewText'].lower() if not c in punctuation])
        words = r.split()
        for w1, w2, w3 in izip(words, words[1:], words[2:]):
            word_dict[w1] += 1
            #bigram_dict[' '.join([w1, w2])] += 1
            #trigram_dict[' '.join([w1, w2, w3])] += 1

find_ngrams('train.json')

print len(word_dict), len(bigram_dict), len(trigram_dict)
interact(local=locals())
