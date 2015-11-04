from code import interact
from collections import deque
import string
from collections import defaultdict
from ast import literal_eval
from itertools import izip
import nltk.data
from gensim.models import doc2vec

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

def make_token_list(review):
    global tokenizer
    raw_sentances = tokenizer.tokenize(review.strip())
    token_list = [token for sentance in raw_sentances
                 for token in make_wordlist(sentance) 
                 if len(sentance) > 0]
    return token_list

def data_to_documents(fname):
    documents = deque()
    count = 0
    for data in yield_data(fname):
        if count % 10000 == 0: print count/10000
        if count >= 750000: break
        review_text = data['reviewText']
        new_token_list = make_token_list(review_text)
        label = count
        new_document = doc2vec.LabeledSentence(words=new_token_list, tags=[count])
        documents.append(new_document)
        count += 1
    return documents

num_features = 500
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-5

print "Building document list..."
documents = data_to_documents('train.json')
print "Training model..."
model = doc2vec.Doc2Vec(documents, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=False)
model_name = "Doc2Vec_500features_40minwords_10context"
model.save(model_name)
