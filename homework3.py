import numpy
import string
import yaml
from ast import literal_eval
from code import interact
import scipy.optimize
from collections import deque
import gc

def line_generator(fname, start, end):
    i = 0
    for line in open(fname):
        if i >= start and i <= end:
            yield literal_eval(line)
        i += 1
        if i > end: break

def load_fields(fname, fields, start, end):
    helpful = []
    for line in line_generator(fname, start, end):
        line_dict = dict((field, line[field]) for field in fields)
        helpful.append(line_dict)
    return helpful

fields = ['helpful', 'reviewText', 'rating']
#data = load_fields('train.json', fields)
train_data = load_fields('train.json', fields, 0, 100000)
valid_data = load_fields('train.json', fields, 900000, 1000000)

train_features = [{'nHelpful':rev['helpful']['nHelpful'],
                   'outOf':rev['helpful']['outOf'],
                   'ratio':(float(rev['helpful']['nHelpful'])/rev['helpful']['outOf'])}
                   for rev in train_data if rev['helpful']['outOf'] > 0]

valid_features = [{'nHelpful':rev['helpful']['nHelpful'],
                   'outOf':rev['helpful']['outOf'],
                   'ratio':(float(rev['helpful']['nHelpful'])/rev['helpful']['outOf'])}
                   for rev in valid_data if rev['helpful']['outOf'] > 0]

train_alpha = sum([rev['ratio'] for rev in train_features])/len(train_features)

tot_error = 0
for review in valid_dat:
    nHelpful = review['helpful']['nHelpful']
    outOf = review['helpful']['outOf']
    prediction = round(train_alpha * outOf)
    error = abs(nHelpful - prediction)
    if outOf > 0:
        tot_error += error

def construct_feature(review):
    punctuation = set(string.punctuation)
    r = ''.join([c for c in review['reviewText'].lower() if not c in punctuation])
    num_words = len(r.split())
    rating = review['rating']
    feature = [1.0, num_words, rating]
    return feature

X = [construct_feature(review) for review in train_data]
y = []
for review in train_data:
    outOf = review['helpful']['outOf']
    nHelpful = review['helpful']['nHelpful']
    if outOf > 0:
        ratio = float(nHelpful)/outOf
    else:
        ratio = 0.0
    y.append(ratio)

theta, residuals, rank, s = numpy.linalg.lstsq(X, y)

interact(local=locals())
