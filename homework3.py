import numpy
from copy import copy
import string
import yaml
from ast import literal_eval
from code import interact
import scipy.optimize
from collections import deque, defaultdict
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

"""
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
for review in valid_data:
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

def construct_labels(data):
    y = []
    for review in data:
        outOf = review['helpful']['outOf']
        nHelpful = review['helpful']['nHelpful']
        if outOf > 0:
            ratio = float(nHelpful)/outOf
        else:
            #ratio = 0.0
            continue
        y.append(ratio)
    return y

X = [construct_feature(review) for review in train_data 
     if review['helpful']['outOf'] > 0]
y = construct_labels(train_data)

theta, residuals, rank, s = numpy.linalg.lstsq(X, y)
print "Theta: ", theta

X_v = [numpy.array(construct_feature(review)) for review in valid_data]
y_v = [[review['helpful']['nHelpful'],
        review['helpful']['outOf']]
        for review in valid_data]

tot_error_2 = 0
for feature, label in zip(X_v, y_v):
    if label[1] > 0:
        p_ratio = numpy.dot(feature, theta)
        prediction = round(p_ratio * label[1])
        error = abs(label[0] - prediction)
        tot_error_2 += error

def pair_generator(fname):
    pairs = []
    for line in open(fname):
        if line.startswith('user'): continue    #skip header
        uid, iid, out_of = line.strip().split('-')
        pairs.append([uid, iid, int(out_of)])
    return pairs

def yield_line(fname):
    for line in open(fname):
        yield literal_eval(line)

def test_feature_gen(pairs_fname, data_fname):
    pair_predictions = []
    pairs = pair_generator(pairs_fname)
    for line in yield_line(data_fname):
        uid = line['reviewerID']
        iid = line['itemID']
        out_of = line['helpful']['outOf']
        pair = [uid, iid, out_of]
        if pair in pairs:
            feature_vec = numpy.array(construct_feature(line))
            ratio_prediction = numpy.dot(feature_vec, theta)
            prediction = ratio_prediction * out_of
            pair_predictions.append([pair, prediction])
        else:
            print "Pair not found in pairs!"
    return pair_predictions

print "making kaggle predictions..."
pair_predictions = test_feature_gen('pairs_Helpful.txt', 'helpful.json')

with open('kpred.txt', 'w') as kpred:
    kpred.write('userID-itemID-outOf,prediction\n')
    for pair in pair_predictions:
        uid = pair[0][0]
        iid = pair[0][1]
        out_of = str(pair[0][2])
        pred = str(pair[1])
        kpred.write('-'.join([uid, iid, out_of])+','+pred+'\n')
"""

############## PART 2 ################

fields = ['reviewText', 'rating', 'reviewerID', 'itemID']
train_data = load_fields('train.json', fields, 0, 100000)
valid_data = load_fields('train.json', fields, 900000, 1000000)

def avg_field(data, field):
    tot = 0.0
    for review in data:
        tot += review[field]
    avg = tot/len(data)
    return avg

alpha = avg_field(train_data, 'rating')

def field_rating_average(data, field):
    field_tot = defaultdict(float)
    field_count = defaultdict(int)
    field_avg = {}
    for review in data:
        field_id = review[field]
        rating = review['rating']
        field_tot[field_id] += rating
        field_count[field_id] += 1
    for fid in field_dict.keys():
        field_avg[fid] = field_tot[fid]/field_cound[fid]
    field_average = sum(field_avg.values())/len(field_avg)

def update_alpha(alpha, b_u, b_i):

def update_bu(alpha, b_u, b_i):

def update_bi(alpha, b_u, b_i):

def iter_params(data, lam):
    alpha = 4.0
    u_init = alpha - field_average(data, 'reviewerID')
    i_init = alpha - field_average(data, 'itemID')
    b_u = defaultdict(lambda: u_init)
    b_i = defaultdict(lambda: i_init)
    while True:
        alpha_old, b_u_old, b_i_old = copy(alpha), copy(b_u), copy(b_i)
        alpha = update_alpha(alpha, b_u, b_i, lam)
        b_u = update_bu(alpha, b_u, b_i, lam)
        b_i = update_bi(alpha, b_u, b_i, lam)
        if CONVERGENCE:
            break
    return alpha, b_u, b_i

interact(local=locals())
