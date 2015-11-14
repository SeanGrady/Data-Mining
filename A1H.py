import numpy
from copy import copy, deepcopy
import string
import yaml
from ast import literal_eval
from code import interact
import scipy.optimize
from collections import deque, defaultdict
import gc

def line_generator(fname, start, end):
    i = 0
    print_set = set([i*100000 for i in range(10)])
    for line in open(fname):
        if i >= start and i <= end:
            if i in print_set: print i
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
print "Loading testing data..."
train_data = load_fields('train.json', fields, 0, 900000)
print "Loading validation data..."
valid_data = load_fields('train.json', fields, 900000, 1000000)
print "Data loaded."

def construct_feature(review):
    #the features are num capital words, num '!', num '?', num words,
    #num chars, rating, num votes
    text = review['reviewText']
    rating = review['rating']
    num_votes = review['helpful']['outOf']
    num_chars = len(text)
    num_exp = len([char for char in text if char is '!'])
    num_ques = len([char for char in text if char is '?'])
    punctuation = set(string.punctuation)
    r = ''.join([c for c in text.lower() 
                 if not c in punctuation])
    words = r.split()
    cap_words = [word for word in words if word.isupper()]
    num_words = len(words)
    num_cap_words = len(cap_words)
    feature = [1.0, num_chars, num_cap_words, num_words, num_exp, num_ques, num_votes, rating]
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

print "Construcing training vectors..."

X = [construct_feature(review) for review in train_data 
     if review['helpful']['outOf'] > 0]
y = construct_labels(train_data)

print "beginning least squares regression..."

theta, residuals, rank, s = numpy.linalg.lstsq(X, y)
print "Theta: ", theta

print "Constructing validation sets..."

X_v = [numpy.array(construct_feature(review)) for review in valid_data]
y_v = [[review['helpful']['nHelpful'],
        review['helpful']['outOf']]
        for review in valid_data]

print "Finding validation error..."

tot_error = 0
for feature, label in zip(X_v, y_v):
    if label[1] > 0:
        p_ratio = numpy.dot(feature, theta)
        prediction = p_ratio * label[1]
        error = abs(label[0] - prediction)
        tot_error += error

print "Total error: ", tot_error
print "Mean error: ", tot_error/len(valid_data)

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

alpha_1 = avg_field(train_data, 'rating')

def field_rating_average(data, field):
    field_tot = defaultdict(float)
    field_count = defaultdict(int)
    field_avg = {}
    for review in data:
        field_id = review[field]
        rating = review['rating']
        field_tot[field_id] += rating
        field_count[field_id] += 1
    for fid in field_tot.keys():
        field_avg[fid] = float(field_tot[fid])/field_count[fid]
    field_average = sum(field_avg.values())/len(field_avg)
    return field_average

def update_alpha(data, alpha, b_u, b_i):
    nominator = []
    for review in data:
        uid = review['reviewerID']
        iid = review['itemID']
        rating = review['rating']
        nominator.append(rating - (b_u[uid] + b_i[iid]))
    a = sum(nominator) / len(data)
    return a

def update_bu(u_dict, alpha, b_u, b_i, lam):
    for uid, i_list in u_dict.iteritems():
        nlist = [(review - (alpha + b_i[iid])) 
                 for iid, review in i_list.iteritems()]
        nominator = sum(nlist)
        b_u[uid] = float(nominator) / (lam + len(i_list))
    return b_u

def update_bi(i_dict, alpha, b_u, b_i, lam):
    for iid, u_list in i_dict.iteritems():
        nlist = [(review - (alpha + b_u[uid]))
                 for uid, review in u_list.iteritems()]
        nominator = sum(nlist)
        b_i[iid] = float(nominator) / (lam + len(u_list))
    return b_i

def build_ui_dicts(data):
    u_dict = defaultdict(dict)
    i_dict = defaultdict(dict)
    for review in data:
        uid = review['reviewerID']
        iid = review['itemID']
        rating = review['rating']
        u_dict[uid][iid] = rating
        i_dict[iid][uid] = rating
    return u_dict, i_dict

def iter_params(data, lam):
    thresh = 0.00000001
    alpha = 4.0
    u_init = field_rating_average(data, 'reviewerID') - alpha 
    i_init = field_rating_average(data, 'itemID') - alpha
    user_list = [review['reviewerID'] for review in data]
    item_list = [review['itemID'] for review in data]
    b_u = defaultdict.fromkeys(user_list, u_init)
    b_i = defaultdict.fromkeys(item_list, i_init)
    b_u.default_factory = float
    b_i.default_factory = float
    u_dict, i_dict = build_ui_dicts(data)
    while True:
        #print alpha
        #alpha_old, b_u_old, b_i_old = copy(alpha), deepcopy(b_u), deepcopy(b_i)
        alpha_old = copy(alpha)
        alpha = update_alpha(data, alpha, b_u, b_i)
        b_u = update_bu(u_dict, alpha, b_u, b_i, lam)
        b_i = update_bi(i_dict, alpha, b_u, b_i, lam)
        if abs(alpha_old - alpha) < thresh:
            break
    return alpha, b_u, b_i, user_list, item_list

print "beginning iteration..."
alpha, b_u, b_i, user_list, item_list = iter_params(train_data, 4.2)
model = {'a': alpha, 'bu': b_u, 'bi': b_i,
         'users': user_list, 'items':item_list}

def predict(uid, iid, model):
    alpha = model['a']
    b_u = model['bu']
    b_i = model['bi']
    rating = alpha + b_u['uid'] + b_i['iid']
    return rating

def MSE(model, data):
    squared_error = 0.0
    for review in data:
        uid = review['reviewerID']
        iid = review['itemID']
        #prediction = predict(uid, iid, model)
        prediction = model['a'] + model['bu'][uid] + model['bi'][iid]
        value = review['rating']
        error = prediction - value
        squared_error += error**2
    mean_squared_error = squared_error / len(data)
    return mean_squared_error

def trivial_MSE(alpha, data):
    squared_error = 0.0
    for review in data:
        rating = review['rating']
        prediction = alpha
        error = prediction - rating
        squared_error += error**2
    MSE = squared_error / len(data)
    return MSE

print "calculating MSE..."
valid_MSE = MSE(model, valid_data)
print valid_MSE

def rpair_generator(fname):
    pairs = []
    with open(fname) as pfile:
        for line in pfile:
            if line.startswith('user'): continue    #skip header
            uid, iid = line.strip().split('-')
            pairs.append({'uid':uid, 'iid':iid})
    return pairs

def kaggle_rating_predictor(test_file, model, kaggle_file):
    pairs = rpair_generator(test_file)
    with open(kaggle_file, 'w') as kfile:
        kfile.write('userID-itemID,prediction\n')
        for pair in pairs:
            uid = pair['uid']
            iid = pair['iid']
            prediction = model['a'] + model['bu'][uid] + model['bi'][iid]
            kfile.write('-'.join([uid, iid]) + ',' + str(prediction) + '\n')

print "making kaggle predictions for rating..."
kaggle_rating_predictor('pairs_Rating.txt', model, 'kaggle_ratings.txt')

interact(local=locals())
"""
