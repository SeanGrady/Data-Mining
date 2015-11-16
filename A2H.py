import numpy as np
import json
from collections import deque
from copy import copy
from random import random
from ast import literal_eval
from code import interact
from collections import defaultdict, namedtuple

testing = False

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

def load_data(fname, start, end):
    data_list = []
    for line in line_generator(fname, start, end):
        data_list.append(line)
    return data_list 

print "Loading training data..."
train_data = load_data('train.json', 0, 1000000)
if testing:
    print "Loading testing data..."
    valid_data = load_data('train.json', 900000, 1000000)
print "Data loaded."

class Model():
    def __init__(self, u_dict, i_dict, user_list, item_list, k):
        self.user_dict = u_dict
        self.item_dict = i_dict
        self.user_list = user_list
        self.item_list = item_list
        self.alpha = 4.2
        self.b_u = defaultdict.fromkeys(user_list, 0.1)
        self.b_i = defaultdict.fromkeys(item_list, 0.1)
        self.b_u.default_factory = float
        self.b_i.default_factory = float
        self.g_u = self.construct_g_u(k)
        self.g_i = self.construct_g_i(k)
        self.g_u.default_factory = lambda: np.zeros(k)
        self.g_i.default_factory = lambda: np.zeros(k)

    def construct_g_u(self, k):
        g_u = defaultdict()
        for user in self.user_list:
            user_vec = np.array([(random() * 2) - 1 for _ in range(k)])
            g_u[user] = user_vec
        return g_u

    def construct_g_i(self, k):
        g_i = defaultdict()
        for item in self.item_list:
            item_vec = np.array([(random() * 2) - 1 for _ in range(k)])
            g_i[item] = item_vec
        return g_i

def update_alpha(data, model, regularizer):
    b_u = model.b_u
    b_i = model.b_i
    g_u = model.g_u
    g_i = model.g_i
    #nominator = deque()
    nominator = 0.0
    for review in data:
        uid = review['reviewerID']
        iid = review['itemID']
        rating = review['rating']
        #nominator.append(rating - (b_u[uid] + b_i[iid] 
        #                           + np.dot(g_u[uid], g_i[iid])))
        nominator += (rating - (b_u[uid] + b_i[iid] 
                                   + np.dot(g_u[uid], g_i[iid])))
    #alpha = sum(nominator)/len(data)
    alpha = nominator / len(data)
    return alpha

def update_bi(model, regularizer):
    item_dict = model.item_dict
    alpha = model.alpha
    b_u = model.b_u
    b_i = model.b_i
    g_u = model.g_u
    g_i = model.g_i
    for iid, u_dict in item_dict.iteritems():
        nlist = [(review - (alpha + b_u[uid] + np.dot(g_u[uid], g_i[iid])))
                 for uid, review in u_dict.iteritems()]
        nominator = sum(nlist)
        b_i[iid] = float(nominator) / (regularizer + len(u_dict))
    return b_i

def update_bu(model, regularizer):
    user_dict = model.user_dict
    alpha = model.alpha
    b_u = model.b_u
    b_i = model.b_i
    g_u = model.g_u
    g_i = model.g_i
    for uid, i_dict in user_dict.iteritems():
        nlist = [(review - (alpha + b_i[iid] + np.dot(g_i[iid], g_u[uid])))
                 for iid, review in i_dict.iteritems()]
        nominator = sum(nlist)
        b_u[uid] = float(nominator) / (regularizer + len(i_dict))
    return b_u

def update_gu(model, regularizer):
    user_dict = model.user_dict
    alpha = model.alpha
    b_u = model.b_u
    b_i = model.b_i
    g_u = model.g_u
    g_i = model.g_i
    for uid, i_dict in user_dict.iteritems():
        for k in range(len(g_u[uid])):
            #nlist = deque()
            numerator = 0.0
            dsum = 0.0
            for iid, review in i_dict.iteritems():
                glist = [g_i[iid][i]*g_u[uid][i]
                         for i in range(len(g_i[iid]))
                         if i != k]
                gdot = sum(glist)
                #nlist.append(g_i[iid][k]*(review - (alpha + b_u[uid] 
                #                                    + b_i[iid] + gdot)))
                numerator += (g_i[iid][k]*(review - (alpha + b_u[uid] 
                                                    + b_i[iid] + gdot)))
                dsum += g_i[iid][k]**2
            #numerator = sum(nlist)
            denominator = float(regularizer + dsum)
            guk = numerator / denominator
            g_u[uid][k] = guk
    return g_u

def update_gi(model, regularizer):
    item_dict = model.item_dict
    alpha = model.alpha
    b_u = model.b_u
    b_i = model.b_i
    g_u = model.g_u
    g_i = model.g_i
    for iid, u_dict in item_dict.iteritems():
        for k in range(len(g_i[iid])):
            #nlist = deque()
            numerator = 0.0
            dsum = 0.0
            for uid, review in u_dict.iteritems():
                glist = [g_i[iid][i]*g_u[uid][i]
                         for i in range(len(g_i[iid]))
                         if i != k]
                gdot = sum(glist)
                #nlist.append(g_u[uid][k]*(review - (alpha + b_u[uid] 
                #                                    + b_i[iid] + gdot)))
                numerator += (g_u[uid][k]*(review - (alpha + b_u[uid] 
                                                    + b_i[iid] + gdot)))
                dsum += g_u[uid][k]**2
            #numerator = sum(nlist)
            denominator = float(regularizer + dsum)
            gik = numerator / denominator
            g_i[iid][k] = gik
    return g_i

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

item_list = [review['itemID'] for review in train_data]
user_list = [review['reviewerID'] for review in train_data]
u_dict, i_dict = build_ui_dicts(train_data)
model = Model(u_dict, i_dict, user_list, item_list, 5)

def iter_params(model, data, regularizer):
    thresh = 0.0001
    print_set = set([i*10 for i in range(10)])
    i = 0
    while True:
        #if i in print_set: print model.alpha
        print model.alpha
        alpha_old = copy(model.alpha)
        model.alpha = update_alpha(data, model, regularizer)
        model.b_u = update_bu(model, regularizer)
        model.b_i = update_bi(model, regularizer)
        model.g_u = update_gu(model, regularizer)
        model.g_i = update_gi(model, regularizer)
        if abs(alpha_old - model.alpha) < thresh:
            break
        #i += 1
    return model

print "Iterating..."
regularizer = 10
model_final = iter_params(model, train_data, regularizer)
print "Alpha: ",model_final.alpha

def predict(uid, iid, model):
    alpha = model.alpha
    b_u = model.b_u
    b_i = model.b_i
    g_i = model.g_i
    g_u = model.g_u
    rating = alpha + b_u[uid] + b_i[iid] + np.dot(g_u[uid], g_i[iid])
    return rating

def MSE(model, data):
    squared_error = 0.0
    for review in data:
        uid = review['reviewerID']
        iid = review['itemID']
        prediction = predict(uid, iid, model)
        #prediction = model['a'] + model['bu'][uid] + model['bi'][iid]
        value = review['rating']
        error = prediction - value
        squared_error += error**2
    mean_squared_error = squared_error / len(data)
    return mean_squared_error

if testing:
    print "calculating MSE"
    valid_mse = MSE(model, valid_data)
    print "MSE: ", valid_mse
    print "Regularizer: ", regularizer

"""
def convert_model(model):
    dumpable_model = {
            'a': model.alpha,
            'b_u': model.b_u,
            'b_i': model.b_i,
            'g_u': model.g_u,
            'g_i': model.g_i
    }
    return dumpable_model 

print "Dumping model..."
with open('model.txt', 'w') as mfile:
    json_model = convert_model(model)
    json.dump(json_model, mfile) 
print "Making Kaggle predictions..."
"""
#################### Kaggle Predictions ######################################

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
            prediction = predict(uid, iid, model) 
            kfile.write('-'.join([uid, iid]) + ',' + str(prediction) + '\n')

print "making kaggle predictions for rating..."
kaggle_rating_predictor('pairs_Rating.txt', model, 'kaggle_ratings.txt')
#interact(local=locals())
