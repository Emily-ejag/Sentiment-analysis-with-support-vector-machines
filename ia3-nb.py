# IA3 Group 37: David Smerkous, Anita Ruangrotsakun, Emily Arteaga
# Date: 11/12/2022


# Imports
import pandas as pd
import numpy as np
import sklearn
import re
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

# %%
# load data from CSV
def load(name):
    data = pd.read_csv(name)
    data = data.sort_values(by=['sentiment'])
    return data

# make text all lowercase
# def lower_case(data):
#     column, out = 'text', 'text_lower'
#     data[out] = data[column].str.lower()
#     return data

# remove all punctuation
# def remove_punctuation(data):
#     column, out = 'text_lower', 'text_no_punctuation'

#     tokens = data[column].str.split()
#     tokens = tokens.apply(lambda words: map(partial(re.sub, "&lt;/?.*?&gt;", " &lt;&gt; "), words))
#     tokens = tokens.apply(lambda words: map(partial(re.sub, "<.*?>", ""), words))
#     tokens = tokens.apply(lambda words: map(partial(re.sub, "(\\d|\\W)+", " "), words))
#     tokens = tokens.apply(lambda words: [word.strip() for word in words if word.strip()])

#     data[out] = tokens.apply(" ".join)
#     return data

# %%
# preprocess training data
train = load('IA3-train.csv')
train.head()

# %%
# Part 0a: generate a feature vector for each tweet where the element of the vector tracks the number of times a specific word appears in the tweet
vectorizer = CountVectorizer(lowercase=True)
vectorizer = vectorizer.fit(train['text'])

# %%
# Part 0a: generate top 10 most frequent words for positive and negative tweets respectively

def get_ind_features(vec):
    # save feature names
    ind_to_feature = {}
    # vocab has structure of [token, column index]
    # we want a structure of [column index, token]
    for name, ind in vec.vocabulary_.items():
        ind_to_feature[int(ind)] = str(name)
    return ind_to_feature


def get_top_counted_words(D, ind_features, n=10, col='text'):
    # gets the top n words from the set of documents D (docs, words) (if D is a df then use col)
    global ind_to_features, vectorizer
    
    # if it's a pandas df then transform into bag of words
    if isinstance(D, pd.DataFrame):
        D = vectorizer.transform(D[col])
        print('Transformed shape', D.shape)
        # row_i = one tweet (tweet i)
        # col_j = number of times that word j appears in tweet
    
    # sum the word counts across all documents (tweets)
    # basically take sum of all columns, collapse into one row
    nsum = np.sum(D, axis=0)

    # sort by indices (most common to least common)
    nsort = np.flip(np.argsort(nsum, axis=-1).reshape((D.shape[1], 1)))
    # get the first n (ie top) words
    res = []
    counts = []
    for i in range(n):
        res.append(ind_features[int(nsort[i])])
        counts.append(nsum[0, int(nsort[i])])
    return res, counts

def split_tweets(df):
    # splits documents into positive sentiment/negative sentiment tweets
    return df[df['sentiment'] == 0], df[df['sentiment'] == 1]

print('START Part 0')
neg, pos = split_tweets(train)
print('Negative samples', len(neg))
print('Positive samples', len(pos))
print()
ind_features = get_ind_features(vectorizer)
print('Part 0A negative words countVectorizer:', get_top_counted_words(neg, ind_features))
print()
print('Part 0A positive words countVectorizer:', get_top_counted_words(pos, ind_features))

# %%
# Part 0b: generate the TF-IDF representation of the tweets
tfIdfVectorizer = TfidfVectorizer(use_idf=True, lowercase=True) 

# fit transform returns bag of words
tfIdf_X = tfIdfVectorizer.fit_transform(train['text'])

# %%
# Part 0b: generate top 10 most frequent words for positive and negative tweets using TF-IDF representation
tfid_ind_features = get_ind_features(tfIdfVectorizer)
neg_tfid_D = tfIdfVectorizer.transform(neg['text'])
pos_tfid_D = tfIdfVectorizer.transform(pos['text'])
print('Negative samples', len(neg))
print('Positive samples', len(pos))
print()
print('Part 0B negative words tfid:', get_top_counted_words(neg_tfid_D, tfid_ind_features))
print()
print('Part 0B positive words tfid:', get_top_counted_words(pos_tfid_D, tfid_ind_features))
print('END Part 0')

# %%
# Part 1: Use sklearn svm.SVC() to train linear SVMs and tune hyperparameter c
# Start with c = 10^i with i = [-4, 4] and then expand the range
# Find the best c for optimizing validation performance

# load dev set for checking classification performance for each best c we find
print('START Part 1')
dev = load('IA3-dev.csv')

dev_tfid = tfIdfVectorizer.transform(dev['text'])
train_tfid = tfIdfVectorizer.transform(train['text'])

all_c = []
all_train_acc = []
all_val_acc = []
all_n_vec = []


def test_hyperparameters(parameters, log=True):
    if log:
        print("\nTesting hyperparameters:", parameters)
    
    # train SVM using chosen parameters
    svc = svm.SVC(**parameters)
    svc.fit(tfIdf_X, train['sentiment'])
    n_vec = np.sum(svc.n_support_)
    
    # find validation performance (accuracy)
    acc_dev = svc.score(dev_tfid, dev['sentiment'])
    acc_train = svc.score(train_tfid, train['sentiment'])
    if log:
        print("Validation accuracy:", acc_dev, "Train accuracy:", acc_train)
    return acc_dev, acc_train, svc, n_vec

# determine best parameters within this range, then expand
def scan_c_param(c_list, start_params={'kernel': 'linear'}):
    global all_train_acc, all_val_acc, all_c, all_n_vec
    parameters = deepcopy(start_params)
    best_acc = 0.0
    best_train_acc = 0.0
    best_params = None
    best_svc = None
    for c in c_list:
        parameters['C'] = c
        acc_dev, acc_train, svc, n_vec = test_hyperparameters(parameters)
        all_c.append(c)
        all_train_acc.append(acc_train)
        all_val_acc.append(acc_dev)
        all_n_vec.append(n_vec)
        if acc_dev > best_acc:
            best_acc = acc_dev
            best_train_acc = acc_train
            best_params = deepcopy(parameters)
            best_svc = svc

    print('\nBest params', best_params, 'with validation accuracy', best_acc, 'train acc', best_train_acc)


# %%
# Plot training and validation accuracy vs values of c
def plot_accuracy(train_acc, val_acc, c, name="accuracy.png", include_last=True):
    # sort the data for increasing c values
    data = np.array(list(zip(c, train_acc, val_acc)), np.float32)
    args = np.argsort(data[:, 0])
    data = data[args]
    c, train_acc, val_acc = data[:, 0], data[:, 1], data[:, 2]
    
    # red dashes for train_acc, blue dashes for val_acc, but we can change this
    if not include_last:
        plt.plot(c[:-2], train_acc[:-2], 'ro', c[:-2], val_acc[:-2], 'bs')
        plt.xlabel('C value')
        plt.ylabel('Accuracy')
        plt.title('Accuracy with respect to C')
        plt.savefig(name)
    else:
        plt.plot(c, train_acc, 'ro', c, val_acc, 'bs')
        plt.xlabel('C value')
        plt.ylabel('Accuracy')
        plt.title('Accuracy with respect to C')
        plt.savefig(name)
    
# Plot number of support vectors vs values of c
def plot_support_vectors(support_vectors, c, name="support_vectors.png", include_last=True):
    # sort the data for increasing c values
    data = np.array(list(zip(c, support_vectors)), np.float32)
    args = np.argsort(data[:, 0])
    data = data[args]
    c, support_vectors = data[:, 0], data[:, 1]
    
    if not include_last:
        plt.plot(c[:-2], support_vectors[:-2], 'go')
        plt.xlabel('C value')
        plt.ylabel('Number of support vectors')
        plt.title('Number of support vectors with respect to C')
        plt.savefig(name)
    else:
        plt.plot(c, support_vectors, 'go')
        plt.xlabel('C value')
        plt.ylabel('Number of support vectors')
        plt.title('Number of support vectors with respect to C')
        plt.savefig(name)

print('\nSTART Part 1')
# set counts trying our actual c values for hyperparameter tuning
all_c = []
all_train_acc = []
all_val_acc = []
all_n_vec = []

scan_c_param([.0001, .001, .01, 0.1, 1, 10, 100, 1000, 10000])

# %%
scan_c_param([1., 2., 3., 4.])

# %%
scan_c_param([1., 1.25, 1.5, 1.75, 2.])

# %%
scan_c_param([1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45])

# %%
scan_c_param([1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39])

# %%
# plot without 10000 and 1000 points
plot_accuracy(all_train_acc, all_val_acc, all_c, name="part1_accuracy_a.png", include_last=False)

# %%
# plot all points
plot_accuracy(all_train_acc, all_val_acc, all_c, name="part1_accuracy_b.png")

# %%
# plot without the 10000 and 1000 points
plot_support_vectors(all_n_vec, all_c, name="part1_support_vectors_a.png", include_last=False)

# %%
plot_support_vectors(all_n_vec, all_c, name="part1_support_vectors_b.png")
print('END Part 1')

print('\nSTART Part 2')
# Part 2: Train SVMs with quadratic kernels and tune hyperparameter c
# Start with c = 10^i with i = [-4, 4] and then expand the range
# Find the best c for optimizing validation performance

# reset counts
all_c = []
all_train_acc = []
all_val_acc = []
all_n_vec = []

quad_params = {
    'kernel': 'poly',
    'degree': 2
}

# %%
scan_c_param([.0001, .001, .01, 0.1, 1, 10, 100, 1000, 10000], quad_params)

# %%
scan_c_param([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], quad_params)

# %%
plot_accuracy(all_train_acc, all_val_acc, all_c, name="part2_accuracy_a.png")

# %%
plot_support_vectors(all_n_vec, all_c, name="part2_support_vectors_a.png")

# %%
# plot without the 10000 and 1000 points
plot_accuracy(all_train_acc, all_val_acc, all_c, name="part2_accuracy_b.png", include_last=False)

# %%
# plot without the 10000 and 1000 points
plot_support_vectors(all_n_vec[:-1], all_c[:-1], name="part2_support_vectors_b.png", include_last=False)

print('END Part 2')


print('\nSTART Part 3')
# Part 3: Train SVMs with RBF kernels and tune hyperparamaters c and gamma
# For c, start with c = 10^i with i = [-4, 4] and then expand the range
# For gamma, start with gamma = 10^i with i = [-5, 1] and then expand the range
# determine best parameters within this range, then expand

# reset counts
all_c = []
sup_g01 = []
sup_c10 = []
all_train_acc = []
all_val_acc = []
all_n_vec = []

def scan_rbf_params(c_list, g_list):
    global all_train_acc, all_val_acc, all_c, all_n_vec, sup_g01, sup_c10
    parameters = {
        'kernel': 'rbf'
    }
    best_acc = 0.0
    best_train_acc = 0.0
    best_params = None
    best_svc = None
    nmap = np.zeros((len(c_list), len(g_list)), np.float32)
    nmap_train = np.zeros((len(c_list), len(g_list)), np.float32)
    for i, c in enumerate(c_list):
        print('Running for c:', c)
        for j, g in enumerate(g_list):
            parameters['C'] = c
            parameters['gamma'] = g
            acc_dev, acc_train, svc, n_vec = test_hyperparameters(parameters, log=False)
            all_c.append(c)
            all_train_acc.append(acc_train)
            all_val_acc.append(acc_dev)
            all_n_vec.append(n_vec)
            
            if g == 0.1:
                sup_g01.append((c, g, n_vec))
            elif c == 10:
                sup_c10.append((c, g, n_vec))
            
            nmap[i, j] = acc_dev
            nmap_train[i, j] = acc_train
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_train_acc = acc_train
                best_params = deepcopy(parameters)
                best_svc = svc

    print('\nBest params', best_params, 'with validation accuracy', best_acc, 'with train acc', best_train_acc)
    return nmap, nmap_train, best_svc


# %%
cs = [.0001, .001, .01, 0.1, 1, 10, 100, 1000, 10000]
gammas = [.00001, .0001, .001, .01, .1, 1, 10]

# parameters = {'kernel':('linear', 'rbf'), 'C': [.0001, .001, .01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
first_nmap, first_nmap_train, svc = scan_rbf_params(cs, gammas)

# %%
g01_c, _, g01_n = zip(*sup_g01)
plot_support_vectors(g01_n, g01_c, name="part3_support_vectors_g01.png")

# %%
_, c10_g, c10_n = zip(*sup_c10)
plot_support_vectors(c10_n, c10_g, name="part3_support_vectors_c10.png") # @TODO change label to gamma

# %%
# Part 3: Use the heatmap function from seaborn https://seaborn.pydata.org/generated/seaborn.heatmap.html
# to plot the training accuracy and validation accuracy (separately in two different heatmaps) as a function of c and gamma
# gamma = 1 / variance = 1 / sigma^2

# rows = C, cols = gamma
# for each c and g, there is an associated accuracy

sns.set_theme()
sns.heatmap(first_nmap, annot=True, xticklabels=gammas, yticklabels=cs)
plt.xlabel('Gamma', fontsize=16)
plt.ylabel('C', fontsize=16)
plt.title('Validation accuracies w.r.t C and Gamma', fontsize=16)
# plt.show()
plt.savefig('part3_heatmap_validation.png')

# %%
cs = [5, 7.5, 10, 20, 30]
gammas = [.001, .01, .1, 1]
secon_nmap, second_svc = scan_rbf_params(cs, gammas)

# %%
sns.set_theme()
sns.heatmap(first_nmap_train, annot=True, xticklabels=gammas, yticklabels=cs)
plt.xlabel('Gamma', fontsize=16)
plt.ylabel('C', fontsize=16)
plt.title('Training accuracies w.r.t C and Gamma', fontsize=16)
# plt.show()
plt.savefig('part3_heatmap_train.png')

# %%
cs = [8.5, 9]
gammas = [0.15, 0.2]
third_nmap, third_svc = scan_rbf_params(cs, gammas)

print('END Part 3')