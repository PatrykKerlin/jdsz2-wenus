#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:24:56 2019

@author: nanokoper
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
import seaborn as sns
import logging, random
from time import time
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# Samples - list of numpy array or numpy array
# Labels needs to be in numpy array

# Initializing process
logging.debug("Initializing...")
initial_time = time()

#Load dataset
logging.debug("Load dataset")
columns = ["target", "ids", "date", "flag", "user", "text"]
filename_raw = '/home/nanokoper/Pulpit/ISA/training.1600000.processed.noemoticon.csv'
filename_shuffled = '/home/nanokoper/Pulpit/ISA/training.1600000.processed.shuffle.csv'

with open(filename_raw, 'r',encoding= "ISO-8859-1") as r, open(filename_shuffled, 'w') as w:
    data = r.readlines()
    rows = data[:]
    random.shuffle(rows)
    rows = '\n'.join([row.strip() for row in rows])
    w.write(rows)

dataset = pd.read_csv(filename_shuffled,names=columns,delimiter = ",", nrows=20000)
dataset.apply(np.random.permutation, axis=1)

logging.debug("Dataset keys: %s" %str(dataset.keys()))

print(dataset.keys())

# Initial data preparation
logging.debug("Initial data preparation")
dataset['text'] = dataset['text'].str.lower()
dataset = dataset[dataset['text'].notnull() & dataset['target'].notnull()]
print(dataset.head(10))
#print(len(df[df[Target] == 1]))
#Remove unwanted chars and stopwords (.,@ etc.)
dfClean = pd.DataFrame(columns=['text'])
#stop = set(stopwords.words('english'))
for i in dataset['text']:
    for j in string.punctuation:
        i = i.replace(j, '')
    #clean_mess = [word for word in i.split() if word.lower() not in stop]
    dfClean = dfClean.append({'text': i}, ignore_index=True)
    
# Split data into training, test and validation set (60:20:20)
X = dfClean['text']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    
#wykres wyrysowac i wywalic ogon
cv = CountVectorizer(ngram_range=(1, 2), max_df = 0.95, min_df = 0.05, stop_words = 'english')
X_train = list(X_train)
X_traincv = cv.fit_transform(X_train)
print(X_traincv.toarray())
for i in X_traincv:
    features = cv.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(X_traincv)
visualizer.poof()
"""
out = []
for i in range(len(X)):
   out.append('_label_ ' + str(y[i]) + ' ' + X [i].replace("\n",""))

out = pd.DataFrame(out)
#out.to_csv('/home/nanokoper/Pulpit/ISA/out.csv',sep='\t',index=False,header=False)


# Split data into training, test and validation set (60:20:20)

"""


