from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
import random

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
filename_raw = 'training.1600000.processed.noemoticon.csv'
filename_shuffled = 'training.1600000.processed.shuffle.csv'

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

# Split data into training, test and validation set (60:20:20)
X = dataset['text']
y = dataset['target']

out = []
for i in range(len(X)):
   out.append('_label_ ' + str(y[i]) + ' ' + X [i].replace("\n",""))

out = pd.DataFrame(out)
out.to_csv('out.csv',sep='\t',index=False,header=False)