import numpy as np
import logging
import random
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from wordcloud import WordCloud
from many_stop_words import get_stop_words
from keras.models import load_model

from scipy import interp
from itertools import cycle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import word2vec
import gensim
import seaborn as sn
from gensim.utils import simple_preprocess
from keras.utils import to_categorical
import pickle
#import h5py
from time import time
np.random.seed(7)
import tensorflow as tf
from datetime import datetime, timedelta

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


columns = ["target", "ids", "date", "flag", "user", "text"]
dataset = pd.read_csv(filename_raw,encoding = "ISO-8859-1",sep=',',names = columns)
dataset = dataset.iloc[np.random.choice(range(len(dataset)), size = 2000,replace =False)]
dataset.index = range(len(dataset))


#dataset = pd.read_csv(filename_shuffled,names=columns,delimiter = ",", nrows=2000)
#dataset.apply(np.random.permutation, axis=1)

logging.debug("Dataset keys: %s" %str(dataset.keys()))

print(dataset.keys())

# Initial data preparation
logging.debug("Initial data preparation")
dataset['text'] = dataset['text'].str.lower()
dataset = dataset[dataset['text'].notnull() & dataset['target'].notnull()]
print(dataset['target'].unique())

#print(dataset.head(100))

# Split data into training, test and validation set (60:20:20)
X = dataset['text']
y = (dataset['target'] >0)*1

logging.debug('Split datas')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)




# Slownik : http://vectors.nlpl.eu/repository/
#  no 5 English Wikipedia Dump of February 2017
print('load dict')
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.txt', binary=False)
print('load dict')


# Define embedding matrix
embedding_matrix = word2vec_model.wv.syn0
print('Shape of embedding matrix: ', embedding_matrix.shape)

# Vectorize X_train and X_test to 2D tensor
top_words = embedding_matrix.shape[0]

# Define max lenght of sentence and number of classes (negative, neutral and positive)
mxlen = 30
nb_classes = 1

# Tokenize words
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#print(word_index)


X_train = sequence.pad_sequences(sequences_train, maxlen=mxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=mxlen)
X_val = sequence.pad_sequences(sequences_val, maxlen=mxlen)

#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#y_val = np_utils.to_categorical(y_val)

nb_epoch = 2
batch_size = 2

# Add embedding_layer weights from existing dictionary and freeze training layer
embedding_layer = Embedding(embedding_matrix.shape[0],
                            output_dim = embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=mxlen
                            )
# Model
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(embedding_matrix.shape[1], dropout=0.5, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

t0 = time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val)

# Measures and confusion matrix

y_pred = model.predict(X_test)
# Convert Y_Test into 1D array
yy_true = [np.argmax(i) for i in y_test]
print(yy_true)

yy_scores = [np.argmax(i) for i in y_pred]
print(yy_scores)

print("Recall: " + str(recall_score(yy_true, yy_scores, average='weighted')))
print("Precision: " + str(precision_score(yy_true, yy_scores, average='weighted')))
print("F1 Score: " + str(f1_score(yy_true, yy_scores, average='weighted')))

# Apply Confusion matrix
#Y_pred = model.predict(X_val, verbose=2)
y_pred = np.argmax(y_pred, axis=1)

confusion_matrix(yy_true,yy_scores)
#
# out = []
# for i in range(len(X)):
#    out.append('_label_ ' + str(y[i]) + ' ' + X [i].replace("\n",""))
#
# out = pd.DataFrame(out)
# out.to_csv('out.csv',sep='\t',index=False,header=False)



