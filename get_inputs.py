
"""
Created on Fri May, 1 17:46:51 2020
@author: miranda
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RUN THE MODEL 
# Output should be an ordered set of not-necessarily equally-sized
# matrices. Each matrix will have the same number of rows, but not necessarily
# the same number of columns. 

"""

import pandas as pd
import numpy as np 
import tensorflow_hub as hub
import csv
import process_text as pt
import keras.preprocessing.text as keraspre
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

''' Import Twitter US Airline Sentiment dataset from Kaggle. Can be found here: 
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    Dimensions: 14640 x 15. Sentiment is 63% negative, 21% neutral, 16% positive. 
    '''
filename = 'Tweets.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = np.array(list(reader))
    
tweets = data[:,10] # Define features (x)
sentiments = data[:,1] # Define labels (y)

# Clean tweet data
# Result is a list of length 14640 containing tokenized, cleaned words (stringso) 
clean_tweets = (pt.process_data(tweets))

# Specify characteristics. 15291 unique tokens in the set. 
vocab_size = len(set(chain.from_iterable(clean_tweets))) + 1 # Max int index + 1
longest_tweet = len(max(clean_tweets, key=len))

# Encode label classes with integers
le = LabelEncoder()
categorical_labels = ['positive', 'negative', 'neutral'] # Positive: 2, Neutral: 1, Negative: 0
e = le.fit(categorical_labels) # Fit instance to data
numerical_labels = e.transform(sentiments) # Transform categorical labels to integers

# Make test and train data 
train_size = round(.7*len(tweets))
test_size = round(.1*len(tweets))
validate_size = round(.2*len(tweets))
test_stop = train_size+test_size # marks divider btwn train and test data

# Average of word vectors to be used in place of missing words, per GloVe author
filename = 'glove_twitter_50d.txt'

# =============================================================================
# Calculate average of some of the glove vectors to be used in place of missing
# values, as per GloVe author. 
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        pass
n_vec = i + 1
hidden_dim = len(line.split(' ')) - 1
vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
average_vec = np.mean(vecs, axis=0)
          
# =============================================================================
# # Split data into training, testing, and validation sets. 
# (train_x, test_x, validate_x) = padded_sequences[:train_size], padded_sequences[train_size:test_stop], padded_sequences[test_stop:]
# (train_y, test_y, validate_y) = numerical_labels[:train_size], numerical_labels[train_size:test_stop], numerical_labels[test_stop:]
# 
# =============================================================================














