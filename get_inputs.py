#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np 
import csv
from process_text import process_data
from itertools import chain
from tensorflow.keras.utils import to_categorical

''' Import Twitter US Airline Sentiment dataset from Kaggle. Can be found here: 
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    Dimensions: 14640 x 15. Sentiment is 63% negative, 21% neutral, 16% positive. '''
  
class Dataset:
    
    def __init__(self, filename):
        self.raw_data = self.load_data(filename)
        self.define_data()
        self.define_labels(self.raw_data)
        
    def load_data(self, filename):
        '''
        Parameters
        ----------
        filename : STRING
            Takes in a filename, including extension and relevant path.
        Returns
        -------
        data : NUMPY ARRAY
            Returns a numpy array of dimensions (M, N).
        '''
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data_array = np.array(list(reader))
            return data_array
        
    def define_data(self):
        tweets = self.raw_data[:,10] # Define features (x)
        self.clean_tweets = process_data(tweets) # Clean/tokenize data. Result: a list of len = 14640
    
        # Specify characteristics. 15291 unique tokens in the set. 
        self.vocab = set(chain.from_iterable(self.clean_tweets))
        self.vocab_size = len(self.vocab) + 1 # Unique words in the set
        self.longest_tweet = len(max(self.clean_tweets, key=len))
        # return self.clean_tweets
            
    def define_labels(self, data_array):
        # Encode label classes with ordinal integers
        label_dict = {'negative': -1, 'neutral': 0, 'positive': 1}
        sentiments = data_array[:,1] # Define labels (y)
        numerical_labels = np.array([label_dict[item] for item in sentiments])
        self.one_hot_numerical_labels = to_categorical(numerical_labels, 3)
    
    def get_data(self):
        return self.raw_data

    def get_clean_tweets(self):
        return self.clean_tweets
    
    def get_vocab_info(self):
        return self.vocab, self.vocab_size
    
    def get_one_hot_numerical_labels(self):
        return self.one_hot_numerical_labels

dataset = Dataset(filename = 'Tweets.csv') 
clean_tweets = dataset.get_clean_tweets() # List 
vocab, vocab_size = dataset.get_vocab_info() # Returns a set and an int, respectively
one_hot_numerical_labels = dataset.get_one_hot_numerical_labels()

