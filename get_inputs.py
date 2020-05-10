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
        self.set_labels(self.raw_data)
        
    def load_data(self, filename):
        '''
        Parameters
        ----------
        filename: a string filename which includes extension and relevant path.
        
        Returns
        ----------
        data_array: a numpy array of dimensions (examples, features).
        '''
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data_array = np.array(list(reader))
            return data_array
        
    def define_data(self):
        '''
        Sets dataset characteristics
        '''
        
        tweets = self.raw_data[:,10] # Define features (x)
        self.clean_sequences = process_data(tweets) # Clean/tokenize data. Result: a list of len = 14640
        self.vocab = set(chain.from_iterable(self.clean_sequences))
        self.vocab_size = len(self.vocab) + 1 # Unique words in the set
        self.longest_sequence = len(max(self.clean_sequences, key=len))
            
    def set_labels(self, data_array):
        '''
        Sets label classes
        '''
        self.label_scheme = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.targets = data_array[:,1] # Define labels (y)
        self.numerical_labels = np.array([self.label_scheme[item] for item in self.targets])
        self.one_hot_numerical_labels = to_categorical(self.numerical_labels, 3)
    
    def get_data(self):
        '''Getter method for raw_data
        '''
        return self.raw_data

    def get_clean_sequences(self):
        '''Getter method for cleaned data
        '''
        return self.clean_sequences
    
    def get_vocab_info(self):
        '''Getter method for vocab_attributes
        '''
        return self.vocab, self.vocab_size
    
    def get_one_hot_numerical_labels(self):
        '''Getter method for one-hot-encoded labels
        '''
        return self.one_hot_numerical_labels


dataset = Dataset(filename = 'Tweets.csv') 
clean_tweets = dataset.get_clean_sequences() # List 
vocab, vocab_size = dataset.get_vocab_info() # Returns a set and an int, respectively
one_hot_numerical_labels = dataset.get_one_hot_numerical_labels()

