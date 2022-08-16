#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy 
import csv
from process_text import process_data
from itertools import chain
from tensorflow.keras.utils import to_categorical

class Dataset:
    '''
    This is a class containing dataset vectors, intended to take raw, 
    unformatted text and convert and store its cleaned and tokenized versions.
    It also creates and carries one-hot-encoded labels for said data. 
    
    Attributes: 
        filename: 
    
    '''
    
    def __init__(self, filename):
        '''
        The constructor for Dataset class.
        
        Parameters:
            raw_data (numpy.ndarray)
            define_data ()

        '''
        self.raw_data = self.load_data(filename)
        self.define_data()
        self.set_labels(self.raw_data)
        
    def load_data(self, filename):
        '''
        Parameters:
            filename (string): path and filename including file extension.
        
        Returns: 
            data (numpy array): dimensions (examples, features)
        '''
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data = numpy.array(list(reader))
            return data
        
    def define_data(self):
        '''Setter method for class Dataset'''
        
        tweets = self.raw_data[:,10] # Define features (x)
        self.clean_sequences = process_data(tweets) # Clean/tokenize data. Result: a list of len = 14640
        self.vocab = set(chain.from_iterable(self.clean_sequences))
        self.vocab_size = len(self.vocab) + 1 # Unique words in the set
        self.longest_sequence = len(max(self.clean_sequences, key=len))
            
    def set_labels(self, data):
        '''
        Setter method for label classes.
        
        Parameters: 
            data (numpy.ndarray): 
        
        '''
        
        self.label_scheme = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.targets = data[:,1] # Define labels (y)
        self.numerical_labels = numpy.array([self.label_scheme[item] for item in self.targets])
        self.one_hot_numerical_labels = to_categorical(self.numerical_labels, 3)
    
    def get_data(self):
        '''Getter method for raw_data attribute of Dataset class.'''
        return self.raw_data

    def get_clean_sequences(self):
        '''Getter method for cleaned data.'''
        return self.clean_sequences
    
    def get_vocab_info(self):
        '''Getter method for vocab_attributes
        '''
        return self.vocab, self.vocab_size
    
    def get_one_hot_numerical_labels(self):
        '''Getter method for one-hot-encoded labels
        '''
        return self.one_hot_numerical_labels


dataset = Dataset(filename = 'Data/Tweets.csv') 
clean_tweets = dataset.get_clean_sequences() # List 
vocab, vocab_size = dataset.get_vocab_info() # Returns a set and an int, respectively
one_hot_numerical_labels = dataset.get_one_hot_numerical_labels()

