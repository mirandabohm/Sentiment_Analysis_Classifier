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
    
    '''
    
    def __init__(self, filename):
        '''
        The constructor for Dataset class.
        
        Parameters:
            filename (string): path to Tweets data file. Accepts .csv format
            
        Defines attributes: 
            raw_data (numpy.ndarray): dimensions are (14640, 15)

        '''
        self.raw_data = self.load_data(filename)
        self.define_data()
        self.set_labels(self.raw_data)
        
    def load_data(self, filename):
        '''
        Loads .csv data into Dataset class for processing and analysis. 
        
        Parameters:
            filename (string): path and filename including file extension.
        
        Returns: 
            raw_data (numpy.ndarray): dimensions are (examples, features)
        '''
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            raw_data = numpy.array(list(reader))
            return raw_data
        
    def define_data(self):
        '''Setter method for class Dataset. Defines: 
            
            clean_sequences (list): each list contains lists of strings of length
                n, where n is equal to the number of examples in the data set (14640).
                Each string is a cleaned and tokenized sentence.
            
            longest_sequence (int): finds the length of the longest sequence 
                (sentence) in the dataset (35 characters).
                
            vocab (set): a set containing a list of all unique vocabulary items
                contained in the cleaned sequences attribute.
                
            vocab_size (int): length of the set vocab, i.e. number of unique
                words in the cleaned and tokenized sequences dataset.
            
        Internal variables:
            tweets (numpy.ndarray): contains features or vector <x> fed to model.
                Dimensions are (10,). 
        '''
        
        tweets = self.raw_data[:,10]
        self.clean_sequences = process_data(tweets)
        self.vocab = set(chain.from_iterable(self.clean_sequences))
        self.vocab_size = len(self.vocab) + 1 # Unique words in the set
        self.longest_sequence = len(max(self.clean_sequences, key=len))
            
    def set_labels(self, data):
        '''
        Setter method for label classes.
        
        Parameters: 
            data (numpy.ndarray): dimensions are (examples, features)
            
        Defines: 
            label_scheme (dict): contains mutually exclusive categorical labels
                "negative" = 0, "neutral" = 1, "positive" = 2. 
                
            targets (numpy.ndarray): defines a label vector <y> containing str
                data labels "negative", "neutral", and "positive" to be used
                by the model. Dimensions are (14640,)
            
            numerical_labels (numpy.ndarray): contains labels as int data valued
                between 0 and 2; dimensions are (14640,)
                
            one_hot_numerical_labels (numpy.ndarray): one-hot-encoded vector
                of dimensions (14640, 3).
                
        '''
        
        self.label_scheme = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.targets = data[:,1]
        self.numerical_labels = numpy.array([self.label_scheme[item] for item in self.targets])
        self.one_hot_numerical_labels = to_categorical(self.numerical_labels, 3)
    
    def get_data(self):
        '''Getter method for raw_data attribute of Dataset class.'''
        return self.raw_data

    def get_clean_sequences(self):
        '''Getter method for clean_sequences attribute of Dataset class.'''
        return self.clean_sequences
    
    def get_vocab_info(self):
        '''Getter method for vocab_size attribute of Dataset class.'''
        return self.vocab, self.vocab_size
    
    def get_one_hot_numerical_labels(self):
        '''Getter method for one_hot_numerical_labels attribute of Dataset class.'''
        return self.one_hot_numerical_labels


dataset = Dataset(filename = 'Data/Tweets.csv') 
clean_tweets = dataset.get_clean_sequences() # List 
vocab, vocab_size = dataset.get_vocab_info() # Returns a set and an int, respectively
one_hot_numerical_labels = dataset.get_one_hot_numerical_labels()

