#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import csv
from itertools import chain

import numpy
from tensorflow.keras.utils import to_categorical

from process_text import clean

# TODO: Rename module "Tweets Data" or similar
# TODO: replace getter/setter methods and private vars, Effective Python pg. 182 

class Dataset:
    '''
    A class containing dataset vectors, intended to take raw, 
    unformatted text and convert and store its cleaned and tokenized versions.
    It also creates and carries one-hot-encoded labels for said data. 
    
    '''
    
    def __init__(self, filename = 'Data/Tweets.csv'):
        '''
        The constructor for class  Dataset.
        
        Args:
            filename (string): path to Tweets data file, which should be a .csv
        
        Returns: None 
        
        Sets attributes: 
            raw_data (numpy.ndarray): dimensions are (14640, 15)

        '''
        self.__filename = filename
        self.__raw_data = self.load_data()
        self.define_data()
        self.set_labels()
        
    def load_data(self):
        '''
        Loads .csv data for processing and analysis. 
               
        Returns: 
            raw_data (numpy.ndarray): dimensions are (examples, features)
            
        '''
        with open(self.__filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            self.__raw_data = numpy.array(list(reader))
            return self.__raw_data
        
    def define_data(self):
        '''
        Setter method for class Dataset. 
        
        Args: None
        Returns: None 
        
        Sets attributes:
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
        tweets = self.__raw_data[:,10]
        self.__clean_sequences = clean(tweets)
        self.__vocab = set(chain.from_iterable(self.__clean_sequences))
        self.__vocab_size = len(self.__vocab) + 1 # Unique words in the set
        self.longest_sequence = len(max(self.__clean_sequences, key=len))
            
    def set_labels(self):
        '''
        Setter method for label classes.
        
        Args: None 
        Returns: None
        
        Sets attributes:
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
        self.__label_scheme = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.__targets = self.__raw_data[:,1]
        self.__numerical_labels = numpy.array([self.__label_scheme[item] for item in self.__targets])
        self.__one_hot_numerical_labels = to_categorical(self.__numerical_labels, 3)
    
    def data(self):
        '''Getter method for Dataset.__raw_data attribute..'''
        return self.__raw_data

    def clean_sequences(self):
        '''Getter method for Dataset.__clean_sequences attribute.'''
        return self.__clean_sequences
    
    def label_scheme(self):
        '''Getter method for Dataset.__label_scheme attribute.'''
        return self.__label_scheme
    
    def vocabulary(self):
        '''Getter method for Dataset.__vocab attribute.'''
        return self.__vocab
    
    def vocab_size(self):
        '''Getter method for Dataset__vocab_size attribute.'''
        return self.__vocab_size
    
    def one_hot_numerical_labels(self):
        '''Getter method for Dataset.__one_hot_numerical_labels attribute.'''
        return self.__one_hot_numerical_labels

def main():
    dataset = Dataset(filename = 'Data/Tweets.csv') 
    clean_tweets = dataset.clean_sequences() # List 
    vocab = dataset.vocabulary() # Returns a set
    vocab_size = dataset.vocab_size() # Returns an int
    one_hot_numerical_labels = dataset.one_hot_numerical_labels() 
    print('Module finished.')
   
if __name__ == "__main__":
    main()
    

