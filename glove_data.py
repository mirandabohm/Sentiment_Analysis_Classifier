#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:24:27 2020
@author: miranda

"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from get_inputs import clean_tweets, average_vec

class Glove:
    
    # TODO: clean up this super messy code. This shouldn't be a class. 
    
    def __init__(self, filename):
        # Load gloVe pre-trained vectors. 
        # Each key is a word; each val is a np array of length 50. 
        print("gloVe vectors loading . . .")
        foo = open(filename,'r')
        self.gloveModel = {}
        for line in foo:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            self.gloveModel[word] = wordEmbedding
        print(len(self.gloveModel),"gloVe vectors loaded.")
    
    def get_glove_dict(self):
        return self.gloveModel
    
make_G = Glove('glove_twitter_50d.txt')
glove = make_G.get_glove_dict()

    
# take every tweet --> convert to sequence of word vectors 
# write function that works on an individual tweet and apply to every tweet

def build_2D_embedding_matrix(tweet):
    ''' Construct a matrix of word vectors for each tweet. 
    Columns = feature vectors; rows = timesteps or instances of words.''' 

    # Create matrix
    rows = len(tweet) # Words in the tweet
    cols = 50 # Length of pre-trained word vectors
    embedding_matrix = np.zeros([rows, cols])
    for i in range(len(tweet)): 
        # for each word in the tweet, call its word vector 
        try:
            word_vector = glove[tweet[i]] 
        except KeyError: 
            word_vector = average_vec
        finally: 
            # Each row is a separate word vector (parenthesis)
            embedding_matrix[i] = word_vector
    
    # Add a dimension to our embedding matrix along axis zero. This will allow
    # for it to be stacked in our 3D tensor containing all embedding matrices.
    embedding_matrix2 = np.expand_dims(embedding_matrix, axis = 0)
    
    # Pad second dimension (sequence length) with zeroes. 
    # Output will be a numpy array of dim (1 x 35 x 50)
    formatted_embedding_matrix = np.pad(embedding_matrix2, ((0,0),(0,35-len(tweet)),(0,0)),'constant')
    return formatted_embedding_matrix

# test_formatted_embedding_matrix = build_2D_embedding_matrix(clean_tweets[0])
# print('Dimensions of one embedding matrix in stack:', test_formatted_embedding_matrix.shape)
# print('AFTER:', embedding_matrix_for_all.shape)
   
def build_3D_embedding_matrix(all_tweets):
    # First thing's first: make empty rows in the larger matrix for our values. 
    axis0 = len(all_tweets)
    large_embedding_matrix = np.zeros([axis0,35,50])
    for j in range(axis0):
        large_embedding_matrix[j] = build_2D_embedding_matrix(all_tweets[j])
    return large_embedding_matrix
        
complete_3D_matrix = build_3D_embedding_matrix(clean_tweets)


#for tweet in all_tweets:
 #   create_readable_sequence(tweet)
# Create an input tensor where each element is result of running create_readable
# on all tweets in the set. 
# Final shape of the returned tensor 
# # tweets x length of the longest tweet (35) x # features in each feature vector (50))

# Create a multidimensional array with final dimensions of: 
# (X x Y x Z ), where X = 35, or # of words in padded tweets 
# Y = 50, or the length of each word vector 
# Z = # of tweets in the data set 
        









