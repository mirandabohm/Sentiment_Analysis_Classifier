#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

'''
TODO: 8/19/2022: rename module to "format_data" or something similar. 

TODO: 9/11/22: 
    1. Make terms consistent throughout documentation and var names
    2. Clean up embedding array function
    3. Lines 80 and 96 are redundant 
    
TERMS USED IN THIS SCRIPT
    - Sequence = Tweet = Sentence composed of Timesteps (Words)
    - Timesteps = Words in each Tweet/Sequence
    - Columns ()
    - Rows = Number of timesteps/words in each Tweet/Sequence
    - Cols = int, fixed length of pre-trained GloVe word vectors (50)

'''

import numpy
from time import time 

import get_inputs
import glove_model

start = time()
    
GloVe_filepath = 'data/glove_twitter_50d.txt' 
GloVe_Model = glove_model.GloVe()
dataset = get_inputs.Dataset()
    
try: 
    glove_model, average_vector = GloVe_Model.load_glove()
except:
    GloVe_Model.save_glove()
    glove_model, average_vector = GloVe_Model.load_glove()
  
def build_stacked_embedding_array(clean_sequences, model, average_vector, cols):
    ''' 
    Builds a 3-dimensional numpy array with the following dimensions: 
    ( # Instances x Padded Sequence Length x # Features ), or
    ( # Tweets in Set x # Words in Each Padded Tweet x # Word Vector Length)
    For Tweets.csv, this is 14640 x 35 x 50. 
        
    Args: 
        clean_sequences (list): each element is a list representing a single tweet; 
            each element of this inner list is a string representing a tokenized
            word from a cleaned and processed Tweet. 
            
            each list contains lists of strings of length
            n, where n is equal to the number of examples in the data set (14640).
            Each string is a cleaned and tokenized sentence.
            
        model (dict): each key is a string representing a word contained in the
            GloVe model. Each corresponding value is a numpy.ndarray of shape 
            (Features,); in this case (50,).
            
        average_vector (numpy.ndarray): 1D array of size (features,), i.e. (50,).
            Contains the arithmetic mean or "average" of word vectors in the 
            GloVe model. A reasonable substitute for the vectors of missing 
            words per the paper's original author. 
            
        cols (int): length of pre-trained word vectors
            
    Returns: 
       large_embedding_matrix (numpy.ndarray): 3D numpy array of shape        
           (instances, padded sequence length, features), i.e. (14640, 35, 50).
          
        missing_words (set): each element is a word (string) appearing in a Tweet
            but not found in the keys of the pre-trained GloVe model dictionary. 
            If a word is in this list, the computed average_vector will be used
            in place of its (nonexistent) GloVe vector in the embedding array.

    ''' 
    
    missing_words = set()
    axis0 = len(clean_sequences)
    large_embedding_matrix = numpy.zeros([axis0, 35, cols])
    
    def build_single_embedding_array(clean_sequences):
        '''
        Construct a matrix of word vectors for each tweet with an added axis 0.
        Rows = Timesteps, e.g. word vectors; columns = feature vectors.
        Returns a 3D array of size ( 1 x # Padded Sequence Length x # Features),
        or (1 x 35 x 50). 
                     
        Returns:
            padded_embedding_matrix (numpy.ndarray): 3D matrix of size (1, 35, 50), 
            i.e. (1, length of padded sequences, features). 

        '''
               
        rows = len(clean_sequences) # Num words in the tweet
        embedding_matrix = numpy.zeros([rows, cols])
        
        for index, word in enumerate(clean_sequences): 
            try: 
                word_vector = model[word]
            except KeyError: 
                word_vector = average_vector
                missing_words.add(word)
            finally: 
                # One row = one word vector
                embedding_matrix[index] = word_vector

        # Add a dimension to the 2D embedding array along axis zero. This satisfies
        # dimensionality requirements such that it can be added to the stacked array.
        embedding_matrix = numpy.expand_dims(embedding_matrix, axis = 0)
        
        # Pad second dimension with zeroes to standardize sequence length. 
        # Set final dimension to the length of the longest sequence (35). 
        padded_embedding_matrix = numpy.pad(embedding_matrix, 
                                               ((0, 0),
                                                (0, 35 - rows),
                                                (0, 0)),
                                               'constant')
        
        return padded_embedding_matrix, missing_words
    
    for j in range(axis0):
        single_array, missing_words = build_single_embedding_array(clean_sequences[j])
        large_embedding_matrix[j] = single_array
            
    return large_embedding_matrix, missing_words

train_percent = 0.80
cols = len(average_vector)
                
clean_sequences = dataset.clean_sequences()
stacked_embedding_array, missing_words = build_stacked_embedding_array(
    clean_sequences, 
    glove_model, 
    average_vector, 
    cols)

train_size = round(len(stacked_embedding_array) * train_percent) # 11712

x_data = stacked_embedding_array
train_x = x_data[:train_size] # (11712, 35, 50)
test_x = x_data[train_size::] # (2928, 35, 50)

one_hot_numerical_labels = dataset.one_hot_numerical_labels()
y_data = one_hot_numerical_labels  # TODO: Why not set y_data directly?
train_y = one_hot_numerical_labels[:train_size] # (11712, 3)
test_y = one_hot_numerical_labels[train_size::] # (2928, 3)

end = time()

def main():
    print('Module finished.')
    print('Runtime: ', str(end-start))
   
if __name__ == "__main__":
    main()
    







