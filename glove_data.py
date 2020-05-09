#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np
from get_inputs import clean_tweets, one_hot_numerical_labels

def get_glove_model():
    filename = 'glove_twitter_50d.txt'
    # Load gloVe pre-trained vectors. 
    # Dict Keys = Words (strings); Values = Word Vectors (np arrays of length 50). 
    print("gloVe vectors loading . . .")
    with open(filename,'r') as foo:
        gloveModel = {}
        for line in foo:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
            
    # Get average of word vectors to be used for unseen words, per GloVe author
    with open(filename, 'r') as foo:
        for i, line in enumerate(foo):
            pass
    n_vec = i + 1
    hidden_dim = len(line.split(' ')) - 1
    
    vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)
    
    with open(filename, 'r') as foo:
        for i, line in enumerate(foo):
            vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
    
    average_vec = np.mean(vecs, axis=0)
    print(len(gloveModel),"gloVe vectors loaded.")
    return gloveModel, average_vec

def build_single_embedding_array(tweet, model, average_vector):
    ''' Construct a matrix of word vectors for each tweet with an added axis 0.
    Rows = Timesteps, eg Word Vectors; Columns = Feature Vectors.
    Returns a 3D array of size ( 1 x # Padded Sequence Length x # Features),
    or (1 x 35 x 50). 
    
    Inputs: tweet must be a list of strings. ''' 
    # Create 2D array of zeros to be populated. 
    rows = len(tweet) # Words in the tweet
    cols = 50 # Length of pre-trained word vectors
    embedding_matrix = np.zeros([rows, cols])
    missing_words = []
    for index in range(len(tweet)): 
        # Return the word vector corresponding to each word in the tweet (sequence)
        try:
            word_vector = model[tweet[index]] 
        except KeyError: 
            word_vector = average_vector
            # missing_words[tweet[index]] = index
            missing_words.append(tweet[index])
        finally: 
            # Each row is a separate word vector (parenthesis)
            embedding_matrix[index] = word_vector
    
    # Add a dimension to the 2D embedding array along axis zero. This satisfies
    # dimensionality requirements such that it can be added to the stacked array.
    embedding_matrix = np.expand_dims(embedding_matrix, axis = 0)
    
    # Pad second dimension with zeroes to standardize sequence length. 
    # Set final dimension to the length of the longest sequence (35). 
    formatted_embedding_matrix = np.pad(embedding_matrix, ((0,0),(0,35-len(tweet)),(0,0)),'constant')
    return formatted_embedding_matrix, missing_words
   
def build_stacked_embedding_array(tweets_list, model, average_vector):
    ''' Builds a 3-dimensional numpy array with the following dimensions: 
        ( # Instances x Padded Sequence Length # Features ), or
        ( # Tweets in Set x # Words in Each Padded Tweet x # Word Vector Length)
        For Tweets.csv, this is 14640 x 35 x 50. 
        
        Inputs: tweets_list will be a list of lists containing strings. ''' 
    
    total_missing_words = []
    axis0 = len(tweets_list)
    large_embedding_matrix = np.zeros([axis0,35,50])
    for j in range(axis0):
        single_array, missing_words = build_single_embedding_array(tweets_list[j], model, average_vector)
        large_embedding_matrix[j] = single_array
        for i in missing_words:
        # for [m,n] in missing_words.items():
            # total_missing_words[m] = j
            total_missing_words.append(i)
    return large_embedding_matrix, total_missing_words

# Missing words is a dictionary of all Tweeted words that are not in our GloVe model.
glove_model, avg_vec = get_glove_model()
stacked_embedding_array, missing_words = build_stacked_embedding_array(clean_tweets, glove_model, avg_vec)

# =============================================================================
# train_size = round(.7 *len(clean_tweets)) # 10248
# test_size = round(.1 *len(clean_tweets)) # 1464
# validate_size = round(.2 *len(clean_tweets)) # 2928
# test_stop = train_size + test_size # marks divider btwn train and test data
# =============================================================================

# Split data 
# train_x = stacked_embedding_array[0:train_size,:,:] # (10248, 35, 50)
# test_x = stacked_embedding_array[train_size:test_stop,:,:] # (1464, 35, 50)

train_size = round(len(stacked_embedding_array) * .8) # 11712

x_data = stacked_embedding_array
train_x = x_data[:train_size] # (11712, 35, 50)
test_x = x_data[train_size::] # (2928, 35, 50)

y_data = one_hot_numerical_labels
train_y = one_hot_numerical_labels[:train_size] # (11712, 3)
test_y = one_hot_numerical_labels[train_size::] # (2928, 3)

# =============================================================================
# validate_x = stacked_embedding_array[test_stop::,:,:] # (2928, 35, 50)
# validate_y = one_hot_numerical_labels[test_stop:] # (2928,))
# =============================================================================

def main():
    print('Module finished.')
# =============================================================================
#     train_x, test_x, train_y, test_y, validate_x, validate_y = build_test_train_validate_sets(clean_tweets)
#     print(train_x, test_x)
#     print(train_y, test_y)
#     print(validate_x, validate_y)
# =============================================================================
    
if __name__ == "__main__":
    main()
    
#for tweet in tweets_list:
 #   create_readable_sequence(tweet)
# Create an input tensor where each element is result of running create_readable
# on all tweets in the set. 
# Final shape of the returned tensor 
# # tweets x length of the longest tweet (35) x # features in each feature vector (50))

        









