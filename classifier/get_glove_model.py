#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue July 16 13:24:27 2020
# @author: miranda (upquark00)

import numpy 
import os 
import pickle

def make_glove_model():
    ''' 
    Load gloVe pre-trained vectors. 
    Dict keys = tokens (strings); values = word vectors (numpy arrays of length 50). 
    
    Parameters: 
        None
        
    Returns: 
        avg_vec (numpy.ndarray): 1D array of size (50,), i.e. (features,). 
            Contains the arithmetic mean or "average" of word vectors in the 
            GloVe model. A reasonable substitute for the vectors of missing 
            words per the  author.
    ''' 

    filename = 'data/glove_twitter_50d.txt'
    print("gloVe vectors loading . . .")
    with open(filename,'r', encoding='utf8') as foo:
        gloveModel = {}
        for line in foo:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = numpy.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
            
    # Get average of word vectors to be used for unseen words, per GloVe author
    with open(filename, 'r', encoding='utf8') as foo:
        for i, line in enumerate(foo):
            pass
    n_vec = i + 1
    hidden_dim = len(line.split(' ')) - 1
    
    vecs = numpy.zeros((n_vec, hidden_dim), dtype=numpy.float32)
    
    with open(filename, 'r', encoding='utf8') as foo:
        for i, line in enumerate(foo):
            vecs[i] = numpy.array([float(n) for n in line.split(' ')[1:]], dtype=numpy.float32)
    
    avg_vec = numpy.mean(vecs, axis=0)
    print(len(gloveModel),"gloVe vectors loaded.")
    return gloveModel, avg_vec


def get_models():
    glove_model, avg_vec = make_glove_model()

    with open('saved_glove_model.pkl', 'wb') as f:
        pickle.dump(glove_model, f)
            
    with open('saved_avg_vec.pkl', 'wb') as f:
        pickle.dump(avg_vec, f)


def main():
    
    if not os.path.isfile('saved_glove_model.pkl'):
        get_models()


if __name__ == "__main__":
    main()