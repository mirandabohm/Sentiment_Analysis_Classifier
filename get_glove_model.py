#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue July 16 13:24:27 2020
# @author: miranda (upquark00)

import numpy as numpy 
import os 

def make_glove_model():
    ''' 
    Load gloVe pre-trained vectors. 
    Dict keys = tokens (strings); values = word vectors (numpy arrays of length 50). 
    ''' 
    filename = 'Data/glove_twitter_50d.txt'
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

if not os.path.isfile('glove_model.npy'):
    glove_model, avg_vec = make_glove_model()
    numpy.save('glove_model.npy', glove_model)  
    numpy.save('avg_vec.npy', avg_vec)  

def main():
    print('type: ', type(avg_vec))
    print('len: ', len(avg_vec))
    print('len of glove_model: ', len(glove_model))

if __name__ == "__main__":
    main()