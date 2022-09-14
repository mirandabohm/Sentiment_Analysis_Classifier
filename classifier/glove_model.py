#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue July 16 13:24:27 2020
# @author: miranda (upquark00)

import numpy 
import os 
import pickle

# TODO: rename load_glove, save_glove etc. simply "load", "save" 

class GloVe:
    
    def __init__(self, filepath = 'data/glove_twitter_50d.txt'):
        
        '''
        Constructor for class GloVe.
        
        Args: 
            filepath (string): path to the location of the downloaded file containing 
                pre-trained GloVe vectors. Must be .txt file type. These are available
                at https://nlp.stanford.edu/projects/glove/.
        
        Returns: None
        
        '''
        
        self.filepath = filepath
   
    def make_glove_model(self):
        ''' 
        Loads GloVe pre-trained word vectors from a designated .txt file, returning 
        a dictionary called GloVe_dict. Each dictionary key is a single GloVe word 
        (token) as a string. Each dictionary value is a numpy array whose dimension 
        is (features,), i.e. (50,), containing a real-valued vector representation 
        of the corresponding key token. 
        
        Args: None
           
        Returns: 
            GloVe_dict (dict): each key is a string representing a token (string)
                contained in pre-trained GloVe model. "Token" signifies a single 
                word which may contain symbols but no spaces. Each corresponding
                value is a numpy array containing the real-valued feature vector
                of that word. Its size is (features,).
            
            average_vector (numpy.ndarray): 1D array of size (features,), i.e. (50,).
                Contains the arithmetic mean or "average" of word vectors in the 
                GloVe model. A reasonable substitute for the vectors of missing 
                words per the paper's original author.
        ''' 
    
        print("gloVe vectors loading . . .")
        
        with open(self.filepath,'r', encoding='utf8') as foo:
            GloVe_dict = {}
            for line in foo:
                splitLines = line.split()
                word = splitLines[0]
                wordEmbedding = numpy.array([float(value) for value in splitLines[1:]])
                GloVe_dict[word] = wordEmbedding
        
        # TODO: create a separate function that creates average_vector 
        # TODO: make this faster 
        
        # Get average of word vectors to be used for unseen words, per GloVe author
        with open(self.filepath, 'r', encoding='utf8') as foo:
            for i, line in enumerate(foo):
                pass
    
        n_vec = i + 1
        hidden_dim = len(line.split(' ')) - 1
        
        vecs = numpy.zeros((n_vec, hidden_dim), dtype=numpy.float32)
        
        with open(self.filepath, 'r', encoding='utf8') as foo:
            for i, line in enumerate(foo):
                vecs[i] = numpy.array([float(n) for n in line.split(' ')[1:]], dtype=numpy.float32)
        
        average_vector = numpy.mean(vecs, axis=0)
        print(len(GloVe_dict),"gloVe vectors loaded.")
        
        return GloVe_dict, average_vector

    def save(self):
        '''
        Calls make_glove_model to create and save pickled versions of the GloVe 
            model dictionary and a calculated average_vector (used as a standin for
            missing words).
    
        Args: None 
        
        Returns:
            saved_glove_model.pkl: serialized object containing glove_model
            saved_avg_vec.pkl: a serialized object containing average_vector

        '''
        
        glove_model, average_vector = self.make_glove_model()
    
        with open('saved_glove_model.pkl', 'wb') as f:
            pickle.dump(glove_model, f)
                
        with open('saved_avg_vec.pkl', 'wb') as f:
            pickle.dump(average_vector, f)

    def load(self):
        '''
        Loads GloVe vectors and computed average vector from separate .pkl files.
        
        Args: None
        
        Returns:
            GloVe_dict (dict): each key is a string representing a token (string)
                contained in pre-trained GloVe model. "Token" signifies a single 
                word which may contain symbols but no spaces. Each corresponding
                value is a numpy array containing the real-valued feature vector
                of that word. Its size is (features,).
            
            average_vector (numpy.ndarray): 1D array of size (features,), i.e. (50,).
                Contains the arithmetic mean or "average" of word vectors in the 
                GloVe model. A reasonable substitute for the vectors of missing 
                words per the paper's original author.
        '''
        
        with open('saved_glove_model.pkl', 'rb') as f:
            GloVe_dict = pickle.load(f)
        
        with open('saved_avg_vec.pkl', 'rb') as f:
            average_vector = pickle.load(f)
        
        return GloVe_dict, average_vector

def main():
    GloVe_Model = GloVe()

    if not os.path.isfile('saved_glove_model.pkl'):
        GloVe_Model.save(filepath = 'data/glove_twitter_50d.txt' )

if __name__ == "__main__":
    main()

