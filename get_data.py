
"""
Created on Fri May, 1 17:46:51 2020
@author: miranda
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RUN THE MODEL 
# Output should be an ordered set of not-necessarily equally-sized
# matrices. Each matrix will have the same number of rows, but not necessarily
# the same number of columns. 

"""

import pandas as pd
import numpy as np 
import tensorflow_hub as hub
import csv
import process_text as pt

# Import sentiment data 
data = pd.read_csv('Tweets.csv', 
                   header = None, 
                   delimiter='\t')

# Segment the data 
sentiments = data[:,1]    
tweets = data[:,10]
encoded_sentiments = pt.encode(sentiments)

def embed(string_array):
    '''
    Run Google Automatic Sentence Encoder.
    Take in a 
    '''
    model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    return model(string_array)

def give_data(tweets):
    stripped_tweets = pt.remove_handles(tweets)
    return embed(stripped_tweets)

# each input is a sequence of 
# take the trace of the covariance matrix of the np array of the output tensor
# covariance vector - 
    






