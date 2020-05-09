#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:00:03 2020
@author: miranda
"""

import re 
from string import punctuation 
from tensorflow import keras

def remove_handles(tweets_array):
    ''' 
    Takes in a numpy array of shape (m,), containing Tweets (strings). 
    Returns a numpy array of same shape, with all @mentions removed. 
    I.e., ''@VirginAmerica What @dhepburn said.' becomes 
    'What @dhepburn said.'
    '''
    for i in range(len(tweets_array)):
            tweets_array[i] = re.sub(r'@[A-Za-z0-9]+','',tweets_array[i])
            # print(tweets_array[i])
    return tweets_array

def make_lowercase(tweets_array):
    ''' Takes in a numpy array; returns a list of strings.'''
    return [tweet.lower() for tweet in tweets_array]

def remove_links(tweets_list):
    '''
    Takes in a list of Tweets (strings) and removes URLS.
    Returns a list of strings. 
    '''
    return [re.sub('https?://[A-Za-z0-9./]+','', tweet) for tweet in tweets_list]

def remove_punct(tweets_list):
    return [tweet.translate(str.maketrans('', '', punctuation)) for tweet in tweets_list] 

def tokenize(tweets_list):
    return [tweet.split(' ') for tweet in tweets_list]

def remove_spaces(tokenized_tweets_list):
    for i in range(len(tokenized_tweets_list)):
        tokenized_tweets_list[i] = [word for word in tokenized_tweets_list[i] if word != '']
    return tokenized_tweets_list

# TODO: add a function that removes/filters "words" comprised of only digits. 
    # Deal with emojis. 

def process_data(tweets_list):
    '''Tokenize and clean data. Defined in process_text.py.
    Includes tokenization, as well as removal of spaces, punctuation, 
    Twitter @mentiones, and links; lowercases words.
    ''' 
    return remove_spaces(tokenize(remove_punct(remove_links(make_lowercase(remove_handles(tweets_list))))))









