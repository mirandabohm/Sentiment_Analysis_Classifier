#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import re 
import string

def process_data(tweets_array):
    '''
    Tokenizes and cleans data. Removes punctuation, Twitter @mentions, URLS,
    and extraneous spaces. Lowercases words.
    
    Parameters:
        tweets_array (numpy array): shape (m,), containing m number of Tweets,
            which are sequences of strings.

    Returns:
        list_of_lists (list): list of lists of strings of length n, where n is 
            equal to the number of examples in the data set.

    '''
    
    for i in range(len(tweets_array)):
        tweets_array[i] = re.sub(r'@[A-Za-z0-9]+','', re.sub(r'#[A-Za-z0-9]+','', tweets_array[i])).lower()                       
    
    list_of_lists = [tweet.lower() for tweet in tweets_array]
    list_of_lists = [re.sub('https?://[A-Za-z0-9./]+','', tweet) for tweet in list_of_lists]
    list_of_lists = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in list_of_lists] 
    list_of_lists = [tweet.split(' ') for tweet in list_of_lists]
    
    for i in range(len(list_of_lists)):
        list_of_lists[i] = [word for word in list_of_lists[i] if word != '']
    
    return list_of_lists

# TODO: add functionality to filter out strings containing only digits.  
# Deal with emojis, imbalanced classes, and the many words not in GloVe. 
# Enact explicit lemmatization
    
def main():
    print('Module finished.')
    # clean_tweets = process_data()
   
if __name__ == "__main__":
    main()
    






