#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

# TODO: add mechanism by which to run RNN.py in the case that model.h5 is 
# absent from the directory. 

import numpy
from tensorflow.keras.models import load_model

import get_inputs
from process_text import clean
from glove_data import build_stacked_embedding_array, glove_model, average_vector

# TODO: create model.h5 if it doesn't exist 

dataset = get_inputs.Dataset()
cols = len(average_vector)
model = load_model('model.h5')
label_scheme = dataset.label_scheme() 

user_input = ''
while user_input != 'n':
    user_input = numpy.array([input('Enter a sentence to evaluate its sentiment: ',)])
    cleaned_data = clean(user_input)
    prediction = model.predict(build_stacked_embedding_array(cleaned_data, glove_model, average_vector, cols)[0])
    likelihood = numpy.amax(prediction)
    decision = [key for key, value in label_scheme.items() if value == numpy.argmax(prediction)][0]
    print('Sentiment is {} with {:.2f}% likelihood.'.format(decision, likelihood * 100))
    user_input = (input('Evaluate another? [Y/N]: ')).lower()