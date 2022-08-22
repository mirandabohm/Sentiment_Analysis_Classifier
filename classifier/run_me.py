#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy
import pickle
from tensorflow.keras.models import load_model
from process_text import process_data
from glove_data import build_stacked_embedding_array, glove_model, average_vector
from get_inputs import dataset

cols = len(average_vector)
model = load_model('model.h5')
label_scheme = dataset.get_label_scheme() 

user_input = ''
while user_input != 'n':
    user_input = numpy.array([input('Enter a sentence to evaluate its sentiment: ',)])
    processed_data = process_data(user_input)
    prediction = model.predict(build_stacked_embedding_array(processed_data, glove_model, average_vector, cols)[0])
    likelihood = numpy.amax(prediction)
    decision = [key for key, value in label_scheme.items() if value == numpy.argmax(prediction)][0]
    print('Sentiment is {} with {:.2f}% likelihood.'.format(decision, likelihood * 100))
    user_input = (input('Evaluate another? [Y/N]: ')).lower()