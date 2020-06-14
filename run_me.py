#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np
from tensorflow.keras.models import load_model
from process_text import process_data
from glove_data import glove_model, avg_vec, build_stacked_embedding_array
from get_inputs import dataset

model = load_model('best_model.h5')

user_input = 'yes'
while user_input != 'no':
    user_input = np.array([input('Enter a sentence to evaluate its sentiment: ',)])
    p = process_data(user_input)
    ans = model.predict(build_stacked_embedding_array(p, glove_model, avg_vec)[0])
    likelihood = np.amax(ans)
    decision = [key for key, value in dataset.label_scheme.items() if value == np.argmax(ans)][0]
    print('Sentiment is {} with {:.2f}% likelihood.'.format(decision, likelihood * 100))
    user_input = (input('Evaluate another sentence? Enter "No" to exit. ')).lower()

