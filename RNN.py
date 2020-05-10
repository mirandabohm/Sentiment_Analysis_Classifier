#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np
import matplotlib.pyplot as plt
from get_inputs import dataset
from glove_data import train_x, test_x, train_y, test_y, glove_model, avg_vec, build_stacked_embedding_array
from process_text import process_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =============================================================================
# - - - - - - - - - The Easier Way (Using Keras) - - - - - - - - -
    
batch_size = 64
num_epochs = 15
num_classes = 3 # Output of Dense is (None, num_classes)

input_length = 35 # Length of input sequences, ie, length of padded sentences 
input_dim = dataset.vocab_size # Num unique tokens in the dataset; needed for Embedding
output_dim = 50 # Dimensionality of space into which words will be embedded. 

model = Sequential()

# Can also add an Embedding layer here for transfer learning.     
model.add(LSTM(units = 50, input_shape = (35,50), return_sequences = True, dropout= 0.25))
model.add(Dropout(0.2))
model.add(LSTM(units = 15, dropout= 0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', 
             optimizer = 'adam', 
             metrics = ['accuracy'])

history = model.fit(train_x, train_y,
          batch_size = batch_size, 
          epochs = num_epochs,
          validation_split = 0.33
          )

model.summary()
loss, accuracy = model.evaluate(test_x, test_y)
print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy vs. Epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
    
user_input = 'yes'
while user_input != 'no':
    user_input = np.array([input('Enter a sentence to evaluate its sentiment: ',)])
    p = process_data(user_input)
    ans = model.predict(build_stacked_embedding_array(p, glove_model, avg_vec)[0])
    likelihood = np.amax(ans)
    decision = [key for key, value in dataset.label_scheme.items() if value == np.argmax(ans)][0]
    print('Sentiment is {} with {}% likelihood.'.format(decision, likelihood * 100))
    user_input = (input('Evaluate another sentence? Enter "No" to exit. ')).lower()

# =============================================================================
# - - - - - - - - - - - The Harder, Math-y Way - - - - - - - - - - - 

# # Define model hyperparameters 
# learning_rate = 0.0001    
# # epochs = 25               
# T = 50 # length of gloVe training vectors; also padded tweet sentences 
# hidden_dim = 100 # Number units in hidden layer 
# out_dim = 3 # Test various
# vocab_size = gi.vocab_size # Length of longest sequence 
# 
# # Initialize weights matrices 
# U = np.random.uniform(low=0, high=1, size=(hidden_dim,T)) # Weights btwn hidden/input layers
# W = np.random.uniform(low=0, high=1, size=(hidden_dim,hidden_dim)) # Weights between hidden units
# V = np.random.uniform(low=0, high=1, size=(out_dim,hidden_dim)) # Weights between hidden/output layers 
# 
# # Define sigmoid function, which will accept a vector of size hidden_dim x 1
# def sigmoid(array):
#     ''' Standard sigmoid activation function.''' 
#     # np.exp applies function to every element in the array 
#     return 1 / (1 + np.exp(-array))
# 
# bptt_truncate = 5 # Define number steps to perform BPTT
# min_clip_value = -10
# max_clip_value = 10
# 
# (Coming Soon)
#
# layer = Dropout(0.5)
# =============================================================================
