#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import numpy as np
import matplotlib.pyplot as plt
from get_inputs import vocab_size
from glove_data import train_x, test_x, train_y, test_y, glove_model, avg_vec, build_stacked_embedding_array
from process_text import process_data

# =============================================================================
# - - - - - - - - - The Easier Way (Using Keras) - - - - - - - - -

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

input_dim = vocab_size # Number of unique words in the Twitter dataset 
output_dim = 50 # Dimensionality of space into which words will be embedded. 
        # Needs to be set to the length of gloVe vectors (50, in our case).
input_length = 35 # Length of input sequences, ie, length of padded sentences 
num_classes = 3

model = Sequential()
# Add an optional Embedding layer for transfer learning. 
# model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length))

model.add(LSTM(units = 50, input_shape = (35,50), return_sequences = True, dropout= 0.25))
model.add(LSTM(units = 15, dropout= 0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', 
             optimizer = 'adam', 
             metrics = ['accuracy'])

# Set training parameters 
batch_size = 264
num_epochs = 15

history = model.fit(train_x, train_y,
          batch_size = batch_size, 
          epochs = num_epochs,
          validation_split = 0.33
          )

model.summary()
loss, accuracy = model.evaluate(test_x, test_y)
print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))

# Make prediction
user_input = np.array([input('Enter a sentence to evaluate its sentiment: ',)])
p = process_data(user_input)
ans = model.predict(build_stacked_embedding_array(p, glove_model, avg_vec)[0])
decision = np.argmax(ans)

if decision == 2:
    print('Negative')
elif decision == 0:
    print('Neutral')
elif decision == 1:
    print('Positive')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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
# =============================================================================


 # for every input sequence (tweet),a single output vector with as many values as you have units in lstm
# sequence length does not affect the dimensionality of the ouput (using default method).
# top layer = last layer 
# x_t is a word; each sequence of x_t-1 to X_1 is a tweet 
# inputs to LSTM: A 3D tensor with shape `[batch, timesteps, feature]`
# Return_sequences = True to retain output at each time point and provide 3D to next LSTM layer
