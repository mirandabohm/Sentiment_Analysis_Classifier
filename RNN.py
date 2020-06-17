#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import matplotlib.pyplot as plt
from get_inputs import dataset
from glove_data import train_x, test_x, train_y, test_y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
start = time.time()

batch_size = 500
max_epochs = 500
num_classes = 3 # Output of Dense is (None, num_classes)

input_length = 35 # Length of input sequences, ie, length of padded sentences 
input_dim = dataset.vocab_size # Num unique tokens in the dataset; needed for Embedding
output_dim = 50 # Dimensionality of space into which words will be embedded. 

model = Sequential()

# Can also add an Embedding layer here for transfer learning.     
model.add(LSTM(units = 25, input_shape = (35,50), return_sequences = True, dropout= 0.20))
# model.add(Dropout(0.2))
model.add(LSTM(units = 15, dropout= 0.1))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', 
             optimizer = 'adam', 
             metrics = [
                 'accuracy', 
                 Precision(name='precision'), 
                 Recall(name='recall'),
                 TruePositives(name='TP'),
                 TrueNegatives(name='TN'),
                 FalsePositives(name='FP'),
                 FalseNegatives(name='FN')])

early_stopping = EarlyStopping(monitor='val_loss', 
                               mode='min', verbose=1, 
                               patience = 9,
                               restore_best_weights = False
                               )

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)

history = model.fit(train_x, train_y,
          batch_size = batch_size, 
          epochs = max_epochs,
          validation_split = 0.33,
          callbacks = [early_stopping, model_checkpoint]
          )

model.summary()
loss, accuracy, precision, recall, TP, TN, FP, FN = model.evaluate(test_x, test_y)
print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))
print('Training stopped after',early_stopping.stopped_epoch,'epochs.')

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

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Precision vs. Epoch')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall vs. Epoch')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save("model.h5")
print("Model saved.")

print('Script completed after',time.time() - start, 'seconds.')

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
