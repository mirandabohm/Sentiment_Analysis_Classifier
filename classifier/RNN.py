#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import time
import matplotlib.pyplot as plt
from get_inputs import dataset
from glove_data import train_x, test_x, train_y, test_y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

start = time.time()

class RNN: 
    '''
    Builds an RNN using the Keras.Sequential model. 
    '''
    
    batch_size = 64
    max_epochs = 500
    num_classes = 3 # Output of Dense is (None, num_classes)
    
    input_length = 35 # Length of input sequences, ie, length of padded sentences 
    input_dim = dataset.get_vocab_size() # Num unique tokens in the dataset; needed for Embedding
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
    
    model_checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1)
    
    history = model.fit(train_x, train_y,
              batch_size = batch_size, 
              epochs = max_epochs,
              validation_split = 0.33,
              callbacks = [early_stopping, model_checkpoint]
              )

def visualize(history):
    '''Plots model results.
    
    Parameters: 
            history (tensorflow.python.keras.callbacks.History object): 'History'
                object which records events and is returned by the 'fit' 
                func of Keras models
    '''
        
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

model = RNN()
model.summary()
loss, accuracy, precision, recall, TP, TN, FP, FN = model.evaluate(test_x, test_y)
visualize(history)

print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))
print('Training stopped after',early_stopping.stopped_epoch,'epochs.')

model.save("model.h5")
print("Model saved.")

print('Script completed after',time.time() - start, 'seconds.')
