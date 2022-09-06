#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue May  5 13:24:27 2020
# @author: miranda (upquark00)

import time

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from glove_data import train_x, test_x, train_y, test_y

start = time.time()

class RNN: 
    
    def __init__(self, batch_size = 64, epochs = 500, num_classes = 3):
        '''
        Constructor for class RNN. Defines hyperparameters of recurrent neural
        network via additive layering of the Tensorflow.Keras Sequential model.
    
        Args:
            batch_size (int or None): number of samples per gradient update.
            epochs (int): number of epochs to train the model. An eopch is an 
                iteration over the entire 'x' and 'y' data provided. Trainng 
                ends once the epoch whose index is 'epochs' is reached. 
            num_classes (int): equal to the number of categorical labels.
        
        Returns: None
                    
        '''
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes # Output of Dense is (None, num_classes)
        
        self.input_length = 35 # Length of input sequences, ie, length of padded sentences 
        # self.input_dim = dataset.vocab_size() # Num unique tokens in the dataset; needed for Embedding
        self.output_dim = 50 # Dimensionality of space into which words will be embedded. 
        self.model, self.history, self.early_stopping = self.build_RNN()
        
    def build_RNN(self):
        '''
        Creates RNN model using a linear stack of layers with the Tensorflow.Keras
        Sequential model. 
        
        Args: None
        
        Returns: 
            model (tensorflow.python.keras.engine.sequential.Sequential object): 
                groups a linear stack of layers into tf.keras.Model. Each layer 
                has exactly one input and one output tensor. 
            history (tensorflow.python.keras.callbacks.History object): 'History'
                object which records events and is returned by the 'fit' 
                func of Keras models
            early_stopping (tensorflow.python.keras.callbacks.EarlyStopping object):
                monitors training and stops if no improvement in validation loss. 
        
        '''
        model = Sequential()
        
        # Can also add an Embedding layer here for transfer learning.     
        model.add(LSTM(units = 25, input_shape = (35,50), 
                       return_sequences = True, 
                       dropout= 0.20))
        # model.add(Dropout(0.2))
        model.add(LSTM(units = 15, dropout= 0.1))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(loss = 'categorical_crossentropy', 
                     optimizer = 'adam', 
                     metrics = ['accuracy', 
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
        
        model_checkpoint = ModelCheckpoint('model.h5', 
                                           monitor='val_loss', 
                                           mode='min', 
                                           verbose=1)
        
        history = model.fit(train_x, 
                            train_y,
                            batch_size = self.batch_size, 
                            epochs = self.epochs,
                            validation_split = 0.33,
                            callbacks = [early_stopping, model_checkpoint]
                            )
        
        return model, history, early_stopping
    
    def visualize(self):
        '''
        Plots model results.
        
        Args: None
        Returns: None    
        
        '''
            
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy vs. Epoch')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss vs. Epoch')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['precision'])
        plt.plot(self.history.history['val_precision'])
        plt.title('Model Precision vs. Epoch')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['recall'])
        plt.plot(self.history.history['val_recall'])
        plt.title('Model Recall vs. Epoch')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

RNN = RNN()
model, history, early_stopping = RNN.build_RNN()

model.summary()
loss, accuracy, precision, recall, TP, TN, FP, FN = model.evaluate(test_x, test_y)
RNN.visualize()

print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))
print('Training stopped after', early_stopping.stopped_epoch,'epochs.')

model.save("model.h5")

print("Model saved.")
print('Script completed after',time.time() - start, 'seconds.')
