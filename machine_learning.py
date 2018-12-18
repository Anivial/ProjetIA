# -*- coding: utf-8 -*-
'''
Function to train and evaluate neural networks using Keras

Organization:
    IRISA/Expression
'''

import sys
#from keras.utils import np_utils
from keras.models import Sequential, save_model, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import SGD, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU#, ParametricSoftplus
from keras.callbacks import Callback, EarlyStopping
import pandas as pd
import numpy as np
from keras import backend as K
import math
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.models import model_from_json
import json
import utils
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


seed=0
np.random.seed(seed)
'''
Set random seed for reproducibility
'''


model_param={
            'loss':'mean_squared_error',
            'optimizer':'adam',
            'nb_epoch':500,
            'batch_size':64,
            'verbose':1
}
'''
Keras model default paramaters
'''


def train_neural_network(output_file,
                         train_input,
                         train_output,
                         dev_input=None,
                         dev_output=None):
    '''
    Create a deep neural network
    
    Arg:
        output_file: file where the model will be saved
        train_input: training data input
        train_output: ground truth for the training data
        dev_input: dev data input if desired
        dev_output: dev data ground truth if desired (needed if dev_input is provided)
    
    Return:
        A neural network model
    '''
    assert len(train_input) != 0, "dataset must not be empty"

    np_input = np.asarray(train_input)
    np_output = np.asarray(train_output)
    np.random.seed(0)

    # Logging utils
    class TrainingHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.predictions = []
            self.i = 0
            self.save_every = 50

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.i += 1        
            if self.i % self.save_every == 0:        
                pred = self.model.predict(np_input)
                self.predictions.append(pred)
                
    class WeightHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.predictions = []
            self.i = 0
            self.save_every = 50

        def on_epoch_end(self, batch, logs={}):
            print self.model.get_weights()
    
    history = TrainingHistory()
    weights = WeightHistory()
    early_stopping = EarlyStopping(monitor="val_loss", patience=30, verbose=0)

    model = Sequential()

    # Exo 1, question 1, input: 4, output: 5
    #model.add(Dense(input_dim=1, output_dim=1, activation="linear"))

    # Exo 1, question 2, input: 4,5, output: 6
    #model.add(Dense(input_dim=2, output_dim=1, activation="linear"))

    # Exo 1, question 2, input: 4,5,6,7 output: 8
    #model.add(Dense(input_dim=5, output_dim=1, activation="linear"))

    # Exo 2, question 1, input: 4,5 output: 6
    #model.add(SimpleRNN(input_length=2, input_dim=1, output_dim=2, activation="linear", return_sequences=True))
    #model.add(SimpleRNN(output_dim=1, activation="linear", return_sequences=False, unroll=True))

    # Exo 2, question 2, input: 4,5,6,7 output: 8
    #model.add(SimpleRNN(input_length=4, input_dim=1, output_dim=4, activation="linear", return_sequences=True))
    #model.add(SimpleRNN(output_dim=1, activation="linear", return_sequences=False, unroll=True))

    # Exo 3, question 1, input: 4,5,6,7,8 output: 9
    model.add(SimpleRNN(input_length=5, input_dim=1, output_dim=32, activation="elu", return_sequences=True))
    model.add(SimpleRNN(input_dim=5, output_dim=16, activation="elu", return_sequences=True))
    model.add(SimpleRNN(output_dim=1, activation="linear", return_sequences=False, unroll=True))
    
    # Compile model
    model.compile(loss=model_param['loss'],
                  optimizer=model_param['optimizer'])
    

    print "Training..."
    print np_input.shape
    print np_input
    print "--"
    print np_output.shape
    print np_output

    if dev_input and dev_output:
        np_v_input = np.asarray(dev_input)
        np_v_output = np.asarray(dev_output)
        hist = model.fit(x=np_input,
                         y=np_output,
                         validation_data=(np_v_input, np_v_output),
                         nb_epoch=model_param['nb_epoch'],
                         batch_size=model_param['batch_size'],
                         verbose=model_param['verbose'],
                         callbacks=[history, early_stopping])
    else:
        hist = model.fit(np_input,
                         np_output,
                         nb_epoch=model_param['nb_epoch'],
                         batch_size=model_param['batch_size'],
                         verbose=model_param['verbose'],
                         callbacks=[history, early_stopping])
    
    print "Saving..."
    save_model(model, output_file)
    
    
    model.summary()
    
    return model




def evaluate(model,
            q_input,
            q_output,
            recurrent):
    '''
    Evaluate the model accuracy
    
    Args:
        model: Model to evaluate
        q_input: data input
        q_output: ground truth for the the data input
        recurrent: whether the data is in the recurrent format or not
    '''
    np_input = np.asarray(q_input)
    np_output = np.asarray(q_output)
    try:
        print "Evaluating..."
        assert len(q_input) != 0, "Query list must not be empty"
        scores = model.evaluate(np_input,
                                np_output,
                                batch_size=model_param['batch_size'],
                                verbose=model_param['verbose'])
        print "\nModel acurracy:", scores
        
        np_predictions = model.predict(np_input)
        
        if not recurrent:
            for j in range(len(np_output.T)):
                for i in range(len(np_input.T)):
                    plt.figure()
                    plt.plot(np_input.T[i].T, np_output.T[j], 'bo', label='Ground truth')
                    plt.plot(np_input.T[i].T, np_predictions.T[j], 'ro', label='Prediction')
                    plt.legend(loc='upper left')
                    t = "Input %i -> Output %i" % (i, j)
                    plt.title(t)
            plt.show()
        else:
            for j in range(len(np_output.T)):
                for i in range(len(np_input[0])):
                    plt.figure()
                    plt.plot(np_input[:, i], np_output.T[j], 'bo', label='Ground truth')
                    plt.plot(np_input[:, i], np_predictions.T[j], 'ro', label='Prediction')
                    plt.legend(loc='upper left')
                    t = "Input %i -> Output %i" % (i, j)
                    plt.title(t)
            plt.show()
        
        plot_prediction(model, len(np_input[0]))
        
    except ValueError, e:
        print 'Wrong keras model or model parameters'


def plot_prediction(model, sequence_length):
    '''
    Print the predicted values against the ground truth.
    
    Args:
        model: Model to evaluate
        q_input: data input
        q_output: ground truth for the the data input
        recurrent: whether the data is in the recurrent format or not
    '''
    import math
    import random
    
    nb_iter = 5 # Number of comparison to plot
    a_values = random.sample(np.arange(0, math.pi, 0.1), nb_iter) # Values for the a constant
    b_values = random.sample(np.arange(0, math.pi, 0.1), nb_iter) # Values for the b constant
    N = 50 # Number of values on the x axis

    for iteration in range(nb_iter):
        # Value of a and b for this run
        a = a_values[iteration]
        b = b_values[iteration]
        
        sin = lambda x: math.sin(a*x + b)
        
        x = [i for i in range(0, N)]
        y_true = [sin(i) for i in x]
        y_pred = y_true[:sequence_length]
        
        for i in range(sequence_length,N):
            #################################
            # Ligne à modifier pour la question 3.2):
            y_pred.append(model.predict(np.array(y_true[i-sequence_length:i]).reshape(1,sequence_length,1)))
            #################################
            
        plt.figure()
        plt.title("a={}, b={}".format(a, b))
        plt.plot(x, y_true, label="Ground Truth")
        plt.plot(x, y_pred, label="Predictions")
        plt.axvline(x=sequence_length, color="red")
        plt.legend()
    plt.show()
            
            
            
