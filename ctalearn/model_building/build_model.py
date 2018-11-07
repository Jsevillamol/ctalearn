#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import yaml, argparse

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import TimeDistributed, LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization, Dropout

def build_model(config):
    """
    Builds a CNN-RNN-FCN model according to the specs in model_config.
    """
    # Extract config options
    input_shape = config.get('input_shape', (None, 120, 120, 1))
    num_classes = config.get('num_classes', 2)
    
    activation_function = config.get('activation_function', 'relu')
    
    use_dropout = config.get('use_dropout', False)
    dropout_rate = config.get('dropout_rate', 0.8)

    use_batchnorm = config.get('use_batchnorm', False)
    
    cnn_layers = config.get('cnn_layers', 
                [
                    {'filters':32, 'kernel_size':3, 'use_maxpool':False},
                    {'filters':64, 'kernel_size':3, 'use_maxpool':True}
                ])    
    
    lstm_units = config.get('lstm_units', 100)
    
    fcn_layers = config.get('fcn_layers', [
                    {'units': 100},
                    {'units': 100}
                ])
    
    optimizer = config.get('optimizer', 'rsmprop')
    loss = config.get('loss', 'categorical_crossentropy')
    metrics = config.get('metrics', ['acc'])

    # Build a model with the functional API
    inputs = Input(input_shape)
    x = inputs

    # CNN feature extractor    
    for cnn_layer in cnn_layers:
        # Extract layer params
        filters = cnn_layer['filters']
        kernel_size = cnn_layer['kernel_size']
        use_maxpool = cnn_layer['use_maxpool']

        # build cnn_layer
        x = TimeDistributed(Conv2D(
                filters, 
                kernel_size, 
                strides=(1, 1), 
                padding='same', 
                data_format=None, 
                dilation_rate=(1, 1), 
                activation=activation_function, 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                bias_initializer='zeros', 
                kernel_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                bias_constraint=None
            ))(x)
        
        if use_batchnorm:
            x = TimeDistributed(BatchNormalization(
                    axis=-1, 
                    momentum=0.99, 
                    epsilon=0.001, 
                    center=True, 
                    scale=True, 
                    beta_initializer='zeros', 
                    gamma_initializer='ones', 
                    moving_mean_initializer='zeros', 
                    moving_variance_initializer='ones', 
                    beta_regularizer=None, 
                    gamma_regularizer=None, 
                    beta_constraint=None, 
                    gamma_constraint=None
                ))(x)

        
        # add maxpool if needed
        if use_maxpool:
            x = TimeDistributed(MaxPooling2D(
                    pool_size=(2, 2), 
                    strides=None, 
                    padding='valid', 
                    data_format=None
                ))(x)
    
    x = TimeDistributed(Flatten())(x)

    # LSTM feature combinator
    x = LSTM(
            lstm_units, 
            activation='tanh', 
            recurrent_activation='hard_sigmoid', 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', 
            bias_initializer='zeros', 
            unit_forget_bias=True, 
            kernel_regularizer=None, 
            recurrent_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            recurrent_constraint=None, 
            bias_constraint=None, 
            dropout=0.0, 
            recurrent_dropout=0.0, 
            implementation=1, 
            return_sequences=False, 
            return_state=False, 
            go_backwards=False, 
            stateful=False, 
            unroll=False
        )(x)

    if use_dropout:
        x = Dropout(dropout_rate)(x)
    
    # FCN classifier    
    for fcn_layer in fcn_layers:
        # extract layer params
        units = fcn_layer['units']
        
        # build layer
        x = Dense(
                units, 
                activation=activation_function, 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                bias_initializer='zeros', 
                kernel_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                bias_constraint=None
            )(x)

    
    prediction = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=prediction)
    
    # Compile the model for training
    model.compile(optimizer, loss, metrics)
    
    model.summary()
    
    return model

if __name__=="__main__":
    # parser options
    parser = argparse.ArgumentParser(
            description=("Build a customized cnn-rnn keras model with ctalearn."))
    
    parser.add_argument(
            'config_file',
            help="path to YAML file containing a training configuration")

    args = parser.parse_args()
    
    # load config file
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
        
    model = build_model(config)
    
    # save model architecture to disk in .h5 format
    model.save('model.h5')