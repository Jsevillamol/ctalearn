#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import yaml, argparse
from contextlib import redirect_stdout

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.python.keras.layers import TimeDistributed, LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2

def build_model(
        input_shape, 
        num_classes,
        activation_function, 
        dropout_rate,
        use_batchnorm,
        l2_regularization,
        cnn_layers,
        lstm_units,
        concat_lstm_output,
        fcn_layers):
    ''' Builds a CNN-RNN-FCN classification model
    
    # Parameters
        input_shape (tuple) -- expected input shape
        num_classes (int) -- number of classes
        activation_function (str) -- non linearity to apply between layers
        dropout_rate (float) -- must be between 0 and 1
        l2_regularization (float)
        cnn_layers (list) -- list specifying CNN layers. 
                             Each element must be of the form 
                             {filters: 32,  kernel_size: 3, use_maxpool: true}
        lstm_units (int) -- number of hidden units of the lstm
        concat_lstm_output (bool) -- if True, the outputs of all LSTM cells 
                                     are concatenated and fed to the next layer
                                     if False, only the last output is fed to the next layer
        fcn_layers (list) -- list specifying Dense layers
                             example element: {units: 1024}
    # Returns
        model -- an uncompiled Keras model
    '''
    # Regularizer
    l2_reg = l2(l2_regularization)
    
    # Build a model with the functional API
    inputs = Input(input_shape)
    x = inputs
    
    # Reshape entry if needed
    if len(input_shape) == 3:
        x = Reshape([1] + input_shape)(x)
    elif len(input_shape) < 3:
        raise ValueError(f"Input shape {input_shape} not supported")

    # CNN feature extractor    
    for i, cnn_layer in enumerate(cnn_layers):
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
                kernel_regularizer=l2_reg, 
                bias_regularizer=l2_reg, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                bias_constraint=None
            ), name=f'conv2D_{i}')(x)
        
        # add maxpool if needed
        if use_maxpool:
            x = TimeDistributed(MaxPooling2D(
                    pool_size=(2, 2), 
                    strides=None, 
                    padding='valid', 
                    data_format=None
                ), name=f'maxpool_{i}')(x)
        
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
                ), name=f'batchnorm_{i}')(x)

    
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = TimeDistributed(Dropout(dropout_rate), name='dropout')(x)

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
            kernel_regularizer=l2_reg, 
            recurrent_regularizer=l2_reg, 
            bias_regularizer=l2_reg, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            recurrent_constraint=None, 
            bias_constraint=None, 
            dropout=dropout_rate, 
            recurrent_dropout=0.0, 
            implementation=1, 
            return_sequences=concat_lstm_output, 
            return_state=False, 
            go_backwards=False, 
            stateful=False, 
            unroll=False
        )(x)
    
    if concat_lstm_output:
        x = Flatten()(x)
    
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
                kernel_regularizer=l2_reg, 
                bias_regularizer=l2_reg, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                bias_constraint=None
            )(x)
        
        x = Dropout(dropout_rate)(x)

    
    prediction = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=prediction)
    
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
        
    model = build_model(**config['model_config'])
    
    # Show model summary through console and then save it to file
    model.summary()

    with open('model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # save model architecture to disk in .h5 format
    model.save('untrained_model.h5', include_optimizer=False)
