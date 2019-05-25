#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import yaml, argparse
from contextlib import redirect_stdout

from tensorflow.keras.models import Model
import tensorflow.keras.layers as ll
from tensorflow.keras.regularizers import l2

def build_model(
        input_shape, 
        num_classes,
        activation_function, 
        dropout_rate,
        use_batchnorm,
        l2_regularization,
        cnn_layers,
        lstm_units,
        combine_mode,
        fcn_layers):
    ''' Builds a CNN-RNN-FCN classification model
    
    # Parameters
        input_shape (tuple) -- expected input shape
        num_classes (int) -- number of classes
        activation_function (str) -- non linearity to apply between layers
        dropout_rate (float) -- must be between 0 and 1
        use_batchnorm (bool) -- if True, batchnorm layers are added between convolutions
        l2_regularization (float)
        cnn_layers (list) -- list specifying CNN layers. 
                             Each element must be of the form 
                             {filters: 32,  kernel_size: 3, use_maxpool: true}
        lstm_units (int) -- number of hidden units of the lstm
                            if lstm_units is None or 0 the LSTM layer is skipped
        combine_mode (str) -- specifies how the encoding of each image in the sequence 
                              is to be combined. Supports:
                                  concat : outputs are stacked on top of one another
                                  last : only last hidden state is returned
                                  attention : an attention mechanism is used to combine the hidden states
                              
        fcn_layers (list) -- list specifying Dense layers
                             example element: {units: 1024}
    # Returns
        model -- an uncompiled Keras model
    '''
    # Regularizer
    l2_reg = l2(l2_regularization)
    
    # Build a model with the functional API
    inputs = ll.Input(input_shape)
    x = inputs
    
    # Reshape entry if needed
    if len(input_shape) == 3:
        x = ll.Reshape([1] + input_shape)(x)
    elif len(input_shape) < 3:
        raise ValueError(f"Input shape {input_shape} not supported")

    # CNN feature extractor    
    for i, cnn_layer in enumerate(cnn_layers):
        # Extract layer params
        filters = cnn_layer['filters']
        kernel_size = cnn_layer['kernel_size']
        use_maxpool = cnn_layer['use_maxpool']

        # build cnn_layer
        x = ll.TimeDistributed(ll.Conv2D(
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
            x = ll.TimeDistributed(ll.MaxPooling2D(
                    pool_size=(2, 2), 
                    strides=None, 
                    padding='valid', 
                    data_format=None
                ), name=f'maxpool_{i}')(x)
        
        if use_batchnorm:
            x = ll.TimeDistributed(ll.BatchNormalization(
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

    
    x = ll.TimeDistributed(ll.Flatten(), name='flatten')(x)
    x = ll.TimeDistributed(ll.Dropout(dropout_rate), name='dropout')(x)

    # LSTM feature combinator
    if lstm_units is not None and lstm_units > 0:
        x = ll.CuDNNLSTM(
                lstm_units,
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
                return_sequences=(combine_mode!='last'),
                return_state=False,
                go_backwards=False,
                stateful=False
                )(x)
    
    # Combine output of each sequence
    if combine_mode == 'concat':
        x = ll.Flatten()(x)
    elif combine_mode == 'last':
        if lstm_units is None or lstm_units == 0:    # if no LSTM was used
            x = ll.Lambda(lambda x : x[:,-1,...])(x) # we extract the last element
    elif combine_mode == 'attention':
        attention = ll.TimeDistributed(ll.Dense(1), name='attention_score')(x)
        attention = ll.Flatten()(attention)
        attention = ll.Softmax()(attention)
        x = ll.dot([x, attention], axes=[-2, -1])
    else: raise ValueError(f"Combine mode {combine_mode} not supported")
    
    # FCN classifier    
    for fcn_layer in fcn_layers:
        # extract layer params
        units = fcn_layer['units']
        
        # build layer
        x = ll.Dense(
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
        
        x = ll.Dropout(dropout_rate)(x)

    
    prediction = ll.Dense(num_classes, activation='softmax')(x)
    
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
