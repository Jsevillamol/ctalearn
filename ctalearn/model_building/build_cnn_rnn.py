#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:15:47 2018

@author: jsevillamol
"""
import argparse
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense

def build_cnn_rnn(
        img_w=108, img_h=108, n_channels=1, n_classes=2,
        optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc']):

    input_shape = (None, img_w, img_h, n_channels)
    
    # Create input
    inputs = Input(input_shape)
    
    # mask the inputs that correspond to padding
    # model.add(Masking(mask_value=0., input_shape=input_shape))
    
    # Add CNN feature extractor
    x = TimeDistributed(Conv2D(
                    filters=16, 
                    kernel_size=(3,3), 
                    padding='same',
                    activation='relu'
                    ), name='conv2D_1')(inputs)
    
    x = TimeDistributed(Conv2D(
                    filters=32, 
                    kernel_size=(3,3), 
                    padding='same',
                    activation='relu'
                    ), name='conv2D_2')(x)
    
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None), name='maxpool_1')(x)
    
    x = TimeDistributed(Conv2D(
                    filters=64, 
                    kernel_size=(3,3), 
                    padding='same',
                    activation='relu'
                    ), name='conv2D_3')(inputs)
    
    x = TimeDistributed(Conv2D(
                    filters=128, 
                    kernel_size=(3,3), 
                    padding='same',
                    activation='relu'
                    ), name='conv2D_4')(x)
    
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None), name='maxpool_2')(x)
    
    x = TimeDistributed(Flatten(), name='flatten')(x)
    
    # Add LSTM feature combinator
    x = LSTM(units=10)(x)
    
    # Add FCN classifier
    x = Dense(units=100, activation='relu')(x)
    prediction = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=prediction)
    
    # Compile the model for training
    model.compile(optimizer, loss, metrics)
    
    model.summary()

    return model


if __name__=="__main__":
    # parser options
    parser = argparse.ArgumentParser(
            description=("Build a simple cnn-rnn keras model with ctalearn."))
    
    parser.add_argument(
            '--img_w',
            type=int,
            default=108,
            help="width in pixels of input")

    parser.add_argument(
            '--img_h',
            type=int,
            default=108,
            help="height in pixels of input")
    
    parser.add_argument(
            '--n_channels',
            type=int,
            default=1,
            help="number of channels of each image")
    
    parser.add_argument(
            '--n_classes',
            type=int,
            default=2,
            help="number of classes")
    
    args = parser.parse_args()
    
    # build model
    model = build_cnn_rnn(args.img_w, args.img_h, args.n_channels, args.n_classes)

    # save model architecture to disk in .h5 format
    model.save('cnn_rnn.h5')
    