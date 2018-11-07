#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:49:03 2018

@author: jsevillamol
"""

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from skimage.transform import resize

class DataProcessor():
    """
    Preprocesses images and events
    """
    def __init__(self, 
                 dataset_metadata = None,
                 normalization='log_normalization',
                 resize_mode='interpolate',
                 event_order=None,
                 max_imgs_per_seq=10,
                 sequence_padding='post'):
        """
        :param config: 
            Python dict with the following entries
        :entry use_cropping:
        :entry use_log_normalization:
        """
        # Set image statistics
        if dataset_metadata != None:
            self._image_charge_min = dataset_metadata.image_charge_min
            self._image_charge_max = dataset_metadata.image_charge_max
        
        # Set img preprocessing options
        self._normalization = normalization
        self._resize_mode = resize_mode # TODO implement other resize modes
        
        # Set event preprocessing options
        self._event_order = event_order
        self._max_imgs_per_seq = max_imgs_per_seq
        
        # Set batch preprocessing options
        self._sequence_padding = sequence_padding
        
    ###########################################
    # Image preprocessing
    ###########################################

    def preprocess_img(self, img, target_size=(120,120)):
        """
        Takes as input a 2D image, and applies the selected preprocessing options
        It also pads or crops the image as needed so it fits the target shape
        
        :param img: numpy array of shape (height, width, n_channels)
        :param target_size: 
            
        :param :
        """
        
        # normalization
        normalization = {
            None: lambda x: x,
            'log_normalization': self._log_normalization
        }
        img = normalization[self._normalization](img)
        
        # cropping / padding
        img = DataProcessor._resize(img, target_size)
        
        return img

    def _log_normalization(self, image):
        image[:,:,0] = np.log(image[:,:,0] - self._image_charge_min + 1.0)
        return image
    
    @staticmethod
    def _resize(img, target_size):
        # using skimage
        img = resize(img, target_size)
        return img
    
    
        
    ############################################
    # Event preprocessing
    ############################################

    def preprocess_event(self, event):
        """
        Preprocesses event data
        :param event: list of already preprocessed images
        :returns event: np.array of preprocessed images
                        should have shape (seq_len, width, height, channels)
        """
        # TODO reorder event images
        
        # Only pick first n events
        event = event[:self._max_imgs_per_seq]
        
        # Stack events
        event = np.stack(event)

        return event
    
    #############################################
    # Batch preprocessing
    #############################################
    
    def preprocess_batch(self, batch):
        """
        :param batch: list of event sequences, each with shape (seq, img_w, img_h, channels)
        :returns batch: np.array of shape (batch_size, max_seq, img_w, img_h, channels)
        """
        # pad sequences
        batch = pad_sequences(batch, padding=self._sequence_padding)
        
        return batch

###########################
#        QUICK TEST
###########################
if __name__ == '__main__':
    dataProcessor = DataProcessor()
    img = np.ones(shape=(100,100,1))
    img = dataProcessor.preprocess_img(img, (120,120))
    
    assert(img.shape == (120,120,1))