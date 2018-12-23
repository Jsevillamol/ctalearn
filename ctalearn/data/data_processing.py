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
                 normalization=None,
                 resize_mode='interpolate',
                 event_order_type=None,
                 event_order_reverse=True,
                 min_imgs_per_seq=1,
                 max_imgs_per_seq=16,
                 sequence_padding='post',
                 sequence_truncating='post',
                 dataset_metadata = None
                 ):
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
        self._event_order_type = event_order_type
        self._event_order_reverse = event_order_reverse
        
        # Set batch preprocessing options
        self._min_imgs_per_seq = min_imgs_per_seq
        self._max_imgs_per_seq = max_imgs_per_seq
        self._sequence_padding = sequence_padding
        self._sequence_truncating = sequence_truncating
        
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
        order_keys = {
                'size': lambda img: np.sum(img) # Order events by brightest to dimmest
                }
        
        if self._event_order_type != None:
            event.sort(key=order_keys[self._event_order_type], reverse=self._event_order_reverse)
        
        # Stack events
        event = np.stack(event)

        return event
    
    #############################################
    # Batch preprocessing
    #############################################
    
    def preprocess_event_batch(self, batch):
        """
        :param batch: list with len batch_size of event sequences, each with shape (seq, img_w, img_h, channels)
        :returns batch: np.array of shape (batch_size, max_seq, img_w, img_h, channels)
        """
        # pad sequences
        max_len = max([len(seq) for seq in batch])
        target_len = np.clip(max_len, self._min_imgs_per_seq, self._max_imgs_per_seq)
        batch = pad_sequences(batch, target_len, 
                              padding=self._sequence_padding,
                              truncating=self._sequence_truncating)
        
        return batch

###########################
#        QUICK TEST
###########################
if __name__ == '__main__':
    dataProcessor = DataProcessor()
    img = np.ones(shape=(100,100,1))
    img = dataProcessor.preprocess_img(img, (120,120))
    
    assert(img.shape == (120,120,1))
    
    key = lambda img: np.sum(img)
    print(key(img))
    
    dataProcessor = DataProcessor(event_order_type='size')
    zero_img = np.zeros(shape=(100,100,1))
    one_img = np.ones(shape=(100,100,1))
    evnt = [zero_img, one_img]
    evnt = dataProcessor.preprocess_event(evnt)
    
    assert(np.all(evnt[0] == one_img))
    assert(np.all(evnt[1] == zero_img))