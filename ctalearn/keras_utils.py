#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:26:11 2018

@author: jsevillamol
"""

import json

import functools
import tensorflow as tf

# metrics

def as_keras_metric(method):
    
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        tf.keras.backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

auroc = as_keras_metric(tf.metrics.auc)

# callbacks
def save_training_results_cb():
    pass

