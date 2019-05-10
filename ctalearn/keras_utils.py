#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:26:11 2018

@author: jsevillamol
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import metrics
import logging
import signal

# callbacks
def get_callbacks(train_config, train_dir):
    
    batch_size = train_config.get('batch_size')
    stop_early = train_config.get('stop_early')
    min_delta = train_config.get('min_delta', None)
    patience = train_config.get('patience', None)
    
    callbacks = []
    
    # Monitors the SIGINT (ctrl + C) to safely stop training when it is sent
    
    class SafeStop(tf.keras.callbacks.Callback):
        """Callback that terminates training when the flag is raised.
        """
        def __init__(self): 
            self.safestop_flag = False
        def on_epoch_end(self, batch, logs=None):
            if self.safestop_flag:    
                self.model.stop_training = True
                self.safestop_flag = False
    safeStop = SafeStop()
                
    def handler(signum, frame):
        logging.info('SIGINT signal received. Training will finish after this epoch')
        safeStop.safestop_flag = True
    
    signal.signal(signal.SIGINT, handler) # We assign a specific handler for the SIGINT signal
    
    callbacks.append(safeStop)
    
    # Creates event files for tensorboard during training
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=f'{train_dir}/logs', 
            histogram_freq=0, 
            batch_size=batch_size, 
            write_graph=True, 
            write_grads=True, 
            write_images=False
            )
    callbacks.append(tensorboard_cb)
    
    if False:
        # Creates checkpoints of the best models found
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=train_dir + '/model.{epoch:02d}-{val_loss:.2f}.h5', 
                monitor='val_loss', 
                verbose=0, 
                save_best_only=True, 
                save_weights_only=False, 
                mode='auto', period=1)
        callbacks.append(checkpoint_cb)
    
    if stop_early is not None:
        # Monitors rate of improvement on a chosen metric and stops training if it enters a plateau
        early_cb = tf.keras.callbacks.EarlyStopping(
                monitor= stop_early, 
                min_delta= min_delta, 
                patience= patience, 
                verbose=0, 
                mode='auto')
        callbacks.append(early_cb)
    
    # Logs the metrics after each epoch to a csv file
    csv_logger = tf.keras.callbacks.CSVLogger(f'{train_dir}/training_history.csv')
    callbacks.append(csv_logger)
    
    return callbacks

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


# taken from https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = tf.cast(tf.greater(tf.clip(y_pred, 0, 1), threshold_value), tf.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = tf.round(tf.sum(tf.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = tf.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + tf.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = tf.cast(tf.greater(tf.clip(y_pred, 0, 1), threshold_value), tf.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = tf.round(tf.sum(tf.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = tf.sum(tf.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + tf.epsilon())
        return recall_ratio
    return recall

# QUICK TESTS
if __name__ == "__main__":
    precision = precision_threshold()
    recall = recall_threshold()

    y_true = np.zeros(shape=(4,2))
    y_true[:, 0] = [1,0,1,0]
    y_true[:, 1] = [0,1,0,1]
    
    y_pred1 = np.zeros(shape=(4,2))
    y_pred1[:, 0] = [1,0,1,0]
    y_pred1[:, 1] = [0,1,0,1]
    
    print(tf.eval(metrics.categorical_accuracy(y_true, y_pred1)))
    print(tf.eval(auroc(y_true, y_pred1)))
    print(tf.eval(precision(y_true, y_pred1)))
    print(tf.eval(recall(y_true, y_pred1)))
    
    y_pred2 = np.zeros(shape=(4,2))
    y_pred2[:, 0] = [0,1,0,1]
    y_pred2[:, 1] = [1,0,1,0]
    
    print(tf.eval(metrics.categorical_accuracy(y_true, y_pred2)))
    print(tf.eval(auroc(y_true, y_pred2)))
    print(tf.eval(precision(y_true, y_pred2)))
    print(tf.eval(recall(y_true, y_pred2)))
    
