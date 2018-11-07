#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import logging
import argparse, time
from datetime import timedelta
import yaml, json

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tensorflow.python.keras.models import load_model

from ctalearn.data.data_loading import DataManager

if __name__ == "__main__":
    
    # Parser options
    parser = argparse.ArgumentParser(
            description=("Train a keras model with ctalearn."))
    parser.add_argument(
            'model_file',
            help="path to .h5 file containing a Keras architecture")
    parser.add_argument(
            'config_file',
            help="path to YAML file containing a training configuration")

    args = parser.parse_args()
    
    # Set up logging
    log_file = "session.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)
    logging.info("Logging has been correctly set up")
    
    # load config file
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
    
    # Set up training options    
    train_config = config['train_config']
    
    val_split = train_config.get('val_split', 0.2)
    seed = train_config.get('seed', None)
    batch_size = train_config.get('batch_size', 32)
    shuffle = train_config.get('shuffle', True)
    epochs = train_config.get('epochs', 1)
    
    data_config = config['data_config']
    
    # Load model
    model = load_model(args.model_file) #TODO find most recent model
    
    # Log model summary
    logging.info('Model summary:')
    model.summary(print_fn=logging.info)
    
    # TODO Load previous history
    initial_epoch = 0
    
    # TODO deduce img_size from model.inputs maybe?

    # Create data manager
    dataManager = DataManager(**data_config)
    
    # Log data manager metadata
    logging.info(dataManager.dataset_metadata.__dict__)
    logging.info(dataManager.tel_array_metadata.__dict__)

    # Get train and validation generators
    train_generator, val_generator = \
        dataManager.get_train_val_gen(val_split, seed, batch_size, shuffle)

    # Train the model
    logging.info("Starting training")
    t_start = time.time()
    history = model.fit_generator(
                train_generator, 
                steps_per_epoch=None, # One epoch = whole dataset by default
                epochs=epochs,
                verbose=1, 
                callbacks=None, 
                validation_data=val_generator, 
                validation_steps=None, 
                class_weight=None, 
                max_queue_size=10, 
                workers=1, 
                use_multiprocessing=False, 
                shuffle=True, 
                initial_epoch=initial_epoch
                )
    t_end = time.time()
    logging.info("Training finished!")
    
    # Log training duration
    t_delta = timedelta(seconds=t_end - t_start)
    logging.info(f"Time elapsed during training: {t_delta}")
    
    # Get time stamp of end of training 
    # to generate unique names for the outputs of the run
    time_stamp = time.strftime('%Y%m%d_%H%M%S')

    # Save architecture, weights, training configuration and optimizer state
    model_fn = 'model_' + time_stamp + '.h5'
    model.save(model_fn)
    
    # TODO Update prev history
    
    # Plot training history
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_history.png')
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_history.png')
    
    # Save history
    history_fn = 'history_' + time_stamp + '.json'
    json.dump(history.history, open(history_fn, mode='w'))


