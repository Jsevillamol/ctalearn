#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import os, logging
import argparse, time
from datetime import timedelta
import yaml, json
from contextlib import redirect_stdout
import itertools

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from ctalearn.build_model import build_model
from ctalearn.keras_utils import auroc
from ctalearn.data.data_loading import DataManager

def train(config, train_dir='.'):
        
    # Set up training options    
    train_config = config['train_config']
    
    val_split = train_config.get('val_split')
    seed = train_config.get('seed')
    batch_size = train_config.get('batch_size')
    shuffle = train_config.get('shuffle')
    epochs = train_config.get('epochs')
    
    optimizer = train_config.get('optimizer')
    loss = train_config.get('loss')
    metrics_names = train_config.get('metrics')
    
    stop_early = train_config.get('stop_early')
    
    data_config = config['data_config']
    model_config = config['model_config']
    
    # Get time stamp of start of training 
    # to generate unique names for the outputs of the run
    time_stamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Set up logging
    log_file = f"{train_dir}/session_{time_stamp}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)
    consoleHandler = logging.StreamHandler(os.sys.stdout)
    logging.getLogger().addHandler(consoleHandler)
    logging.info(f"Starting training session {time_stamp}")
        
    # Load model
    logging.info('Building model from configuration file.')
    model = build_model(**model_config)
    logging.info('New model built')
    
    # Show model summary through console and then save it to file
    model.summary()

    with open(f'{train_dir}/model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # Prepare metrics
    metric_dict = {'acc':'acc', 'auc':auroc}
    metrics = [metric_dict[metric] for metric in metrics_names]
    
    # Compile model
    model.compile(optimizer, loss, metrics)
    logging.info("Model compiled")
    
    # TODO Load previous history
    initial_epoch = 0
    
    # Callbacks
    callbacks = []
    
    # Creates event files for tensorboard during training
    tensorboard_cb = TensorBoard(
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
        checkpoint_cb = ModelCheckpoint(
                filepath=train_dir + '/model.{epoch:02d}-{val_loss:.2f}.h5', 
                monitor='val_loss', 
                verbose=0, 
                save_best_only=True, 
                save_weights_only=False, 
                mode='auto', period=1)
        callbacks.append(checkpoint_cb)
    
    if stop_early:
        # Monitors rate of decrease and stops training if it enters a plateau
        early_cb = EarlyStopping(
                monitor='val_loss', 
                min_delta=0, 
                patience=1, 
                verbose=0, 
                mode='auto')
        callbacks.append(early_cb)
    
    # Logs the metrics after each epoch to a csv file
    csv_logger = CSVLogger(f'{train_dir}/training_{time_stamp}.csv')
    callbacks.append(csv_logger)

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
                callbacks=callbacks, 
                validation_data=val_generator, 
                validation_steps=None, 
                class_weight=None, 
                max_queue_size=10, 
                workers=1, 
                use_multiprocessing=False, 
                shuffle=shuffle, 
                initial_epoch=initial_epoch
                )
    t_end = time.time()
    logging.info("Training finished!")
    
    # Log training duration
    t_delta = timedelta(seconds=t_end - t_start)
    logging.info(f"Time elapsed during training: {t_delta}")

    # Save architecture, weights, training configuration and optimizer state
    model_fn = f'{train_dir}/model_{time_stamp}.h5'
    model.save(model_fn)
    
    # TODO Update prev history
    
    # Plot training history
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{train_dir}/history_loss.png')
    plt.clf()
    
    # summarize history for other metrics
    for metric in metrics_names:
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{train_dir}/history_{metric}.png')
        plt.clf()
    
    # Save history
    history_fn = f'{train_dir}/history_{time_stamp}.json'
    json.dump(history.history, open(history_fn, mode='w'))
    
def make_combinations(config):
    multi_config = config.pop('multi_config')
    
    # Flatten multi_config
    multi_config_flat = {(section, key): value 
                         for section in multi_config.keys() 
                         for key,value in multi_config[section].items()}
    
    keys, values = zip(*multi_config_flat.items())
    
    for values in itertools.product(*values):
        
        experiment = dict(zip(keys, values))
        
        for (section, key), value in experiment.items():
            config[section][key] = value
            
        yield config

if __name__ == "__main__":
    
    # Parser options
    parser = argparse.ArgumentParser(
            description=("Train a keras model with ctalearn."))
    parser.add_argument(
            'config_file',
            help="path to YAML file containing a training and data configuration")
    
    args = parser.parse_args()
    
    # load config file
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
    
    # Process multioptions
    multi_config = config.get('multi_config', None)
    
    if multi_config == None:
        # There are no multiple configurations to try
        train(config)
    else:
        # If there are multiple configuration values to try
        for run_number, config in enumerate(make_combinations(config)):
            run_number = str(run_number).zfill(2)
            train_dir = f'run{run_number}'
            os.mkdir(train_dir)
            with open(f'{train_dir}/run{run_number}_config.yaml', 'w') as f:
                yaml.dump(config, f)
            train(config, train_dir)
    
    

    
