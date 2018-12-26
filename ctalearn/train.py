#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:22:03 2018

@author: jsevillamol
"""

import os, logging, signal
import argparse, time
from datetime import timedelta
import yaml, csv
from contextlib import redirect_stdout
import itertools
from pandas.io.json.normalize import nested_to_record
import operator
from functools import reduce
from collections import MutableMapping
from contextlib import suppress

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.python.keras.optimizers import Adam

from ctalearn.build_model import build_model
from ctalearn.keras_utils import auroc
from ctalearn.data.data_loading import DataManager

def train(config, model_file=None, train_dir='.'):
        
    # Set up training options    
    train_config = config['train_config']
    
    train_split = train_config.get('train_split')
    val_split = train_config.get('val_split')
    seed = train_config.get('seed')
    batch_size = train_config.get('batch_size')
    shuffle = train_config.get('shuffle')
    epochs = train_config.get('epochs')
    
    optimizer = train_config.get('optimizer')  
    learning_rate = train_config.get('learning_rate', None)
    decay = train_config.get('decay', None)
    eps = train_config.get('epsilon', None)
    
    loss = train_config.get('loss')
    metrics_names = train_config.get('metrics')
    
    stop_early = train_config.get('stop_early')
    min_delta = train_config.get('min_delta', None)
    patience = train_config.get('patience', None)
    
    class_weight = train_config.get('class_weight')
    save_model = train_config.get('save_model')
    
    data_config = config['data_config']
    model_config = config['model_config']
    
    # Get time stamp of start of training 
    # to generate unique names for the outputs of the run
    time_stamp = time.strftime('%Y%m%d_%H%M%S')
        
    # Load model
    if model_file == None:
        logging.info('Building model from configuration file.')
        model = build_model(**model_config)
    else:
        logging.info(f'Loading model from file {model_file}')
        model = load_model(model_file)
    
    # Show model summary through console and then save it to file
    model.summary()

    with open(f'{train_dir}/model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # Prepare metrics
    metric_dict = {'acc':'acc', 'auc':auroc}
    metrics = [metric_dict[metric] for metric in metrics_names]
    
    # Prepare optimizer
    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=eps, decay=decay, amsgrad=False)
    
    # Compile model
    model.compile(optimizer, loss, metrics)
    logging.info("Model compiled")
    
    # Callbacks
    callbacks = []
    
    # Monitors the SIGINT (ctrl + C) to safely stop training when it is sent
    flag = False
    class SafeStop(Callback):
        """Callback that terminates training when the flag is raised.
        """
        def on_epoch_end(self, batch, logs=None):
            print("safestop callback activated")
            if flag:    
                self.model.stop_training = True
                
    def handler(signum, frame):
        logging.info('SIGINT signal received. Training will finish after this epoch')
        global flag
        flag = True
    
    signal.signal(signal.SIGINT, handler) # We assign a specific handler for the SIGINT signal
    safeStop = SafeStop()
    callbacks.append(safeStop)
    
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
    
    if stop_early is not None:
        # Monitors rate of improvement on a chosen metric and stops training if it enters a plateau
        early_cb = EarlyStopping(
                monitor= stop_early, 
                min_delta= min_delta, 
                patience= patience, 
                verbose=0, 
                mode='auto')
        callbacks.append(early_cb)
    
    # Logs the metrics after each epoch to a csv file
    csv_logger = CSVLogger(f'{train_dir}/training_history.csv')
    callbacks.append(csv_logger)

    # Create data manager
    dataManager = DataManager(**data_config)
    
    # Log data manager metadata
    logging.info(f"dataset_metadata = {dataManager.dataset_metadata.__dict__}")
    logging.info(f"tel_array_metadata = {dataManager.tel_array_metadata.__dict__}")

    # Get train and validation generators
    train_generator, val_generator = \
        dataManager.get_train_val_gen_(train_split, val_split, seed, batch_size, shuffle)
    
    logging.info(f"train_metadata = {dataManager.train_metadata.__dict__}")
    logging.info(f"val_metadata = {dataManager.val_metadata.__dict__}")
    
    # Set up class weights
    if class_weight: class_weight = dataManager.train_metadata.class_weight
    else: class_weight = None
    
    # Train the model
    logging.info("Starting training")
    t_start = time.time()
    history = model.fit_generator(
                train_generator, 
                steps_per_epoch=None, # One epoch = whole dataset by default
                epochs=epochs,
                verbose=1, # Show progress bar per epoch
                callbacks=callbacks, 
                validation_data=val_generator, 
                validation_steps=None, # All data in val_generator is used for validation
                class_weight=class_weight, 
                max_queue_size=10, 
                workers=1, 
                use_multiprocessing=False, 
                shuffle=shuffle, 
                initial_epoch=0
                )
    t_end = time.time()
    logging.info("Training finished!")
    
    # Log training duration
    t_delta = timedelta(seconds=t_end - t_start)
    logging.info(f"Time elapsed during training: {t_delta}")

    # Save architecture, weights, training configuration and optimizer state
    if save_model:
        model_fn = f'{train_dir}/model_{time_stamp}.h5'
        model.save(model_fn)
    
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
    
    # After every training session we need to clear the tf session
    # to avoid OOM errors when running multiple training sessions
    # one after another
    tf.keras.backend.clear_session()
  
############################
# GRID SEARCH FUNCTIONALITY 
############################

def multi_train(config, model_file=None, start_from_run=0):
    # Process 'multi_' options
    run_number = None
    for run_number, config in enumerate(make_combinations(config)):
        # Skip the runs already done in a previous session
        if run_number < start_from_run: continue
    
        # Create a directory where to store the training results
        run_number_name = str(run_number).zfill(3)
        train_dir = f'run{run_number_name}'
        try:
            os.mkdir(train_dir)
        except OSError:
            pass
        
        # Save a copy of the configuration combination in the target training folder
        with open(f'{train_dir}/run{run_number_name}_config.yaml', 'w') as f:
            yaml.dump(config, f)
            
        # Launch the training session
        logging.info(f'Starting run number {run_number_name}')
        train(config, model_file, train_dir)
    
    # If no `multi_` options were found
    if run_number is None: return 0
    
    # Read the results from the runs
    total_runs = run_number + 1
    histories = []
    for run_number in range(total_runs):
        run_number_name = str(run_number).zfill(3)
        train_dir = f'run{run_number_name}'
        csv_fn = f'{train_dir}/training_history.csv'
        with open(csv_fn) as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_contents = []
            for row in csv_reader: 
                csv_contents.append(row)
            metric_names = csv_contents[0]
            history = np.array(csv_contents[1:])
            histories.append(history) 
    
    # Plot the runs
    for i, metric in enumerate(metric_names):
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        for history in histories: 
            hist = list(map(float, history[:, i]))
            plt.plot(hist)
        plt.savefig(f'history_{metric}.png')
        plt.clf()
    
    # Save the final results to a CSV
    csv_fn = 'multi_summary.csv'
    with open(csv_fn, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['run_number'] + metric_names)
        for run_number, history in enumerate(histories):
            csv_writer.writerow([run_number] + list(history[-1, :]))
    
    # return the number of training runs launched
    return total_runs
    
def make_combinations(config):
    """
    Generate all possible configurations that a config file specifies via 'multi_' parameters
    If there are no 'multi_' parameters this generator is empty
    """
    flat = nested_to_record(config)
    flat = { tuple(key.split('.')): value for key, value in flat.items()}
    multi_config_flat = {key[:-1] + (key[-1][6:],) : value 
                         for key, value in flat.items() 
                         if key[-1].startswith('multi')}
    if len(multi_config_flat) == 0: return # if there are no multi params this generator is empty
    keys, values = zip(*multi_config_flat.items())
    
    # delete the multi_params
    # taken from https://stackoverflow.com/a/49723101/4841832
    def delete_keys_from_dict(dictionary, keys):
        """
        Delete fields in a nested dict
        """
        for key in keys:
            with suppress(KeyError):
                del dictionary[key]
        for value in dictionary.values():
            if isinstance(value, MutableMapping):
                delete_keys_from_dict(value, keys)
                
    to_delete = ['multi_' + key[-1] for key in multi_config_flat]
    delete_keys_from_dict(config, to_delete)
    
    for values in itertools.product(*values):
        experiment = dict(zip(keys, values))
        for setting, value in experiment.items():
            pointer_to_inner_dict = reduce(operator.getitem, setting[:-1], config)
            pointer_to_inner_dict[setting[-1]] = value
        yield config

########################
# LAUNCH SCRIPT
########################

if __name__ == "__main__":
    
    # Parser options
    parser = argparse.ArgumentParser(
            description=("Train a keras model with ctalearn."))
    parser.add_argument(
            'config_file',
            help="path to YAML file containing a training and data configuration")
    parser.add_argument(
            'model_file',
            nargs='?',
            default=None,
            help="path to H5 file containing a keras model. If not specified, the model will be built from the configuration")
    parser.add_argument('-s', '--start_from_run', default=0, type=int,
                        help="What run number to start from in multi mode. Useful when a multi conf run is interrupted and you need to resume it.")
    args = parser.parse_args()

    # Set up logging
    log_file = f"session.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)
    consoleHandler = logging.StreamHandler(os.sys.stdout)
    logging.getLogger().addHandler(consoleHandler)
    logging.info(f"Starting training session")
    
    # load config file
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
    
    # Try running the multi configuration train sequence
    run_number = multi_train(config, args.model_file, args.start_from_run)
        
    if run_number == 0:
        # There are no multiple configurations to try, just call the training procedure
        train(config, args.model_file)
        