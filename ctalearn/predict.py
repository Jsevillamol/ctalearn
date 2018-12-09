#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:10:45 2018

@author: jsevillamol
"""

import yaml, argparse
from tensorflow.python.keras.model import load_model

from ctalearn.data_loading import DataManager

if __name__ == "__main__":
    # Parser options
    parser = argparse.ArgumentParser(
            description=("Train a keras model with ctalearn."))
    parser.add_argument(
            'config_file',
            help="path to YAML configuration file with training options")
    parser.add_argument(
            'model_file',
            help="path to YAML file containing a Keras architecture")

    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
        
    predict_config = config_file['predict']
    batch_size = predict_config['batch_size']
    
    # Load model
    with open(args.model_file, 'r') as model_file:
        model = load_model(args.model_file)

    # Load data table
    dataManager = DataManager(config['data_config'])
    predict_generator = dataManager.get_pred_gen(batch_size)
    
    # predict
    predictions = model.predict_generator(
            predict_generator, 
            steps=None, 
            max_queue_size=10, 
            workers=1, 
            use_multiprocessing=False, 
            verbose=0
        )
    
    # TODO store predictions
    


