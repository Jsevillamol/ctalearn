#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:10:45 2018

@author: jsevillamol
"""

import tensorflow as tf
import yaml, json

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
    parser.add_argument(
            '--debug',
            action='store_true',
            help="print debug/logger messages")
    parser.add_argument(
            '--log_to_file',
            action='store_true',
            help="log to a file in model directory instead of terminal")

    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_file, 'r') as config_file:
        config = yaml.load(config_file)
    
    # Load model
    with open(args.model_file, 'r') as model_file:
        yaml_model = yaml.load(model_file)
        model = model_from_yaml(yaml_string)

    # Load data table
    dataManager = DataManager(config['data_config'])

    # Train the model
    predict(model, dataManager, config['pred_config'])


