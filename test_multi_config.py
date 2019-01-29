#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:25:18 2018

@author: jsevillamol
"""
from pandas.io.json.normalize import nested_to_record 
import itertools
import operator
from functools import reduce

from collections import MutableMapping
from contextlib import suppress

config = {
          'train_config': {'param1': 1, 'param2': [1,2,3], 'multi_param3':[2,3,4]}, 
          'model_config': {'cnn_layers': [{'units':3},{'units':4}], 'multi_param4': [[1,2], [3,4]]}
         }

def generate_multi_conf_combinations(config):
    """
    Generate all possible configurations that a config file specifies via 'multi_' parameters
    If there are no 'multi_' parameters this generator is empty
    """
    flat = nested_to_record(config)
    flat = { tuple(key.split('.')): value for key, value in flat.items()}
    multi_config_flat = { key[:-1] + (key[-1][6:],) : value for key, value in flat.items() if key[-1][:5]=='multi'}
    if len(multi_config_flat) == 0: return # if there are no multi params this generator is empty
    keys, values = zip(*multi_config_flat.items())
    
    # delete the multi_params
    # taken from https://stackoverflow.com/a/49723101/4841832
    def delete_keys_from_dict(dictionary, keys):
        for key in keys:
            with suppress(KeyError):
                del dictionary[key]
        for value in dictionary.values():
            if isinstance(value, MutableMapping):
                delete_keys_from_dict(value, keys)
    to_delete = ['multi_' + key[-1] for key, _ in multi_config_flat.items()]
    delete_keys_from_dict(config, to_delete)
    
    for values in itertools.product(*values):
        experiment = dict(zip(keys, values))
        for setting, value in experiment.items():
            reduce(operator.getitem, setting[:-1], config)[setting[-1]] = value
        yield config

c = None        
for c in generate_multi_conf(config):
    print(c)
if c is None: print("no multi config!")