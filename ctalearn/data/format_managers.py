#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:25:41 2018

NOTE: This is an incomplete proof-of-concept indicating how I would go adapting
the DataManager to work with multiple file formats.

@author: jsevillamol
"""

from abc import ABC, abstractmethod

import threading, tables

class DataFormatManager(ABC):
    @abstractmethod
    def get_channel_trace(fn, event_id, tel_type, image_id, channel):
        pass
    
    @abstractmethod
    def get_class_name(self, fn):
        pass
    
    @abstractmethod
    def iter_rows(self, fn):
        pass
    
    
    
    @abstractmethod
    def __len__(self):
        pass

class HDF5Manager(DataFormatManager):
    
    def __init__(self, file_list_fn):
        # Set class fields
        self._PARTICLE_ID_TO_CLASS_NAME = {0 : 'gamma', 101 : 'proton'}
        self.classes = list(self.PARTICLE_ID_TO_CLASS_NAME.values())
        
        # Load data files
        self._load_files_(file_list_fn)
        
    
    def _load_files_(self, file_list_fn):
        """
        Loads the .hdf5 files containing the data
        :param file_list_fn: filename of the .txt file containing
                             the paths of the .h5 datafiles
        """
        # Load file list
        data_files = []
        with open(file_list_fn) as file_list:
            aux = [line.strip() for line in file_list]
            # Filter empty and commented lines
            data_files = [line for line in aux if line and line[0] != '#']
    
        # Load data files
        self.files = {}
        for filename in data_files:
            self.files[filename] = \
                    self.__synchronized_open_file(filename, mode='r')
    
    #####################################
    # Interface
    #####################################
    
    def get_channel_trace(self, fn, event_id, tel_type, image_id, channel):
        datafile = self.files[fn]
        record = datafile.root._f_get_child(tel_type)[image_id]
        trace = record[channel]
        return trace
    
    def get_class_name(self, fn):
        datafile = self.files[fn]
        particle_type = datafile.root._v_attrs.particle_type
        class_name = self._PARTICLE_ID_TO_CLASS_NAME[particle_type]
        return class_name
    
    def iter_rows(self, fn):
        """
        Yields a dict of image indices for each event in fn
        """
        datafile = self.files[fn]
        for row in datafile.root.Event_Info.iterrows():
            img_idxs_per_tel = None # TODO
            yield img_idxs_per_tel
    
    #####################################
    # Auxiliary functions
    #####################################
    
    @staticmethod
    def __synchronized_open_file(*args, **kwargs):
        with threading.Lock() as lock:
            return tables.open_file(*args, **kwargs)

    @staticmethod
    def __synchronized_close_file(self, *args, **kwargs):
        with threading.Lock() as lock:
            return self.close(*args, **kwargs)
    