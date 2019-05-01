#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:55:12 2018

@author: jsevillamol
"""

import logging, time
from datetime import timedelta

import math
import threading
import tables

import pandas as pd
import numpy as np

from tensorflow.keras.utils import Sequence, to_categorical

from ctalearn.data.image_mapping import ImageMapper
from ctalearn.data.data_processing import DataProcessor

class DataManager():
    """
    Manages access to the datafiles
    
    :method get_train_val_gen:
    :method get_pred_gen:
    """
    def __init__(self, file_list_fn, 
                 image_mapping_config = {},
                 preprocessing_config = {},
                 selected_tel_types=['LST'],
                 data_type='array',
                 img_size=(120,120), 
                 channels=['image_charge'],
                 min_triggers_per_event=1
                 ):
        """
        :param file_list_fn: path to .txt file which lists paths to all .h5 files containing data
        :param imageMapper_config = {}
        :param dataProcessor_config = {}
        :param img_shape:
        :param channels:
        :param selected_tel_types:
        """
        logging.info('initializing DataManager...')
        t_start = time.time()
        
        # Set class fields
        self._selected_tel_types = selected_tel_types
        self._data_type = data_type
        self._img_size = img_size
        self._channels = channels
        
        self._min_triggers_per_event = min_triggers_per_event
        self._keep_telescope_position = False #TODO choose better name
        
        self._classes = list(PARTICLE_ID_TO_CLASS_NAME.values())
        self._class_name_to_class_label = {'proton':0, 'gamma':1} #TODO fix hardcoding
        
        # Load files
        self._load_files_(file_list_fn)

        # Create a common index of events
        self._load_dataset_info_()
        
        # Compute dataset statistics
        self.dataset_metadata = self._compute_dataset_metadata()
        
        # Load telescope array info
        self.tel_array_metadata = self._compute_telescope_array_metadata()
        
        # Initialize ImageMapper and DataProcessor
        self._imageMapper = ImageMapper(**image_mapping_config)
        self._dataProcessor = DataProcessor(dataset_metadata = self.dataset_metadata, **preprocessing_config)
        
        # Logging end of initialization
        t_end = time.time()
        t_delta = timedelta(seconds=t_end - t_start)
        logging.info('DataManager initialized')
        logging.info(f"Time elapsed during DataManager initialization: {t_delta}")

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
        self.files = {filename: self.__synchronized_open_file(filename, mode='r')
                      for filename in data_files}
                    
    
    @staticmethod
    def __synchronized_open_file(*args, **kwargs):
        with threading.Lock() as lock:
            return tables.open_file(*args, **kwargs)

    @staticmethod
    def __synchronized_close_file(self, *args, **kwargs):
        with threading.Lock() as lock:
            return self.close(*args, **kwargs)

    def _load_dataset_info_(self):
        """
        Creates two pandas dataframes indexing all events from all files
        and the images from the selected telescope types that pass the specified filters
        
        :returns image_index:
        :returns event_index:
        """
        image_index = []
        event_index = []

        for fn in self.files:
            datafile = self.files[fn]
            
            # All events in the same file have the same particle type
            particle_type = datafile.root._v_attrs.particle_type
            class_name = PARTICLE_ID_TO_CLASS_NAME[particle_type]
            
            # Each row of each datafile is a different event
            for row in datafile.root.Event_Info.iterrows():
                # skip the row if the event associated does not pass the filter
                if not self._filter_event(fn, row): continue
            
                event_img_idxs = []
                for tel_type in self._selected_tel_types:
                    try:
                        img_rows_tel = row[tel_type + '_indices']
                    except KeyError:
                        logging.warning(f"No such telescope {tel_type} in file {fn}")
                        continue
                    
                    img_idxs = []
                    for img_row in img_rows_tel:                        
                        # If the image was not triggered or does not pass the filter
                        if img_row == 0 or not self._filter_img(fn, tel_type, img_row): 
                            if self._keep_telescope_position:
                                img_idxs.append(-1)
                            continue
                    
                        # Compute image statistics
                        record = datafile.root._f_get_child(tel_type)[img_row]
                        energy_trace = record['image_charge']
                        min_energy = np.min(energy_trace)
                        max_energy = np.max(energy_trace)
                        total_energy = np.sum(energy_trace)
                        
                        img_idxs.append(len(image_index))
                        
                        image_index.append({
                                'filename': fn,
                                'tel_type': tel_type,
                                'img_row': img_row,
                                'event_index': len(event_index),
                                'class_name': class_name,
                                'min_energy': min_energy,
                                'max_energy': max_energy,
                                'total_energy' : total_energy
                                })
                    
                    # Add global image indices to the event indices
                    event_img_idxs += img_idxs
                
                # If there is at least one non-dummy image associated to this event
                # add it to the event index
                if len([idx for idx in event_img_idxs if idx != -1]) >= self._min_triggers_per_event:
                    event_index.append({
                            'filename': fn, 
                            'image_indices': event_img_idxs, 
                            'class_name': class_name
                            })
            
        # Create pandas dataframes
        self._image_index_df = pd.DataFrame(image_index)
        self._event_index_df = pd.DataFrame(event_index)
    
    def _filter_event(self, fn, event_row):
        """
        Returns True if the event passes the specified filters
        """
        return True #TODO implement event filter
    
    def _filter_img(self, fn, tel_type, img_row):
        """
        Returns True if the image passes the specified filters
        """
        # if the img_row == 0 then the telescope did not trigger,
        # so there is no such image
        if img_row == 0: return False
        return True #TODO
        
    def _compute_dataset_metadata(self, selected_indices=None):
        """
        Computes some handy data of the chosen indices of the dataset
        
        :param event_indices: indices of the rows to be used to compute the metadata
                              if None, the whole dataset is used
                              This allows us to compute metadata for the train / val sets
        :return metadata: container object whose fields store information about the dataset
        """
        if selected_indices is not None and self._data_type == 'array':
            # Select chosen rows of event and image index
            event_index_df = self._event_index_df.iloc[selected_indices]
            image_indices = event_index_df.image_indices.sum()
            image_index_df = self._image_index_df.iloc[image_indices]
        elif selected_indices is not None and self._data_type == 'single_tel':
            image_index_df = self._image_index_df.iloc[selected_indices]
            event_index_df = self._event_index_df
        else:
            # Use the whole dataset
            event_index_df = self._event_index_df
            image_index_df = self._image_index_df
        
        # Create a empty object to store the data
        metadata = type('', (), {})()
        
        # Event metadata
        metadata.n_events_per_class = event_index_df.groupby(['class_name']).size()
        metadata.max_seq_length = event_index_df.image_indices.map(len).max()
        
        # Image metadata
        metadata.n_images_per_telescope = image_index_df.groupby(['tel_type']).size()
        metadata.n_images_per_class = image_index_df.groupby(['class_name']).size()
        
        metadata.image_charge_min = image_index_df.min_energy.min()
        metadata.image_charge_max = image_index_df.min_energy.max()
        
        # Class weights
        if self._data_type == 'array':
            count_by_class = metadata.n_events_per_class
        elif self._data_type == 'single_tel':
            count_by_class = metadata.n_images_per_class
        
        total = count_by_class.sum()
        metadata.class_weight = \
            {self._class_name_to_class_label[class_name]: count_by_class[class_name] / total
            for class_name in count_by_class.keys()}
        
        return metadata
        
    
    def _compute_telescope_array_metadata(self):
        # Create empty object to hold the relevant information
        tel_array_metadata = type('', (), {})()
        tel_array_metadata.n_telescopes_per_type = {}
        tel_array_metadata.max_telescope_position = [0,0,0]
        
        # all files contain the same array information
        f = next(iter(self.files.values()))
        
        for row in f.root.Array_Info.iterrows():
            tel_type = row['tel_type']
            if tel_type not in tel_array_metadata.n_telescopes_per_type:
                tel_array_metadata.n_telescopes_per_type[tel_type] = 0
            tel_array_metadata.n_telescopes_per_type[tel_type] += 1
            tel_pos = row['tel_x'], row['tel_y'], row['tel_z']
            for i in range(3):
                if tel_array_metadata.max_telescope_position[i] < tel_pos[i]:
                    tel_array_metadata.max_telescope_position[i] = tel_pos[i]
        
        return tel_array_metadata

    ###################################
    # EVENT AND IMAGE GETTERS
    ###################################
    
    def _get_batch(self, batch_idxs):
        """ 
        Returns the data and labels associated to an example
        
        :param example_id: unique identifier of an example
        :returns data: 
        :returns labels:
        """
        if self._data_type == 'array':
            # Get and stack img data, padding shorter sequences with 0 values
            events = [self._get_event_imgs(example_idx) for example_idx in batch_idxs]
            data = self._dataProcessor.preprocess_event_batch(events)
        elif self._data_type == 'single_tel':
            imgs = [self._get_image(img_id) for img_id in batch_idxs]
            data = np.stack(imgs)
            
        # Get and stack labels
        labels = [self._get_labels(example_idx) for example_idx in batch_idxs]
        labels = np.stack(labels)
        
        return data, labels
    
    def _get_event_imgs(self, event_id):
        """
        Returns the imgs associated to an event

        :param event_id: unique identifier of an event in the dataset
        
        :return imgs: numpy array of shape (n_triggers, width, heigth, n_channels)
        """
        # get indices of images associated to this event
        img_idxs = self._event_index_df.at[event_id, 'image_indices']
        
        # get images
        imgs = [self._get_image(img_id) for img_id in img_idxs]
        
        # Preprocess event
        imgs = self._dataProcessor.preprocess_event(imgs)
        
        return imgs
    
    def _get_image(self, img_id):
        """
        Loads and prepares an image for Keras consumption
        
        :param img_id: id of the image in the global image index
        """
        # If the image has index -1 it is replaced by dummy data
        if img_id == -1: 
            output_shape = self._img_size + (len(self._channels),)
            return np.zeros(output_shape)
        
        # load vector trace
        trace = self._load_trace(img_id)

        # tranform vector data to 2D img
        tel_type = self._image_index_df.at[img_id, 'tel_type']
        img = self._imageMapper.vector2image(trace, tel_type)
        
        # preprocess img
        img = self._dataProcessor.preprocess_img(img, self._img_size)
        
        return img
    
    def _load_trace(self, img_id):
        """
        Loads a vector trace from the .hdf5 datafiles
        
        :param fn: filename of datafile containing the trace of a img
        :param tel_type: type of telescope of image
        :param idx: row where the trace is stored
        
        :returns: np array of shape (n_pix, n_channels)
        """
        # retrieve index data
        fn       = self._image_index_df.at[img_id, 'filename']
        tel_type = self._image_index_df.at[img_id, 'tel_type']
        img_idx  = self._image_index_df.at[img_id, 'img_row' ]
        
        # retrieve image entry
        f = self.files[fn]
        record = f.root._f_get_child(tel_type)[img_idx]
        
        # Load channels
        shape = (record['image_charge'].shape[0] + 1, len(self._channels))
        trace = np.empty(shape, dtype=np.float32)

        for i, channel_name in enumerate(self._channels):
            trace[1:, i] = record[channel_name]

        return trace

    def _get_labels(self, obj_id):
        """
        :param obj_id: id identifying a unique entry in event_df or image_df
        :returns labels: list of labels per example
        """
        if self._data_type == 'array':
            class_name = self._event_index_df.at[obj_id,'class_name']
        elif self._data_type == 'single_tel':
            class_name = self._image_index_df.at[obj_id,'class_name']
        
        class_label = self._class_name_to_class_label[class_name]
        one_hot_label = to_categorical(class_label, len(self._classes))
        
        return one_hot_label
        

    ###################################
    # DATA SEQUENCE GETTERS
    ###################################

    def get_train_val_gen_(self, train_split=0.9, val_split=0.1, seed=None, batch_size=32):
        """
        Returns a train and a validation _DataGenerator object to be fed to 
        model.fit_generator()
        
        :param val_split: approximate fraction of the dataset to go in the validation set
        :param seed: seed to reproduce the train / val data split
        :param batch_size:
        :param shuffle:
            
        :return train_gen: _DataSequence object wrapping the training set
        :return val_gen: _DataSequence object wrapping the validation set
        """
        # Split the dataset into train and validation sets
        if self._data_type == 'array':
            n_examples = len(self._event_index_df)
        elif self._data_type == 'single_tel':
            n_examples = len(self._image_index_df)
        
        train_idxs, val_idxs = self._create_split_idxs(n_examples, train_split, val_split, seed)
        
        # Create DataGenerator objects wrapping the train and val sets
        train_gen = _DataGenerator(self, train_idxs, batch_size)
        val_gen   = _DataGenerator(self, val_idxs,   batch_size)
        
        # Create metadata objects
        self.train_metadata = self._compute_dataset_metadata(train_idxs)
        self.val_metadata = self._compute_dataset_metadata(val_idxs)
        
        return train_gen, val_gen
    
    @staticmethod
    def _create_split_idxs(index_size, train_split=0.9, val_split=0.1, seed=None):
        """
        Creates indices for the train and validation set
        :param data_index_df: 
        :param val_split: approximate fraction of the dataset to go to the val dataset
        :param seed: 
        :returns train_idxs:
        :returns val_idxs:
        """
        # Check that 0 < split < 1
        if train_split < 0 or 1 < train_split:
            raise ValueError("Invalid train split: {}. Must be between 0.0 and 1.0".format(val_split))
        if val_split < 0 or 1 < val_split:
            raise ValueError("Invalid validation split: {}. Must be between 0.0 and 1.0".format(val_split))
        if 1 < train_split + val_split:
            raise ValueError("Train + val split = {}. Must be between 0.0 and 1.0".format(train_split + val_split))
        
        # seed the random generator
        if seed != None:
           np.random.seed(seed)
        
        # create indices
        idxs = np.array(range(index_size))
        
        # shuffle the indices
        np.random.shuffle(idxs)
        
        # split the indices
        train_split_idx = round(train_split * index_size)
        val_split_idx = train_split_idx + round(val_split * index_size)
        train_idxs = idxs[:train_split_idx]
        val_idxs = idxs[train_split_idx:val_split_idx]

        return train_idxs, val_idxs
    
    def get_pred_gen(self, batch_size=32):
        """
        Returns a DataGenerator object of the whole dataset for prediction
        """
        # Create indices
        if self._data_type == 'array':
            idxs = np.array(range(len(self._data_index_df)))
        elif self._data_type == 'single_tel':
            idxs = np.array(range(len(self._image_index_df)))
        
        # Create _DataSequence
        pred_gen = _DataGenerator(self, idxs, batch_size)

        return pred_gen
    

class _DataGenerator(Sequence):
    """
    Sequence object to be fed to model.fit_generator()
    It wraps a list of indexes that reference examples.
    The actual example data is served by the DataManager.
    """
    def __init__(self, dataManager, idxs, batch_size=32):
        """
        :param dataManager: DataManager object, serves the examples
        :param idxs: List of the indexes of all the examples
        :batch_size: Indicates how many examples should be produced per batch
        """
        self.dataManager = dataManager
        self.idxs = idxs
        self.batch_size = batch_size
    
    def __len__(self):
        """ Returns the number of batches in this Sequence """
        return math.ceil(len(self.idxs) / self.batch_size)
    
    def __getitem__(self, idx):
        """
        Gets batch at position `idx`.
        :param idx: position of the batch in the Sequence.
        :return X: numpy array of data features. shape (batch_size, ?)
        :return y: numpy array of labels. shape (batch_size, ?)
        """
        
        # select batch of events
        batch_idxs = self.idxs[idx*self.batch_size : (idx+1)*self.batch_size]

        # load each event in the batch
        X, y = self.dataManager._get_batch(batch_idxs)
        
        return X, y
            
###############################
#       Telescope data
###############################
PARTICLE_ID_TO_CLASS_NAME = {0 : 'gamma', 101 : 'proton'}

##############################
#        QUICK TEST
##############################
    
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    # fn = '/home/jsevillamol/Documentos/datasample/sample_files.txt'
    fn = '/data2/deeplearning/ctlearn/tests/prototype_files_class_balanced.txt'
    dataManager = DataManager(fn, data_type='array')
    print(dataManager.dataset_metadata.__dict__)
    
    train_gen, val_gen = dataManager.get_train_val_gen_(seed=1111, batch_size=4)
    
    print(len(train_gen))
        
    X,y = train_gen[0]
    
    print(X.shape)
    print(y.shape)
    
    print(np.max(X))
    assert(np.any(X > 0.00001))
