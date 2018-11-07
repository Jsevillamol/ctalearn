#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:19:47 2018

@author: jsevillamol
"""

import logging
import os
import numpy as np
from scipy.sparse import csr_matrix


###################################
# ImageMapper Class
###################################

class ImageMapper():
    
    # Telescopes supported by this version of the ImageMapper
    supported_telescopes = \
        [
         'LST', 'MSTF', 'MSTN', 'MSTS', 'SST1', 'SSTC', 'SSTA', 
         'VTS', 'MGC', 'FACT', 'HESS-I', 'HESS-II'
        ]
    
    def __init__(self, hex_conversion_algorithm='oversampling'):
        # Set class fields
        self.hex_conversion_algorithm = hex_conversion_algorithm
        
        # Initialize telescope parameters
        self._initialize_telescope_parameters_()
        
        # Initialize mapping tables
        self._initialize_mapping_tables_()
    
    ########################################
    # HARD CODED TELESCOPE PARAMETERS
    ########################################
    def _initialize_telescope_parameters_(self):
        """
        Decorator adding the static class parameters
        that correspond to fixed telescope parameters
        """
    
        self._pixel_lengths = {
            'LST'    : 0.05            ,
            'MSTF'   : 0.05            ,
            'MSTN'   : 0.05            ,
            'MSTS'   : 0.00669         ,
            'SST1'   : 0.0236          ,
            'SSTC'   : 0.0064          ,
            'SSTA'   : 0.0071638       ,
            'VTS'    : 1.0 * np.sqrt(2),
            'MGC'    : 1.0 * np.sqrt(2),
            'FACT'   : 0.0095          ,
            'HESS-I' : 0.0514          ,
            'HESS-II': 0.0514
        }
    
        self._num_pixels = {
            'LST'    : 1855 ,
            'MSTF'   : 1764 ,
            'MSTN'   : 1855 ,
            'MSTS'   : 11328,
            'SST1'   : 1296 ,
            'SSTC'   : 2048 ,
            'SSTA'   : 2368 ,
            'VTS'    : 499  ,
            'MGC'    : 1039 ,
            'FACT'   : 1440 ,
            'HESS-I' : 960  ,
            'HESS-II': 2048 ,
        }
    
        self._image_shapes = {
            'LST'    : (108, 108),
            'MSTF'   : (110, 110),
            'MSTN'   : (108, 108),
            'MSTS'   : (120, 120),
            'SST1'   : (94, 94)  ,
            'SSTC'   : (48, 48)  ,
            'SSTA'   : (56, 56)  ,
            'VTS'    : (54, 54)  ,
            'MGC'    : (82, 82)  ,
            'FACT'   : (90, 90)  ,
            'HESS-I' : (72, 72)  ,
            'HESS-II': (104, 104),
        }
        
        self._pixel_positions = \
            {
                tel_type: ImageMapper._read_pix_pos_files(tel_type) 
                for tel_type in ImageMapper.supported_telescopes 
                if (tel_type != 'VTS' and tel_type != 'MGC')
            }
    
    @staticmethod
    def _read_pix_pos_files(tel_type):
        """
        Auxiliary function that loads the pix_pos_files associated to a telescope
        :param tel_type:
        :return pix_pos_table:
        """
        if tel_type in ImageMapper.supported_telescopes: 
            relative_fn = "pixel_pos_files/{}_pos.npy".format(tel_type)
            #TODO fix this hardcoding
            infile = os.path.join(os.path.dirname(__file__), relative_fn) 
            return np.load(infile)
        else:
            logging.error(f"Telescope type {tel_type} isn't supported.")
            return False
        
    ###########################
    # Mapping table generation
    ###########################

    def _initialize_mapping_tables_(self):
        """
        Initializes the mapping tables
        """

        # Prepare mapping tables
        self._mapping_tables = {
            'LST':  self._generate_table_generic('LST'),
            'MSTF': self._generate_table_generic('MSTF'),
            'MSTN': self._generate_table_generic('MSTN'),
            #'MSTS': self.generate_table_MSTS(),
            'SST1': self._generate_table_generic('SST1'),
            #'SSTC': self.generate_table_SSTC(),
            #'SSTA': self.generate_table_SSTA(),
            #'VTS': self.generate_table_VTS(),
            #'MGC': self.generate_table_MGC(),
            'FACT': self._generate_table_generic('FACT'),
            #'HESS-I': self.generate_table_HESS('HESS-I'),
            #'HESS-II': self.generate_table_HESS('HESS-II')
        }
    
    def _generate_table_generic(self, tel_type, pixel_weight=1.0/4):
        if self.hex_conversion_algorithm == 'oversampling':
            mapping_matrix = self._oversampling_generate_table_generic(tel_type, pixel_weight)

        else:
            raise NotImplementedError("Cannot convert hexagonal camera image without valid conversion algorithm.")

        return mapping_matrix
    
    def _oversampling_generate_table_generic(self, tel_type, pixel_weight=1.0/4):
    
        # Note that this only works for Hex cams
        # Get telescope pixel positions for the given tel type
        pos = self._pixel_positions[tel_type]

        # Get relevant parameters
        output_dim = self._image_shapes[tel_type][0]
        num_pixels = self._num_pixels[tel_type]
        pixel_length = self._pixel_lengths[tel_type]

        # For LST and MSTN cameras, rotate by a fixed amount to
        # align for oversampling
        if tel_type in ["LST", "MSTN"]:
            pos = ImageMapper._rotate_cam(pos)

        # Compute mapping matrix
        pos_int = pos / pixel_length * 2
        pos_int[0, :] = pos_int[0, :] / np.sqrt(3) * 2
        # below put the image in the corner
        pos_int[0, :] -= np.min(pos_int[0, :])
        pos_int[1, :] -= np.min(pos_int[1, :])
        p0_lim = np.max(pos_int[0, :]) - np.min(pos_int[0, :])
        p1_lim = np.max(pos_int[1, :]) - np.min(pos_int[1, :])
        if output_dim < p0_lim or output_dim < p1_lim:
            logging.warning("Danger! output image shape too small, will be cropped!")
        # below put the image in the center
        if tel_type in ["FACT"]:
            pos_int[0, :] += (output_dim - p0_lim) / 2. - 1.0
            pos_int[1, :] += (output_dim - p1_lim - 0.8) / 2. - 1.0
        else:
            pos_int[0, :] += (output_dim - p0_lim) / 2.
            pos_int[1, :] += (output_dim - p1_lim - 0.8) / 2.


        mapping_matrix = np.zeros((num_pixels + 1, output_dim, output_dim), dtype=float)
        
        for i in range(num_pixels):
            x, y = pos_int[:, i]
            x_S = int(round(x))
            x_L = x_S + 1
            y_S = int(round(y))
            y_L = y_S + 1
            # leave 0 for padding, mapping matrix from 1 to 499
            mapping_matrix[i + 1, y_S:y_L + 1, x_S:x_L + 1] = pixel_weight

        # make sparse matrix of shape (num_pixels + 1, output_dim * output_dim)
        mapping_matrix = csr_matrix(mapping_matrix.reshape(num_pixels + 1, output_dim * output_dim))
        
        return mapping_matrix
    
    @staticmethod
    def _rotate_cam(pos):
        rotation_matrix = np.matrix([[0.98198181,  0.18897548],
                                     [-0.18897548, 0.98198181]], dtype=float)
        pos_rotated = np.squeeze(np.asarray(np.dot(rotation_matrix, pos)))
    
        return pos_rotated
    
    #####################################
    # Translation function
    #####################################
    
    def vector2image(self, pixels, telescope_type):
        """
        :param pixels: a numpy array of values for each pixel, in order of pixel index.
                       The array should have dimensions [N_pixels, N_channels] where N_channels is e.g. 
                       1 when just using charges and 2 when using charges and peak arrival times. 
        :param telescope_type: a string specifying the telescope type as defined in the HDF5 format, 
                            e.g., 'MSTS' for SCT data, which is the only currently implemented telescope type.
        :return: a numpy array of shape [img_width, img_length, N_channels]
        """
        # Check that telescope is of supported type
        if telescope_type in self._image_shapes.keys():
            telescope_type = telescope_type
        else:
            raise ValueError('Sorry! Telescope type {} isn\'t supported.'.format(telescope_type))
    
        # Process each channel of result
        n_channels = pixels.shape[1]
        result = []
        for channel in range(n_channels):
            vector = pixels[:,channel]
            
            if telescope_type == "MSTS":
                image_2D = vector[self._mapping_tables[telescope_type]].T[:,:,np.newaxis]
            elif telescope_type in ['LST', 'MSTF', 'MSTN', 'SST1', 'SSTC', 'SSTA', 'VTS', 'MGC', 'FACT','HESS-I','HESS-II']:
                aux = vector.T @ self._mapping_tables[telescope_type]
                image_2D = aux.reshape(
                        self._image_shapes[telescope_type][0], 
                        self._image_shapes[telescope_type][1], 
                        1)

            result.append(image_2D)

        telescope_image = np.concatenate(result, axis = -1)
    
        return telescope_image
    
    ###################################################################
    # Pixel position auxiliary functions
    ###################################################################
    
    # internal methods to create pixel pos numpy files 
    @staticmethod
    def _get_pos_from_h5(tel_table, tel_type="MSTF", write=False, outfile=None):
        filtered_table = tel_table.where('tel_type=={}'.format(tel_type))
        selected_tel_rows = np.array([row.nrow for row in filtered_table])[0]
        pixel_pos = tel_table.cols.pixel_pos[selected_tel_rows]
        if write:
            if outfile is None:
                rel_outfile = "pixel_pos_files/{}_pos.npy".format(tel_type)
                outfile = os.path.join(os.path.dirname(__file__), rel_outfile)
            np.save(outfile, pixel_pos)
        return pixel_pos

if __name__=='__main__':
    imageMapper = ImageMapper()