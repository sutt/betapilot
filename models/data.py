import os, sys
import random, copy, time
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import pandas as pd


class DataXPrep:

    '''
        Apply pre-processing steps to X-data;
        Keep track of data slices

        TODO
        [ ] track test vs train
        [ ] allow an arg in methods to call different slices
            [ ] some centralized arg to do this
        [ ] store each processing step within the class
        [x] resize
        [ ] grayscale<->rgb
        [ ] get data shape method

    '''
    
    def __init__(self, data=None):
        self.data = data
        self.data_t = None
        self.data_params = None

    def buildDataT(  self, 
                    b_resize=False,
                    resize_width=None,
                    b_cvt_grayscale=False,
                    **kwargs
                    ):

        _data = self.data.copy()

        if b_resize:
            _data = self.resizeImgs_static(_data)
        if b_cvt_grayscale:
            _data = self.cvtGrayscale_static(_data)

        _data = self.normData_static(_data)

        self.data_t = _data.copy()

        data_params = {
            'b_resize': b_resize
            ,'resize_width': resize_width
            ,'b_grayscale': b_cvt_grayscale

        }
        self.data_params = data_params

    def getImgTShape(self):
        return self.data_t[0].shape

    def getImgShape(self):
        return self.data[0].shape

    def getDataT(self):
        return self.data_t.copy()
    
    def normData(self, **kwargs):
        return self.normData_static(self.data)

    @classmethod
    def normData_static(cls, img_list):
        ''' three steps:
            1. convert to numpy (if in list format)
            2. add 1-color-channel (if image in grayscale)
            3. convert type to float32
        '''
        data = img_list.copy()

        #1
        if isinstance(data, list):
            data = np.array(data)
        
        #2
        if len(data[0].shape) == 2:
            
            #only if it's grayscale
            len_data = data.shape[0]
            img_x, img_y = data[0].shape[0], data[0].shape[1]
        
            data = data.reshape(len_data, img_x, img_y, 1)
        
        #3
        data = data.astype('float32')

        cls.normData_validate(data)
        
        return data
    
    @classmethod
    def normData_validate(cls, img_list):
        try:
            assert isinstance(img_list, np.ndarray)
            assert len(img_list.shape) == 4
            assert img_list.shape[3] in [1,3] #color-channel is 1 or 3
        except Exception as e:
            print('failed to validate normed data in %s with err: %s'
                % (str(cls.__class__.__name__), str(e))
            )

    @staticmethod
    def resizeImgs_static(img_list, width=108):
        ''' list of imgs -> list of imgs
            imgs changes size: but preserves number of color channels
        '''
        def resize(img, width):
            return imutils.resize(img, width=width)

        resize_img_list = [resize(_img, width=width) 
                            for _img in img_list]
        
        return resize_img_list



class DataYPrep:

    ''' process Y-data into form accepted by keras models '''
    
    def __init__(self):
        self.data = None

    @staticmethod
    def normData_static(scalar_list):
        ''' make a list of scalars into a np array '''
        
        if isinstance(scalar_list, list):
            tmp_list = scalar_list.copy()
            np_scalar = np.array(tmp_list)
            return np_scalar

        elif isinstance(scalar_list, np.ndarray):
            return scalar_list
        
        else:
            err_type = str(type(scalar_list))
            raise('unrecognized type for scalar list: %s' % err_type)



