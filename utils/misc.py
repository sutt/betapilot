import random, copy, time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

def contrivedLabel(img, ret='min_y'):
    ''' derive a int index of min/max non-zero point in image'''

    h, w = img.shape[0], img.shape[1]
    Y, X = np.ogrid[:h, :w]
    
    where_x = np.where(img>0, X, 999)
    where_y = np.where(img>0, Y, 999)
    min_x = where_x.min()
    min_y = where_y.min()
    
    where_x = np.where(img>0, X, -999)
    where_y = np.where(img>0,Y, -999)
    max_x = where_x.max()
    max_y = where_y.max()
    
    min_point, max_point = (min_x, min_y), (max_x, max_y)    
    
    if ret == 'min_y':
        return min_y
    if ret == 'max_y':
        return max_y
    if ret == 'min_x':
        return min_x
    if ret == 'max_x':
        return max_y

def plotLabel(img, y=None, x=None, point=None):
    plt.imshow(img, cmap='gray')
    if point is not None:
        plt.scatter(point[0], point[1])
    if y is not None:
        plt.scatter(1, y)
    if x is not None:
        plt.scatter(x, 1)
    plt.show()

def bagData(list_img, N=10):
    frac = int(len(list_img) / N)
    bag_data = [list_img[ (i*frac) : ((i+1)*frac)]
             for i in range(N)
            ]
    print(len(bag_data), len(bag_data[0]), len(bag_data[1]))
    return bag_data

def summaryStats(y, ret=False):
    mean = sum(y) / len(y)
    abs_errs = [abs(mean-_x) for _x in y]
    sq_errs = [(mean-_x)**2 for _x in y]
    mean_abs_err = sum(abs_errs) / len(abs_errs)
    mean_sq_err = sum(sq_errs) / len(sq_errs)
    print('mean y      : %s' % str(round(mean,2)))
    print('mean abs err: %s' % str(round(mean_abs_err,2)))
    print('mean sq err:  %s' % str(round(mean_sq_err,2)))
    if ret:
        return (mean, mean_abs_err)