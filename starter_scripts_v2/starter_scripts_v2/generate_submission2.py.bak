# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results import *
import time

training_path = 'Data_Training/Data_Training/'

img_file = glob.glob(training_path + '*.JPG')
# img_keys = [img_i.split('/')[-1]  for img_i in img_file]
# maybe this works in py3 / not on windows
img_keys = [img_i.split('\\')[-1]  for img_i in img_file]

# Instantiate a new detector
finalDetector = GenerateFinalDetections()
# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for img_key in img_keys:
    # img =cv2.imread('testing/images/'+img_key)
    # img =cv2.imread('testing/'+img_key)
    # img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # tic = time.monotonic()
    tic = time.time()

    bb_all = finalDetector.predict(None)
    # toc = time.monotonic()
    toc = time.time()
    pred_dict[img_key] = bb_all
    time_all.append(toc-tic)

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('random_submission3.json', 'w') as f:
    json.dump(pred_dict, f)
