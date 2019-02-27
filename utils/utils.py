import os, sys, copy, random, shutil, json
import cv2
import numpy as np
from matplotlib import pyplot as plt

# sample image dir -------------------------------

def uniqueDir(root_dir, fn_root):
    i = 1
    while True:
        files = os.listdir(root_dir)
        proposed_fn = str(fn_root) + str(i)
        if proposed_fn not in files:
            return proposed_fn
        i += 1

def buildNewDir(root_dir, new_dir):
    try:
        os.mkdir(os.path.join(root_dir, new_dir))
        return 0
    except Exception as e:
        print e
        return -1

def randomCopy(p_src, p_dest, N):

    dir_src = os.listdir(p_src)
    
    for _i in random.sample(range(len(dir_src)), N):

        fn = str(dir_src[_i])
        fn_src = os.path.join(p_src, fn)
        fn_dst = os.path.join(p_dest, fn)
        shutil.copy(fn_src, fn_dst)
    
    fns = os.listdir(p_dest)
    print len(fns)
    print str(fns[:min(len(fns), 3)]) + '...'


def miniTruth(p_dst, truth_input, truth_output):
    ''' build a truth.json for that mini dir '''

    d_truth = {}
    
    img_names = [str(fn) for fn in os.listdir(p_dst)]

    with open(truth_input, 'r') as f:
        d_load_truth = json.load(f)

    for _img in img_names:
        if _img in d_load_truth.keys():
            val = d_load_truth[_img]
            if val == [[]]:   # correct problem with formatting
                val = []
            d_truth[_img] = val

    with open(os.path.join(p_dst, truth_output), 'w') as f:
        json.dump(d_truth, f)
        
    if truth_output in os.listdir(p_dst):
        print 'output truth ', truth_output
    else:
        print 'failed to output truth to: ', str(p_dst)    


def makeMini( root_dir = '../mini/'
             ,fn_root = 'mini'
             ,N=50
             ,p_src = '../Data_Training/Data_Training/'
             ,b_truth = True
             ,truth_input = '../training_GT_labels_v2.json'
             ,truth_output = 'truth.json'
            ):
    ''' copy files from training to mini/miniX/ '''
    
    guid_dir = uniqueDir(root_dir, fn_root)
    
    ret = buildNewDir(root_dir, guid_dir)
    
    if ret != 0:
        print 'failed to build new dir; exiting'
        return None
    
    p_dst = os.path.join(root_dir, guid_dir)
    print p_dst
    
    randomCopy(p_src, p_dst, N)
    
    if b_truth:
        miniTruth(p_dst, truth_input, truth_output)

# plot helpers -------------------------------------------------------

def drawRect(img, rect, color='yellow', thick=3):
    COLOR = (0, 255, 255)
    if color == 'blue':
        COLOR = (255,0,0)
    return cv2.rectangle(img, rect[0], rect[1], COLOR, thick)
def drawPoly(img, poly, color='yellow', thick=3):
    COLOR = (0, 255, 255)
    if color == 'blue':
        COLOR = (255,0,0)
    return cv2.polylines(img, [poly], True, COLOR, thick)
def answerToRect(answer):
    a = answer[0]
    x = [v for i,v in enumerate(a) if i % 2 == 0]
    y = [v for i,v in enumerate(a) if i % 2 == 1]
    return ((min(x), min(y)), (max(x), max(y)))  
def answerToPoints(answer):
    a = answer[0]
    x = [v for i,v in enumerate(a) if i % 2 == 0]
    y = [v for i,v in enumerate(a) if i % 2 == 1]
    return np.array( [[e[0], e[1]] for e in zip(x,y)] )
def loadImg(fn, p_training):
    return cv2.imread(p_training + fn)
def plotImg(img, rect=None, poly=None):
    if rect is not None:
        img = drawRect(img.copy(), rect, thick=10)
    if poly is not None:
        img = drawPoly(img.copy(), poly, thick=10)
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def myPlot( ind
            ,truth
            ,poly=None
            ,p_training = '../Data_Training/Data_Training/'
            ):

    fn = os.listdir(p_training)[ind]
    img = loadImg(fn, p_training)
    answer = truth[fn]
    rect = answerToRect(answer)
    if poly is None:
        poly = answerToPoints(answer)    
    plotImg(img, poly=poly)
    plt.show()
    print fn
    print poly

# find average --------------------------------------------------------

def meanTruth(truth):
    ''' return list len-8 of mean of each coord'''

    all_points = []
    for _item in truth.items():
        _answer = _item[1]
        np_points = answerToPoints(_answer)
        points = np_points.tolist()
        all_points.append(points)

    has_points = filter(lambda point: point != [], all_points)

    means = [np.mean([_point[p][x] for _point in has_points]) 
                for p in range(4) for x in range(2)
            ]
    
    return means