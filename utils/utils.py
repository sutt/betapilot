import os, sys, copy, random, shutil, json

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
