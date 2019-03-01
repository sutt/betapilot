import json
from scorer import *
import argparse

# usage: python score_detections.py -g {ground truth json} -p {prediction json}

def score(truth_fn=None
         ,predict_fn=None
         ,d_truth=None
         ,d_predict=None
         ):
    
    mAP_scorer = mAPScorer()

    if d_truth is not None:
        GT_data = d_truth
    else:
        with open(truth_fn,'r') as f:
            GT_data = json.load(f)

    if d_predict is not None:
        pred_data = d_predict
    else:
        with open(predict_fn,'r') as f:
            pred_data = json.load(f)
    
    # print len(GT_data), len(pred_data)

    n_GT = mAP_scorer.countBoxes(GT_data)
    n_Pred = mAP_scorer.countBoxes(pred_data)
    
    coco_score = mAP_scorer.COCO_mAP(GT_data,pred_data)

    # print("COCO mAP for detector is {}".format(coco_score))

    return coco_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--groundtruthfile", type=str,
                        help="name of groundtruth file")
    parser.add_argument("-p","--predictionfile", type=str,
                        help="name of prediction file")

    args = parser.parse_args()
    mAP_scorer = mAPScorer()

    with open(args.groundtruthfile,'r') as f:
        GT_data = json.load(f)
    with open(args.predictionfile,'r') as f:
        pred_data = json.load(f)
        
    n_GT = mAP_scorer.countBoxes(GT_data)
    n_Pred = mAP_scorer.countBoxes(pred_data)
    print len(GT_data.keys())
    print len(GT_data.keys())
    coco_score = mAP_scorer.COCO_mAP(GT_data,pred_data)

    print("COCO mAP for detector is {}".format(coco_score))


