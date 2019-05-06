"""This scripts aims to eval the results from detection model
: > 1. calculate F score / score threshold under threshold 0.9
: > 2. choose the best F score to submit
: > 3. eval mAP
:
: > Author [tkianai] 
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class TrafficSignEval(object):
    def __init__(self, dt_file, gt_file=None, iou_threshold=0.9):
        self.gt_file = gt_file
        self.dt_file = dt_file
        self.iou_threshold = iou_threshold
        
        self.precision = None
        self.recall = None
        self.results = None

    def calculate_F_score(self, Precision, Recall):
        eps = 1e-7
        F_score = 2.0 * Precision * Recall / (Precision + Recall + eps)
        return F_score

    def save_pr_curve(self, save_name='./results/P_R_curve.png'):
        if self.precision is None or self.recall is None:
            return

        # save the P-R curve
        save_dir = os.path.dirname(save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.clf()
        plt.plot(self.recall, self.precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Precision-Recall Curve')
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(save_name, dpi=400)
        print('Precision-recall curve has been written to {}'.format(save_name))

    def eval_map(self, mode='segm'):
        """evaluate mean average precision
        
        Keyword Arguments:
            mode {str} -- could be choose from ['bbox' | 'segm'] (default: {'segm'})
        """
        if mode not in ['bbox', 'segm']:
            raise NotImplementedError(
                "Mode [{}] doesn't been implemented, choose from [bbox, segm]!".format(mode))

        # eval map
        Gt = COCO(self.gt_file)
        Dt = Gt.loadRes(self.dt_file)

        evalObj = COCOeval(Gt, Dt, mode)
        imgIds = sorted(Gt.getImgIds())
        evalObj.params.imgIds = imgIds
        evalObj.evaluate()
        evalObj.accumulate()
        evalObj.summarize()

    def eval_F_Score(self, threshold=None, mode='segm'):
        iou_threshold = self.iou_threshold if threshold is None else threshold
        assert iou_threshold >= 0.0 and iou_threshold <= 1.0, "The IOU threshold [{}] is illegal!".format(
            iou_threshold)
        if mode not in ['bbox', 'segm']:
            raise NotImplementedError(
                "Mode [{}] doesn't been implemented, choose from [bbox, segm]!".format(mode))

        # eval map
        Gt = COCO(self.gt_file)
        Dt = Gt.loadRes(self.dt_file)

        evalObj = COCOeval(Gt, Dt, mode)
        imgIds = sorted(Gt.getImgIds())
        evalObj.params.imgIds = imgIds
        evalObj.params.iouThrs = [iou_threshold]
        evalObj.params.areaRng = [[0, 10000000000.0]]
        evalObj.params.maxDets = [100]

        evalObj.evaluate()
        evalObj.accumulate()

        Precision = evalObj.eval['precision'][0, :, 0, 0, 0]
        Recall = evalObj.params.recThrs
        Scores = evalObj.eval['scores'][0, :, 0, 0, 0]

        F_score = self.calculate_F_score(Precision, Recall)

        # calculate highest F score
        idx = np.argmax(F_score)
        results = dict(
            F_score=F_score[idx],
            Precision=Precision[idx],
            Recall=Recall[idx],
            Score=Scores[idx],
        )

        self.precision = Precision
        self.recall = Recall
        self.results = results

        # summarize
        print('---------------------- F1 ---------------------- ')
        print('Maximum F-score: %f' % results['F_score'])
        print('  |-- Precision: %f' % results['Precision'])
        print('  |-- Recall   : %f' % results['Recall'])
        print('  |-- Score    : %f' % results['score'])
        print('------------------------------------------------ ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation on TrafficSign')
    parser.add_argument('--gt-file', default='data/gt.json', type=str, help='annotation | groundtruth file.')
    parser.add_argument('--dt-file', default='data/dt.json', type=str, help='detection results of coco annotation style.')
    args = parser.parse_args()

    if not os.path.exists(args.gt_file):
        print("File Not Found Error: {}".format(args.gt_file))
        exit(404)
    if not os.path.exists(args.dt_file):
        print("File Not Found Error: {}".format(args.dt_file))
        exit(404)
    
    eval_ts = TrafficSignEval(args.dt_file, args.gt_file)
    eval_ts.eval_map(mode='bbox')
    eval_ts.eval_map(mode='segm')
    eval_ts.eval_F_Score(threshold=0.9, mode='segm')
    eval_ts.save_pr_curve(save_name='work_dirs/conlusions/P_R_curve.png')
