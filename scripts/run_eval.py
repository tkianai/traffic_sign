"""This scripts handle the whole evaluation process
"""

import os
import argparse
import json
from scripts.eval_trafficSign import TrafficSignEval
from scripts.submision_style import StyleTrans


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--dt-file', type=str, help="detection coco style results")
    parser.add_argument('--gt-file', type=str, help="annotations providing id -> image name information")
    parser.add_argument('--pn-file', type=str, default=None, help="middle file")
    parser.add_argument('--score-thr', type=str, default=0.8, help="middle file")
    args = parser.parse_args()
    if not os.path.exists(args.gt_file):
        print("File Not Found Error: {}".format(args.gt_file))
        exit(404)
    if not os.path.exists(args.dt_file):
        print("File Not Found Error: {}".format(args.dt_file))
        exit(404)
    return args

def run(args):
    # choose the best score threshold
    anns = json.load(open(args.gt_file))
    eval_flag = False
    thr = args.score_thr
    if len(anns['annotations']) != 0:
        thr = eval_ts.results['Score']
        eval_flag = True
    
    if eval_flag:
        # evaluate the original detection performance
        eval_ts = TrafficSignEval(args.dt_file, args.gt_file)
        eval_ts.eval_map(mode='segm')
        eval_ts.eval_F_Score(threshold=0.9, mode='segm')
        eval_ts.save_pr_curve(save_name='work_dirs/conlusions/P_R_curve.png')

    # transfer mask results to polygons
    st = StyleTrans(args.dt_file, args.gt_file, args.pn_file)
    if args.pn_file is None:
        st.segm_to_polygons(save_name='work_dirs/conlusions/results_polygons.json')
        args.pn_file = 'work_dirs/conlusions/results_polygons.json'
    
    if eval_flag:
        # evaluate the detection performance after formatter
        eval_ts = TrafficSignEval(args.pn_file, args.gt_file)
        eval_ts.eval_map(mode='segm')
        eval_ts.eval_F_Score(threshold=0.9, mode='segm')
        eval_ts.save_pr_curve(save_name='work_dirs/conlusions/P_R_curve_formatter.png')

    # save the submission results
    st.segm_to_csv(score_thr=thr, save_name='work_dirs/conlusions/predict.csv')

if __name__ == "__main__":
    args = parse_args()
    run(args)
