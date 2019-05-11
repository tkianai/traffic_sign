"""This helper script aims to merge detected results from six parts.
"""

import os
import json
import argparse

BBOX_OFFSET = {
    # None stands for scale 2
    # (width offset, height offset)
    "1": None,
    "2": (0, 0),
    "3": (1600, 0),
    "4": (0, 900),
    "5": (1600, 900),
    "6": (800, 450),
    "7": (800, 0),
    "8": (800, 900),
    "9": (0, 450),
    "10": (1600, 450),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Merge bounding boxes")
    parser.add_argument('--dt-file', help="Detected results")
    parser.add_argument('--gt-file', help="Groundtruth annotation file")
    parser.add_argument('--save', default='./work_dirs/conclusion/predict_up.json', help="The merged bounding box json file")
    parser.add_argument('--gt-file-up', help="The groundtruth annotation in the upper level image")
    # parser.add_argument('--csv', default='./work_dirs/conclusion/predict.csv', help="The submission file")

    return parser.parse_args()


def _merge_bboxes(id2name, name2id_up, ann_list):
    _out = {}
    if len(ann_list) == 0:
        return _out
    
    filename = id2name[ann_list[0]['image_id']]
    filename_up = filename.split('_')[0] + '.jpg'
    _out['image_id'] = name2id_up[filename_up]
    _out['segmentation'] = {}
    
    # TODO:
    width = 1600
    height = 900
    closest_id = -1
    offset_id = None
    closest_distance = -1
    for i, ann in enumerate(ann_list):
        filename = id2name[ann['image_id']]
        filename_idx = filename.split('_')[-1].split('.')[0]
        if filename_idx == "1":
            continue
        bbox_center_x = ann['bbox'][0] + ann['bbox'][2] // 2
        bbox_center_y = ann['bbox'][1] + ann['bbox'][3] // 2
        distance_ = pow(bbox_center_x - width / 2, 2) + pow(bbox_center_y - height / 2, 2)
        if distance_ < closest_distance or i == 0:
            closest_id  = i
            closest_distance = distance_
            offset_id = filename_idx

    ann = ann_list[closest_id]
    offset = BBOX_OFFSET[offset_id]
    _out['bbox'] = [ann['bbox'][0] + offset[0], ann['bbox'][1] + offset[1], ann['bbox'][2], ann['bbox'][3]]
    _out['score'] = ann['score']
    _out['category_id'] = ann['category_id']

    return _out


def _merge(dt_file, gt_file, gt_file_up, save_name='merged.json'):
    anns_gt = json.load(open(gt_file))
    anns_gt_up = json.load(open(gt_file_up))
    result_dt = json.load(open(dt_file))
    id2name = {itm['id']:itm['file_name'] for itm in anns_gt['images']}
    name2id_up = {itm['file_name']:itm['id'] for itm in anns_gt_up['images']}
    dt_infos = {}
    for itm in result_dt:
        filename = id2name[itm['image_id']]
        filename_up = filename.split('_')[0] + '.jpg'
        if filename_up not in dt_infos:
            dt_infos[filename_up] = []
        dt_infos[filename_up].append(itm)

    # process
    merged_dt = []
    for key, itms in dt_infos.items():
        itms = _merge_bboxes(id2name, id2name_up, itms)
        if len(itms) != 0:
            merged_dt.append(itms)

    with open(save_name, 'w') as w_obj:
        json.dump(merged_dt, w_obj)

def run_merge(args):
    # generate merged bbboxes 
    _merge(args.dt_file, args.gt_file, args.gt_file_up, args.save)

if __name__ == "__main__":
    args = parse_args()
    run_merge(args)
