"""This script aims to transfer coco style detection results to csv submision style
: > 1. Change segmentation format only, keep the whole detection style.
: > 2. save to csv results
: > Author [tkianai]
"""

import os
import json
import cv2
import csv
import argparse
from tqdm import tqdm
import numpy as np
import pycocotools.mask as maskUtils
from imantics import Mask as maskToPolygon

class StyleTrans(object):

    def __init__(self, dt_file, gt_file=None, pn_file=None):
        self.dt_file = dt_file
        self.gt_file = gt_file
        self.mid_out = None
        if pn_file is not None:
            self.mid_out = json.load(open(pn_file))
        self.points_length = 4

    def check_points_length(self, points, bbox):
        if len(points) == self.points_length:
            return points

        roi_area = cv2.contourArea(points)
        thr = 0.03
        while thr < 0.03:
            points_validate = []
            idx_remove = []
            for p in range(len(points)):
                index = list(range(len(points)))
                index.remove(p)
                for k in idx_remove:
                    index.remove(k)
                area = cv2.contourArea(points[index])
                if np.abs(roi_area - area) / roi_area > thr:
                    points_validate.append(points[p])
                else:
                    idx_remove.append(p)
            if len(points_validate) == self.points_length:
                return np.array(points_validate)

            thr += 0.01

        # return minAreaRect
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box[box[:, 0] < bbox[0], 0] = bbox[0]
        box[box[:, 1] < bbox[1], 1] = bbox[1]
        box[box[:, 0] > bbox[0] + bbox[2], 0] = bbox[0] + bbox[2]
        box[box[:, 1] > bbox[1] + bbox[3], 1] = bbox[1] + bbox[3]

        return box.astype(np.int)

    def segm_to_polygons(self, save_name=None):
        if save_name is None:
            save_name = '.'.join(self.dt_file.split('.')[:-1]) + '_polygons.json'
        dir_ = os.path.dirname(save_name)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        
        results = json.load(open(self.dt_file))
        for itm in tqdm(results):
            _mask = maskUtils.decode(itm['segmentation']).astype(np.bool)
            _polygons = maskToPolygon(_mask).polygons()
            roi_areas = [cv2.contourArea(points) for points in _polygons.points]
            if max(roi_areas) < 1:
                continue
            idx = roi_areas.index(max(roi_areas))
            points = _polygons.points[idx]
            roi_area = roi_areas[idx]

            # eliminate unnecessarily points
            points_validate = []
            idx_remove = []
            for p in range(len(points)):
                index = list(range(len(points)))
                index.remove(p)
                for k in idx_remove:
                    index.remove(k)
                area = cv2.contourArea(points[index])
                if np.abs(roi_area - area) / roi_area > 0.00002:
                    points_validate.append(points[p])
                else:
                    idx_remove.append(p)
            points_validate = np.array(points_validate)
            # minAreaRect much better than deliminate region
            points_validate = self.check_points_length(points_validate, itm['bbox'])
            points_validate = points_validate.tolist()
            assert len(points_validate) == self.points_length, "the submision type requires {} points in each detection.".format(self.points_length)
            polygons_validate = []
            for i in range(self.points_length):
                polygons_validate.append(points_validate[i][0])
                polygons_validate.append(points_validate[i][1])
            itm['segmentation'] = [polygons_validate]
        
        self.mid_out = results
        with open(save_name, 'w') as w_obj:
            json.dump(results, w_obj)
        
        print("segm has changed to polygons with 4 points, and saved to {}".format(save_name))

    def segm_to_csv(self, score_thr=0.8, save_name=None):
        if save_name is None:
            save_name = '.'.join(self.dt_file.split('.')[:-1]) + '.csv'
        dir_ = os.path.dirname(save_name)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        if self.mid_out is None:
            self.segm_to_polygons()
        
        img_id2name = {}
        anns = json.load(open(self.gt_file))
        for itm in anns['images']:
            img_id2name[itm['id']] = itm['file_name']

        csv_submit = []
        csv_det_names = set()
        det_whole = {}
        # each element: [filename, X1, Y1, X2, Y2, X3, Y3, X4, Y4, type]
        for itm in tqdm(self.mid_out):
            if itm['image_id'] not in det_whole:
                det_whole[itm['image_id']] = []
            det_whole[itm['image_id']].append(itm)
            if itm['score'] < score_thr:
                continue
            csv_det_names.add(img_id2name[itm['image_id']])

            csv_row = []
            csv_row.append(img_id2name[itm['image_id']])
            for p in itm['segmentation'][0]:
                csv_row.append(str(int(p)))
            _type = itm['category_id']
            if _type == 21:
                _type = 0
            csv_row.append(str(_type))
            csv_submit.append(csv_row)
        
        # check whether whole images has it's detection
        _counter = 0
        for key, value in img_id2name.items():
            if value in csv_det_names:
                continue
            csv_row = []
            csv_row.append(value)
            if key in det_whole:
                # choose the highest score itm
                score = -1
                select_itm = None
                for itm in det_whole[key]:
                    if itm['score'] > score:
                        select_itm = itm
                for p in select_itm['segmentation'][0]:
                    csv_row.append(str(int(p)))
                _type = select_itm['category_id']
                if _type == 21:
                    _type = 0
                csv_row.append(str(_type))
            else:
                _counter += 1
                for i in range(9):
                    csv_row.append('0')
            csv_submit.append(csv_row)
        
        print("There are {} images have not been detected traffic signs.".format(_counter))

        with open(save_name, 'w') as w_obj:
            csv_writer = csv.writer(w_obj)
            headers = ['filename', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'type']
            csv_writer.writerow(headers)
            csv_writer.writerows(csv_submit)

        print("Results has been saved to {}".format(save_name))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Transformation")
    parser.add_argument('--dt-file', type=str, help="detection coco style results")
    parser.add_argument('--gt-file', type=str, help="annotations providing id -> image name information")
    parser.add_argument('--pn-file', type=str, default=None, help="annotations providing id -> image name information")
    args = parser.parse_args()

    if not os.path.exists(args.gt_file):
        print("File Not Found Error: {}".format(args.gt_file))
        exit(404)
    if not os.path.exists(args.dt_file):
        print("File Not Found Error: {}".format(args.dt_file))
        exit(404)

    st = StyleTrans(args.dt_file, args.gt_file, args.pn_file)
    # st.segm_to_polygons(save_name='work_dirs/conlusions/results_polygons.json')
    st.segm_to_csv(score_thr=0.8, save_name='work_dirs/conlusions/predict.csv')
