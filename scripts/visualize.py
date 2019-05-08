"""This helper function aims to visualize the detection results and
the comparison between detected box and groundtruth.
"""
import os
import cv2
import json
import csv
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Utils")
    parser.add_argument('--img-dir', required=True, help="Image directory.")
    parser.add_argument('--save-dir', default="work_dirs/visualization/bbox", help="Image directory.")
    parser.add_argument('--dt-file', default=None, help="Detected json file from model.")
    parser.add_argument('--csv', default=None, help="CSV submitted style file.")
    parser.add_argument('--gt-file', default=None, help="groundtruth json file from model.")
    
    args = parser.parse_args()
    if args.dt_file is None and args.csv is None:
        print("dt-file | csv should be provided at least one item.")
        exit(200)
    if args.dt_file not None and args.gt_file is None:
        print("Must provide gt-file to identify the image name from id.")
        exit(201)
    
    return args


def json_visualize(gt_file, dt_file, img_dir, save_dir, mode=None, VIS_N=100):
    anns = json.load(open(gt_file))
    id2name = {itm['id']: itm['file_name'] for itm in anns['images']}
    anns_bbox = None
    if len(anns['annotations']) != 0:
        anns_bbox = {}
        for itm in anns['annotations']:
            file_name = id2name[itm['image_id']]
            if file_name not in anns_bbox:
                anns_bbox[file_name] = {
                    "bbox": [],
                    "category": [],
                }
            anns_bbox[file_name]['bbox'].append(itm['bbox'])
            anns_bbox[file_name]['category'].append(itm['category_id'])
    
    results = json.load(open(dt_file))
    dt_bbox = {}
    for itm in results:
        file_name = id2name[itm['image_id']]
        if file_name not in dt_bbox:
            dt_bbox[file_name] = {
                "bbox": [],
                "score": [],
                "category": [],
            }
        dt_bbox[file_name]['bbox'].append(itm['bbox'])
        dt_bbox[file_name]['score'].append(itm['score'])
        dt_bbox[file_name]['category'].append(itm['category_id'])

    # draw bbox
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(img_dir)
    if mode not None:
        random.shuffle(files)
    for file in files[:VIS_N]:
        image = cv2.imread(os.path.join(img_dir, file))
        # draw detected box
        if file in dt_bbox:
            for i, itm in enumerate(dt_bbox[file]['bbox']):
                image = cv2.rectangle(image, (itm[0], itm[1]), (itm[0] + itm[2], itm[1] + itm[3]), (0, 255, 0), 3)
                # score
                cv2.putText(image, "{}: {}".format(dt_bbox[file]['category'][i], dt_bbox[file]['score'][i]), (itm[0], itm[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)

        if anns_bbox not None:
            if file in anns_bbox:
                for i, itm in enumerate(anns_bbox[file]['bbox']):
                    image = cv2.rectangle(image, (itm[0], itm[1]), (itm[0] + itm[2], itm[1] + itm[3]), (0, 0, 255), 3)
                    cv2.putText(image, "{}".format(anns_bbox[file]['category'][i]), (
                        itm[0], itm[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)

        # save file
        cv2.imwrite(os.path.join(save_dir, file), image)


def convert_poly_to_rect(coordinateList):
    X = [int(coordinateList[2 * i]) for i in range(int(len(coordinateList) / 2))]
    Y = [int(coordinateList[2 * i + 1]) for i in range(int(len(coordinateList) / 2))]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)

    return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin]

def csv_visualize(csv_file, img_dir, save_dir, VIS_N=100):
    """This visualize the submission file
    """
    dt_bbox = {}
    with open(csv_file, 'r') as r_obj:
        csv_obj = csv.reader(r_obj)
        headers = next(csv_obj)
        for row in csv_obj:
            if len(row) != 0 and int(row[-1]) != 0:
                file_name = row[0]
                if file_name not in dt_bbox:
                    dt_bbox[file_name] = {
                        "bbox": [],
                        "category": [],
                    }
                polygons = [int(p) for p in row[1:9]]
                dt_bbox[file_name]['bbox'].append(convert_poly_to_rect(polygons))
                dt_bbox[file_name]['category'].append(int(row[-1]))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    counter = 0
    for key, value in dt_bbox.items():
        if counter > VIS_N:
            break
        counter += 1
        # load image
        image = cv2.imread(os.path.join(img_dir, key))
        for i, itm enumerate(value['bbox']):
            image = cv2.rectangle(image, (itm[0], itm[1]), (itm[0] + itm[2], itm[1] + itm[3]), (0, 255, 0), 3)
            cv2.putText(image, "{}".format(itm['category'][i]), (itm[0], itm[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)

        # save image
        cv2.imwrite(os.path.join(save_dir, key), image)
        

def run_visualize(args):

    if args.dt_file not None:
        json_visualize(args.gt_file, args.dt_file, args.img_dir, os.path.join(args.save_dir, 'json'))
    
    if args.csv not None:
        csv_visualize(args.csv, args.img_dir, os.path.join(args.save_dir, 'csv'))

if __name__ == "__main__":
    
    args = parse_args()
    run_visualize(args)
