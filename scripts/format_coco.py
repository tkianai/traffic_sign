"""This helper function aims to transfer csv annotations to coco-style for training
"""

import os
import csv
import argparse
import json
from random import shuffle
from PIL import Image

CATEGORIES = [
    {'id': 1, 'name': '停车场'},
    {'id': 2, 'name': '停车让行'},
    {'id': 3, 'name': '右侧行驶'},
    {'id': 4, 'name': '向左和向右转弯'},
    {'id': 5, 'name': '大客车通行'},
    {'id': 6, 'name': '左侧行驶'},
    {'id': 7, 'name': '慢行'},
    {'id': 8, 'name': '机动车直行和右转弯'},
    {'id': 9, 'name': '注意行人'},
    {'id': 10, 'name': '环岛行驶'},
    {'id': 11, 'name': '直行和右转弯'},
    {'id': 12, 'name': '禁止大客车通行'},
    {'id': 13, 'name': '禁止摩托车通行'},
    {'id': 14, 'name': '禁止机动车通行'},
    {'id': 15, 'name': '禁止非机动车通行'},
    {'id': 16, 'name': '禁止鸣喇叭'},
    {'id': 17, 'name': '立交直行和转弯行驶'},
    {'id': 18, 'name': '限制速度40公里每小时'},
    {'id': 19, 'name': '限速30公里每小时'},
    {'id': 20, 'name': '鸣喇叭'},
    {'id': 21, 'name': '其他'},  # origin label: 0
]

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer csv annotations to coco style.")
    parser.add_argument('--csv-file', type=str, default='datasets/train_label_fix.csv', help='the original csv style train label file.')
    parser.add_argument('--img-dir', type=str, default='datasets/images/', help='the directory contains train and test images.')
    parser.add_argument('--thr', type=int, default=15, help='the validation set percentage.')

    args = parser.parse_args()
    return args


def convert_poly_to_rect(coordinateList):
    X = [int(coordinateList[2 * i]) for i in range(int(len(coordinateList) / 2))]
    Y = [int(coordinateList[2 * i + 1]) for i in range(int(len(coordinateList) / 2))]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)

    return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin]

def gather_infos_from_csv(csv_file):
    """extract infos from label file
    
    Arguments:
        csv_file {path} -- label file path
    
    Raises:
        ValueError: judge whether file exists
    
    Returns:
        dict -- each element is one image
    """
    if not os.path.exists(csv_file):
        raise ValueError("File not Found error: {}".format(csv_file))
        exit()

    infos = {}
    with open(csv_file, 'r') as r_obj:
        reader = csv.reader(r_obj)
        for i, line in enumerate(reader):
            if i != 0:
                if line[0] not in infos:
                    infos[line[0]] = []
                assert len(line) == 10, "annotation [{}] error!".format(line)
                points = [int(p) for p in line[1:-1]]
                info = {}
                info['segmentation'] = [points]
                info['bbox'] = convert_poly_to_rect(points)
                info['type'] = int(line[-1])
                infos[line[0]].append(info)
                    
            else:
                # "filename,X1,Y1,X2,Y2,X3,Y3,X4,Y4,type"
                print('The header is: ' + ','.join(line))
    
    print("There are {} valid images.".format(len(infos)))
    return infos


def add_annotations(coco_output, annotations, annotation_id, image_id):
    for annotation in annotations:
        ann_info = {}
        ann_info['id'] = annotation_id
        ann_info['image_id'] = image_id
        ann_info['segmentation'] = annotation['segmentation']
        ann_info['category_id'] = annotation['type'] if annotation['type'] != 0 else 21
        ann_info['iscrowd'] = 0
        ann_info['bbox'] = annotation['bbox']
        ann_info['area'] = ann_info['bbox'][2] * ann_info['bbox'][3]
        coco_output['annotations'].append(ann_info)
        annotation_id += 1
        
    return annotation_id

def main(args):

    infos = gather_infos_from_csv(args.csv_file)
    
    image_ids = list(infos.keys())
    val_num = int(float(len(image_ids)) * args.thr / 100)
    print("There are {} images in total.".format(len(image_ids)))
    print("There are {} images for training.".format(len(image_ids) - val_num))
    print("There are {} images for validation.".format(val_num))

    shuffle(image_ids)
    image_ids_val = set(image_ids[:val_num])

    image_id = 1
    annotation_id = 1
    coco_output_train = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_val = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_test = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # process train set
    for key, value in infos.items():
        img_path = os.path.join(args.img_dir, 'train', key)
        if not os.path.exists(img_path):
            print("File not found: {}".format(img_path))
            continue
        try:
            image = Image.open(img_path)
            image.verify()
        except:
            print("File cannot open: {}".format(img_path))
            continue
        
        if image is None:
            print("File has destroyed: {}".format(img_path))
            continue
        
        image_info = {}
        image_info['id'] = image_id
        image_info['width'] = image.size[0]
        image_info['height'] = image.size[1]
        image_info['file_name'] = key

        if key in image_ids_val:
            coco_output_val['images'].append(image_info)
            annotation_id = add_annotations(
                coco_output_val, value, annotation_id, image_id)
        else:
            coco_output_train['images'].append(image_info)
            annotation_id = add_annotations(
                coco_output_train, value, annotation_id, image_id)
        
        image_id += 1

    # process test set
    image_id = 1
    for file in os.listdir(os.path.join(args.img_dir, 'test')):
        image = Image.open(os.path.join(args.img_dir, 'test', file))
        image_info = {}
        image_info['id'] = image_id
        image_info['width'] = image.size[0]
        image_info['height'] = image.size[1]
        image_info['file_name'] = file
        coco_output_test['images'].append(image_info)
        image_id += 1

    # write to json file
    with open('datasets/train.json', 'w') as w_obj:
        json.dump(coco_output_train, w_obj)
    
    with open('datasets/val.json', 'w') as w_obj:
        json.dump(coco_output_val, w_obj)
    
    with open('datasets/test.json', 'w') as w_obj:
        json.dump(coco_output_test, w_obj)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
