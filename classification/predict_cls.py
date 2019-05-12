"""This script aims to classify bounding box
"""

import os
import cv2
import json
import argparse

import torch
from torchvision.models import resnet101
import torchvision.transforms as transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Classifying bounding boxes")
    parser.add_argument('--dt-file', help="Detected results")
    parser.add_argument('--gt-file', help="Groundtruth annotation file")
    parser.add_argument('--img-dir', help="Directory for test image")
    parser.add_argument('--save', default='./work_dirs/conclusion/classified_up.json',
                        help="The classified bounding box json file")

    return parser.parse_args()


def classify(args):
    
    # model initialization
    model = resnet101()
    checkpoint = torch.load('classification/resnet101/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    ann_gt = json.load(open(args.gt_file))
    id2name = {itm['id']:itm['file_name'] for itm in ann_gt['annotations']}
    results = json.load(open(args.dt_file))
    for itm in results:
        filename = id2name[itm['image_id']]
        img_path = os.path.join(args.img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError, "{}".format(img_path)
        image = cv2.imread(img_path)
        bbox = itm['bbox']

        new_bbox = []
        new_bbox.append(bbox[0] - bbox[2] / 5)
        new_bbox.append(bbox[1] - bbox[3] / 5)
        new_bbox.append(bbox[2] * 1.4)
        new_bbox.append(bbox[3] * 1.4)
        new_bbox = [int(p) for p in new_bbox]
        # 判断是否越界
        new_bbox[0] = max(0, new_bbox[0])
        new_bbox[1] = max(0, new_bbox[1])
        if new_bbox[0] + new_bbox[2] >= image.shape[1]:
            new_bbox[2] = image.shape[1] - new_bbox[0]
        if new_bbox[1] + new_bbox[3] >= image.shape[0]:
            new_bbox[3] = image.shape[0] - new_bbox[1]
        image_roi = image[new_bbox[1]:new_bbox[1] + new_bbox[3], new_bbox[0]:new_bbox[0] + new_bbox[2]]

        with torch.no_grad():
            image_roi = Image.fromarray(image_roi)
            image_roi = image_roi.unsqueeze(0).cuda()
            output = model(image_roi)
            output = output[0]
            _, idx = torch.max(output, 0)
            # category = idx + 1
            itm['category_id'] = idx + 1
    
    with open(args.save, 'w') as w_obj:
        json.dump(results, w_obj)


if __name__ == "__main__":
    args = parse_args()
    classify(args)
