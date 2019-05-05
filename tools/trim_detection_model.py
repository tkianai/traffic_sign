import os
import argparse

import torch

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-path', help='pretrained model path')
    parser.add_argument('--save-path', help='the path to save transfered model')

    args = parser.parse_args()
    return args

def trim_model(args):

    origin_model = torch.load(args.pretrained_path)
    origin_model = origin_model['state_dict']

    # remove the unfitted layers
    removed_keys = ['fc_cls', 'fc_reg', 'conv_logits']
    newdict = dict(origin_model)
    for key in origin_model.keys():
        for removed_key in removed_keys:
            if removed_key in key:
                newdict.pop(key)
                break

    res_dict = {}
    res_dict['state_dict'] = newdict
    torch.save(res_dict, args.save_path)


if __name__ == "__main__":
    args = parse_args()
    trim_model(args)