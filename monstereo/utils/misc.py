import json
import shutil
import os

import numpy as np


def append_cluster(dic_jo, phase, xx, ys, kps):
    """Append the annotation based on its distance"""

    if ys[3] <= 10:
        dic_jo[phase]['clst']['10']['kps'].append(kps)
        dic_jo[phase]['clst']['10']['X'].append(xx)
        dic_jo[phase]['clst']['10']['Y'].append(ys)
    elif ys[3] <= 20:
        dic_jo[phase]['clst']['20']['kps'].append(kps)
        dic_jo[phase]['clst']['20']['X'].append(xx)
        dic_jo[phase]['clst']['20']['Y'].append(ys)
    elif ys[3] <= 30:
        dic_jo[phase]['clst']['30']['kps'].append(kps)
        dic_jo[phase]['clst']['30']['X'].append(xx)
        dic_jo[phase]['clst']['30']['Y'].append(ys)
    elif ys[3] < 50:
        dic_jo[phase]['clst']['50']['kps'].append(kps)
        dic_jo[phase]['clst']['50']['X'].append(xx)
        dic_jo[phase]['clst']['50']['Y'].append(ys)
    else:
        dic_jo[phase]['clst']['>50']['kps'].append(kps)
        dic_jo[phase]['clst']['>50']['X'].append(xx)
        dic_jo[phase]['clst']['>50']['Y'].append(ys)


def get_task_error(dd):
    """Get target error not knowing the gender, modeled through a Gaussian Mixure model"""
    mm = 0.046
    return dd * mm


def get_pixel_error(zz_gt):
    """calculate error in stereo distance due to 1 pixel mismatch (function of depth)"""

    disp = 0.54 * 721 / zz_gt
    error = abs(zz_gt - 0.54 * 721 / (disp - 1))
    return error


def open_annotations(path_ann):
    try:
        with open(path_ann, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        annotations = []
    return annotations


def make_new_directory(dir_out):
    """Remove the output directory if already exists (avoid residual txt files)"""
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    print("Created empty output directory for {} txt files".format(dir_out))


def normalize_hwl(lab):

    AV_H = 1.72
    AV_W = 0.75
    AV_L = 0.68
    HLW_STD = 0.1

    hwl = lab[4:7]
    hwl_new = list((np.array(hwl) - np.array([AV_H, AV_W, AV_L])) / HLW_STD)
    lab_new = lab[0:4] + hwl_new + lab[7:]
    return lab_new
