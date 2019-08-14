
""""Generate stereo baselines for kitti evaluation"""

import math
import os
import glob
import json
import shutil
import itertools
import copy

import torch

from ..network.process import preprocess_pifpaf
from ..utils import get_keypoints, pixel_to_camera, xyz_from_distance, depth_from_disparity, factory_basename, \
    open_annotations, get_calibration


def re_id_representation():
    """George: From images crop - run reid and compute representation vectors"""
    return None


def pose_representation(dir_ann, basename):
    """Extract pifpaf pose representation for distance comparison from left-right images"""
    path_l = os.path.join(dir_ann, basename + '.png.pifpaf.json')
    path_r = os.path.join(dir_ann, basename + '.png.pifpaf.json')
    annotations_l = open_annotations(path_l)
    annotations_r = open_annotations(path_r)
    boxes_l, keypoints_l = preprocess_pifpaf(annotations_l, im_size=(1242, 374))
    boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(1242, 374))
    return boxes_l, keypoints_l, boxes_r, keypoints_r


def generate_baselines(dir_ann):
    """Create txt files for evaluation for stereo baselines"""
    cnt_ann = cnt_file = cnt_no_file = cnt_no_stereo = cnt_disparity = 0
    dir_out = os.path.join('data', 'kitti', 'baseline_stereo')
    dir_kk = os.path.join('data', 'kitti', 'calib')

    # List of images
    list_basename = factory_basename(dir_ann)

    # Remove the output directory if alreaady exists (avoid residual txt files)
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    print("Created empty output directory for txt baseline stereo files")

    for basename in list_basename:
        path_calib = os.path.join(dir_kk, basename + '.txt')
        p_left, _ = get_calibration(path_calib)
        boxes_l, keypoints_l, boxes_r, keypoints_r = pose_representation(dir_ann, basename)

        if not keypoints_r and not keypoints_r:
            cnt_no_file += 1
            break

        matrix_similarity = cosine_distance(keypoints_l, keypoints_r)
        zzs = similarity_to_depth(matrix_similarity)

        # Save the file
        path_txt = os.path.join(dir_out, basename + '.txt')
        save_txts(path_txt, all_inputs, all_outputs, p_left)
        # Update counting
        cnt_ann += len(boxes_l)
        cnt_file += 1

        # Print statistics
        print("Saved in {} txt {} annotations. Not found {} images."
              .format(cnt_file, cnt_ann, cnt_no_file))
        print("Annotations corrected using stereo: {:.1f}%, not found {} stereo files"
              .format(cnt_disparity / cnt_ann * 100, cnt_no_stereo))
