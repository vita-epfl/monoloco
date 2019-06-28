"""Run monoloco over all the pifpaf joints of KITTI images
and extract and save the annotations in txt files"""


import math
import os
import glob
import json
import shutil
import itertools

import numpy as np
import torch

from predict.monoloco import MonoLoco
from utils.kitti import get_calibration
from eval.geom_baseline import compute_distance
from utils.pifpaf import preprocess_pif
from utils.camera import get_depth_from_distance, get_keypoints, pixel_to_camera, pixel_to_camera_old


def generate_kitti(model, dir_ann, p_dropout=0.2, n_dropout=0):

    cnt_ann = 0
    cnt_file = 0
    cnt_no_file = 0

    dir_kk = os.path.join('data', 'kitti', 'calib')
    dir_out = os.path.join('data', 'kitti', 'monoloco')

    # Remove the output directory if alreaady exists (avoid residual txt files)
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    print("Created empty output directory for txt files")

    # Load monoloco
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    monoloco = MonoLoco(model_path=model, device=device, n_dropout=n_dropout, p_dropout=p_dropout)

    # Run monoloco over the list of images
    list_basename = factory_basename(dir_ann)
    for basename in list_basename:
        path_calib = os.path.join(dir_kk, basename + '.txt')
        annotations, kk, tt, _ = factory_file(path_calib, dir_ann, basename)
        boxes, keypoints = preprocess_pif(annotations, im_size=(1242, 374))

        if not keypoints:
            cnt_no_file += 1
        else:
            # Run the network and the geometric baseline
            outputs, varss = monoloco.forward(keypoints, kk)
            dds_geom = eval_geometric(keypoints, kk, average_y=0.48)

            # Update counting
            cnt_ann += len(boxes)
            cnt_file += 1

        # Save the file
        all_outputs = [outputs, varss, dds_geom]
        all_inputs = [boxes, keypoints]
        all_params = [kk, tt]
        path_txt = os.path.join(dir_out, basename + '.txt')
        save_txts(path_txt, all_inputs, all_outputs, all_params)

    # Print statistics
    print("Saved in {} txt {} annotations. Not found {} images"
          .format(cnt_file, cnt_ann, cnt_no_file))


def save_txts(path_txt, all_inputs, all_outputs, all_params):

    outputs, varss, dds_geom = all_outputs[:]
    uv_boxes, keypoints = all_inputs[:]
    kk, tt = all_params[:]

    uv_centers = get_keypoints(keypoints, mode='center')
    xy_centers = pixel_to_camera(uv_centers, kk, 1)
    zzs = get_depth_from_distance(outputs, xy_centers)

    with open(path_txt, "w+") as ff:
        for idx in range(outputs.shape[0]):
            xx_1 = float(xy_centers[idx][0])
            yy_1 = float(xy_centers[idx][1])
            std_ale = math.exp(float(outputs[idx][1])) * float(outputs[idx][0])
            
            cam_0 = [xx_1 * zzs[idx] + tt[0], yy_1 * zzs[idx] + tt[1], zzs[idx] + tt[2]]
            cam_0.append(math.sqrt(cam_0[0] ** 2 + cam_0[1] ** 2 + cam_0[2] ** 2))  # X, Y, Z, D

            for el in uv_boxes[idx][:]:
                ff.write("%s " % el)
            for el in cam_0:
                ff.write("%s " % el)
            ff.write("%s " % std_ale)
            ff.write("%s " % varss[idx])
            ff.write("%s " % dds_geom[idx])
            ff.write("\n")

        # Save intrinsic matrix in the last row
        for kk_el in itertools.chain(*kk):  # Flatten a list of lists
            ff.write("%f " % kk_el)
        ff.write("\n")


def factory_basename(dir_ann):
    """ Return all the basenames in the annotations folder"""

    list_ann = glob.glob(os.path.join(dir_ann, '*.json'))
    list_basename = [os.path.basename(x).split('.')[0] for x in list_ann]
    assert list_basename, " Missing json annotations file to create txt files for KITTI datasets"

    return list_basename


def factory_file(path_calib, dir_ann, basename, ite=0):
    """Choose the annotation and the calibration files. Stereo option with ite = 1"""

    stereo_file = True
    p_left, p_right = get_calibration(path_calib)

    if ite == 0:
        kk, tt = p_left[:]
        path_ann = os.path.join(dir_ann, basename + '.png.pifpaf.json')
    else:
        kk, tt = p_right[:]
        path_ann = os.path.join(dir_ann + '_right', basename + '.png.pifpaf.json')

    try:
        with open(path_ann, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        annotations = None
        if ite == 1:
            stereo_file = False

    return annotations, kk, tt, stereo_file


def eval_geometric(keypoints, kk, average_y=0.48):
    """ Evaluate geometric distance"""

    dds_geom = []

    uv_centers = get_keypoints(keypoints, mode='center')
    uv_shoulders = get_keypoints(keypoints, mode='shoulder')
    uv_hips = get_keypoints(keypoints, mode='hip')

    xy_centers = pixel_to_camera(uv_centers, kk, 1)
    xy_shoulders = pixel_to_camera(uv_shoulders, kk, 1)
    xy_hips = pixel_to_camera(uv_hips, kk, 1)

    for idx, xy_center in enumerate(xy_centers):
        zz = compute_distance(xy_shoulders[idx], xy_hips[idx], average_y)
        xyz_center = np.array([xy_center[0], xy_center[1], zz])
        dd_geom = float(np.linalg.norm(xyz_center))
        dds_geom.append(dd_geom)

    return dds_geom





