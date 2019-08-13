"""Run monoloco over all the pifpaf joints of KITTI images
and extract and save the annotations in txt files"""


import math
import os
import shutil
import itertools

import numpy as np
import torch

from ..network import MonoLoco
from ..network.process import preprocess_pifpaf
from ..eval.geom_baseline import compute_distance
from ..utils import get_keypoints, pixel_to_camera, xyz_from_distance, get_calibration, factory_basename, \
    open_annotations, depth_from_disparity


class GenerateKitti:

    def __init__(self, model, dir_ann, p_dropout=0.2, n_dropout=0, stereo=True):

        # Load monoloco
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.monoloco = MonoLoco(model=model, device=device, n_dropout=n_dropout, p_dropout=p_dropout)
        self.dir_ann = dir_ann

        # List of images
        self.list_basename = factory_basename(dir_ann)
        self.dir_kk = os.path.join('data', 'kitti', 'calib')

        # Calculate stereo baselines
        self.stereo = stereo

    def run(self):
        """Run Monoloco and save txt files for KITTI evaluation"""

        cnt_ann = cnt_file = cnt_no_file = 0
        dir_out = os.path.join('data', 'kitti', 'monoloco')
        make_new_directory(dir_out)
        print("\nCreated empty output directory for txt files")

        if self.stereo:
            cnt_disparity = cnt_no_stereo = 0
            dir_out_stereo = os.path.join('data', 'kitti', 'monoloco_stereo')
            make_new_directory(dir_out_stereo)
            print("\nCreated empty output directory for txt files")

        # Run monoloco over the list of images
        for basename in self.list_basename:
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)
            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1242, 374))

            if not keypoints:
                cnt_no_file += 1
                continue

            # Run the network and the geometric baseline
            outputs, varss = self.monoloco.forward(keypoints, kk)
            dds_geom = eval_geometric(keypoints, kk, average_y=0.48)

            # Save the file
            uv_centers = get_keypoints(keypoints, mode='bottom')  # Kitti uses the bottom center to calculate depth
            xy_centers = pixel_to_camera(uv_centers, kk, 1)
            outputs = outputs.detach().cpu()
            zzs = xyz_from_distance(outputs[:, 0:1], xy_centers)[:, 2].tolist()

            all_outputs = [outputs.detach().cpu(), varss.detach().cpu(), dds_geom, zzs]
            all_inputs = [boxes, xy_centers]
            all_params = [kk, tt]
            path_txt = os.path.join(dir_out, basename + '.txt')
            save_txts(path_txt, all_inputs, all_outputs, all_params)

            # Correct using stereo disparity and save in different folder
            if self.stereo:
                annotations_r, _, _ = factory_file(path_calib, self.dir_ann, basename, mode='right')
                boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(1242, 374))

                if keypoints_r:
                    zzs, cnt = depth_from_disparity(zzs, keypoints, keypoints_r)
                    cnt_disparity += cnt
                    all_outputs[-1] = zzs
                path_txt = os.path.join(dir_out_stereo, basename + '.txt')
                save_txts(path_txt, all_inputs, zzs, all_params, mode='baseline')

            # Update counting
            cnt_ann += len(boxes)
            cnt_file += 1
        print("Saved in {} txt {} annotations. Not found {} images\n".format(cnt_file, cnt_ann, cnt_no_file))

        if self.stereo:
            print("Annotations corrected using stereo: {:.1f}%, not found {} stereo files"
                  .format(cnt_disparity / cnt_ann * 100, cnt_no_stereo))


def save_txts(path_txt, all_inputs, all_outputs, all_params, mode='monoloco'):

    assert mode in ('monoloco', 'baseline')
    if mode == 'monoloco':
        outputs, varss, dds_geom, zzs = all_outputs[:]
    else:
        zzs = all_outputs
    uv_boxes, xy_centers = all_inputs[:]
    kk, tt = all_params[:]

    with open(path_txt, "w+") as ff:
        for idx, zz_base in enumerate(zzs):

            xx = float(xy_centers[idx][0]) * zzs[idx] + tt[0]
            yy = float(xy_centers[idx][1]) * zzs[idx] + tt[1]
            zz = zz_base + tt[2]
            cam_0 = [xx, yy, zz]
            output_list = [0.]*3 + uv_boxes[idx][:-1] + [0.]*3 + cam_0 + [0.] + uv_boxes[idx][-1:]  # kitti format
            ff.write("%s " % 'pedestrian')
            for el in output_list:
                ff.write("%f " % el)

            # add additional uncertainty information
            if mode == 'monoloco':
                ff.write("%f " % float(outputs[idx][1]))
                ff.write("%f " % float(varss[idx]))
                ff.write("%f " % dds_geom[idx])
            ff.write("\n")


def factory_file(path_calib, dir_ann, basename, mode='left'):
    """Choose the annotation and the calibration files. Stereo option with ite = 1"""

    assert mode in ('left', 'right')
    p_left, p_right = get_calibration(path_calib)

    if mode == 'left':
        kk, tt = p_left[:]
        path_ann = os.path.join(dir_ann, basename + '.png.pifpaf.json')

    else:
        kk, tt = p_right[:]
        path_ann = os.path.join(dir_ann + '_right', basename + '.png.pifpaf.json')

    annotations = open_annotations(path_ann)

    return annotations, kk, tt


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


def make_new_directory(dir_out):
    """Remove the output directory if already exists (avoid residual txt files)"""
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
