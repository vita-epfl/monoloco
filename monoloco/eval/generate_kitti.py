"""Run monoloco over all the pifpaf joints of KITTI images
and extract and save the annotations in txt files"""


import math
import os
import glob
import json
import shutil
import itertools
import copy

import numpy as np
import torch

from ..network import MonoLoco
from ..network.process import preprocess_pifpaf
from ..eval.geom_baseline import compute_distance
from ..utils import get_keypoints, pixel_to_camera, xyz_from_distance, get_calibration, depth_from_disparity


class GenerateKitti:

    def __init__(self, model, dir_ann, p_dropout=0.2, n_dropout=0):

        # Load monoloco
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.monoloco = MonoLoco(model_path=model, device=device, n_dropout=n_dropout, p_dropout=p_dropout)
        self.dir_out = os.path.join('data', 'kitti', 'monoloco')
        self.dir_ann = dir_ann

        # List of images
        self.list_basename = factory_basename(dir_ann)
        self.dir_kk = os.path.join('data', 'kitti', 'calib')

    def run_mono(self):
        """Run Monoloco and save txt files for KITTI evaluation"""

        cnt_ann = cnt_file = cnt_no_file = 0
        dir_out = os.path.join('data', 'kitti', 'monoloco')
        # Remove the output directory if alreaady exists (avoid residual txt files)
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        os.makedirs(dir_out)
        print("\nCreated empty output directory for txt files")

        # Run monoloco over the list of images
        for basename in self.list_basename:
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)
            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1242, 374))

            if not keypoints:
                cnt_no_file += 1
                continue
            else:
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

            # Update counting
            cnt_ann += len(boxes)
            cnt_file += 1
        print("Saved in {} txt {} annotations. Not found {} images\n".format(cnt_file, cnt_ann, cnt_no_file))

    def run_stereo(self):
        """Run monoloco on left and right images and alculate disparity if a match is found"""

        cnt_ann = cnt_file = cnt_no_file = cnt_no_stereo = cnt_disparity = 0
        dir_out = os.path.join('data', 'kitti', 'monoloco_stereo')

        # Remove the output directory if alreaady exists (avoid residual txt files)
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        os.makedirs(dir_out)
        print("Created empty output directory for txt STEREO files")

        for basename in self.list_basename:
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            stereo = True

            for mode in ['left', 'right']:
                annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename, mode=mode)
                boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1242, 374))

                if not keypoints and mode == 'left':
                    cnt_no_file += 1
                    break

                elif not keypoints and mode == 'right':
                    stereo = False

                else:
                    # Run the network and the geometric baseline
                    outputs, varss = self.monoloco.forward(keypoints, kk)
                    dds_geom = eval_geometric(keypoints, kk, average_y=0.48)

                    uv_centers = get_keypoints(keypoints, mode='bottom')  # Kitti uses the bottom to calculate depth
                    xy_centers = pixel_to_camera(uv_centers, kk, 1)

                if mode == 'left':
                    outputs_l = outputs.detach().cpu()
                    varss_l = varss.detach().cpu()
                    zzs_l = xyz_from_distance(outputs_l[:, 0:1], xy_centers)[:, 2].tolist()
                    kps_l = copy.deepcopy(keypoints)
                    boxes_l = boxes
                    xy_centers_l = xy_centers
                    dds_geom_l = dds_geom
                    kk_l = kk
                    tt_l = tt

                else:
                    kps_r = copy.deepcopy(keypoints)

            if stereo:
                zzs, cnt = depth_from_disparity(zzs_l, kps_l, kps_r)
                cnt_disparity += cnt
            else:
                zzs = zzs_l

            # Save the file
            all_outputs = [outputs_l, varss_l, dds_geom_l, zzs]
            all_inputs = [boxes_l, xy_centers_l]
            all_params = [kk_l, tt_l]
            path_txt = os.path.join(dir_out, basename + '.txt')
            save_txts(path_txt, all_inputs, all_outputs, all_params)

            # Update counting
            cnt_ann += len(boxes_l)
            cnt_file += 1

        # Print statistics
        print("Saved in {} txt {} annotations. Not found {} images."
              .format(cnt_file, cnt_ann, cnt_no_file))
        print("Annotations corrected using stereo: {:.1f}%, not found {} stereo files"
              .format(cnt_disparity / cnt_ann * 100, cnt_no_stereo))


def save_txts(path_txt, all_inputs, all_outputs, all_params):

    outputs, varss, dds_geom, zzs = all_outputs[:]
    uv_boxes, xy_centers = all_inputs[:]
    kk, tt = all_params[:]

    with open(path_txt, "w+") as ff:
        for idx in range(outputs.shape[0]):

            xx = float(xy_centers[idx][0]) * zzs[idx] + tt[0]
            yy = float(xy_centers[idx][1]) * zzs[idx] + tt[1]
            zz = zzs[idx] + tt[2]
            dd = math.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
            cam_0 = [xx, yy, zz, dd]

            for el in uv_boxes[idx][:]:
                ff.write("%s " % el)
            for el in cam_0:
                ff.write("%s " % el)
            ff.write("%s " % float(outputs[idx][1]))
            ff.write("%s " % float(varss[idx]))
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

    try:
        with open(path_ann, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        annotations = []

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
