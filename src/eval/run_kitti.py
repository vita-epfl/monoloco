"""Run monoloco over all the pifpaf joints of KITTI images
and extract and save the annotations in txt files"""


import math
import os
import glob
import json
import logging
import shutil

import numpy as np
import torch

from models.architectures import LinearModel
from utils.misc import laplace_sampling
from utils.kitti import eval_geometric, get_calibration
from utils.normalize import unnormalize_bi
from utils.pifpaf import get_network_inputs, preprocess_pif, get_input_data
from utils.camera import get_depth_from_distance, get_keypoints_torch


class RunKitti:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    cnt_ann = 0
    cnt_file = 0
    cnt_no_file = 0
    average_y = 0.48
    n_samples = 100

    def __init__(self, model, dir_ann, dropout, hidden_size, n_stage, n_dropout):

        self.dir_ann = dir_ann
        self.n_dropout = n_dropout
        self.dir_kk = os.path.join('data', 'kitti', 'calib')
        self.dir_out = os.path.join('data', 'kitti', 'monoloco')

        # Remove the output directory if alreaady exists (avoid residual txt files)
        if os.path.exists(self.dir_out):
            shutil.rmtree(self.dir_out)
        os.makedirs(self.dir_out)
        print("Created output directory for txt files")

        self.list_basename = factory_basename(dir_ann)

        # Load the model
        input_size = 17 * 2
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = LinearModel(input_size=input_size, output_size=2, linear_size=hidden_size,
                                 p_dropout=dropout, num_stage=n_stage)
        self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def run(self):

        # Run inference

        for basename in self.list_basename:
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt, _ = factory_file(path_calib, self.dir_ann, basename)

            boxes, keypoints = preprocess_pif(annotations)

            inputs_torch = get_network_inputs(torch.tensor(keypoints).to(self.device), torch.tensor(kk).to(self.device))
            uv_centers_t = get_keypoints_torch(keypoints, mode='center')
            uv_shoulders_t = torch.round(get_keypoints_torch(keypoints, mode='shoulder')).int().tolist()

            (inputs, xy_kps), (uv_kps, uv_boxes, uv_centers, uv_shoulders) = get_input_data(boxes, keypoints, kk)

            dds_geom, xy_centers = eval_geometric(uv_kps, uv_centers, uv_shoulders, kk, average_y=0.48)

            # Update counting
            self.cnt_ann += len(boxes)
            if not inputs:
                self.cnt_no_file += 1
            else:
                self.cnt_file += 1

            # Run the model
            inputs = torch.from_numpy(np.array(inputs)).float().to(self.device)
            if self.n_dropout > 0:
                total_outputs = torch.empty((0, len(uv_boxes))).to(self.device)
                self.model.dropout.training = True
                for _ in range(self.n_dropout):
                    outputs = self.model(inputs)
                    outputs = unnormalize_bi(outputs)
                    samples = laplace_sampling(outputs, self.n_samples)
                    total_outputs = torch.cat((total_outputs, samples), 0)
                varss = total_outputs.std(0)
            else:
                varss = [0]*len(uv_boxes)

            # Don't use dropout for the mean prediction and aleatoric uncertainty
            self.model.dropout.training = False
            outputs_net = self.model(inputs)
            outputs = outputs_net.cpu().detach().numpy()

            list_zzs = get_depth_from_distance(outputs, xy_centers)
            all_outputs = [outputs, varss, dds_geom]
            all_inputs = [uv_boxes, xy_centers, xy_kps]
            all_params = [kk, tt]

            # Save the file
            all_outputs.append(list_zzs)
            path_txt = os.path.join(self.dir_out, basename + '.txt')
            save_txts(path_txt, all_inputs, all_outputs, all_params)

        # Print statistics
        print("Saved in {} txt {} annotations. Not found {} images"
              .format(self.cnt_file, self.cnt_ann, self.cnt_no_file))


def save_txts(path_txt, all_inputs, all_outputs, all_params):

    outputs, varss, dds_geom, zzs = all_outputs[:]
    uv_boxes, xy_centers, xy_kps = all_inputs[:]
    kk, tt = all_params[:]

    with open(path_txt, "w+") as ff:
        for idx in range(outputs.shape[0]):
            xx_1 = float(xy_centers[idx][0])
            yy_1 = float(xy_centers[idx][1])
            xy_kp = xy_kps[idx]
            dd = float(outputs[idx][0])
            std_ale = math.exp(float(outputs[idx][1])) * dd
            zz = zzs[idx]
            xx_cam_0 = xx_1 * zz + tt[0]
            yy_cam_0 = yy_1 * zz + tt[1]
            zz_cam_0 = zz + tt[2]
            dd_cam_0 = math.sqrt(xx_cam_0 ** 2 + yy_cam_0 ** 2 + zz_cam_0 ** 2)

            uv_box = uv_boxes[idx]

            twodecimals = ["%.3f" % vv for vv in [uv_box[0], uv_box[1], uv_box[2], uv_box[3],
                                                  xx_cam_0, yy_cam_0, zz_cam_0, dd_cam_0,
                                                  std_ale, varss[idx], uv_box[4], dds_geom[idx]]]

            # keypoints_str = ["%.5f" % vv for vv in xy_kp]
            # for item in twodecimals:
            #     ff.write("%s " % item)
            # for item in keypoints_str:
            #     ff.write("%s " % item)
            # ff.write("\n")

        # Save intrinsic matrix in the last row
        kk_list = kk.reshape(-1, ).tolist()
        for kk_el in kk_list:
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

    return annotations, kk.tolist(), tt, stereo_file




