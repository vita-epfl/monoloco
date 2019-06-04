
"""
From a json file output images and json annotations
"""

import sys
from collections import defaultdict
import os
import json
import logging
import time

import numpy as np
import torch
from PIL import Image

from models.architectures import LinearModel
from visuals.printer import Printer
from utils.camera import get_depth
from utils.misc import laplace_sampling, get_idx_max
from utils.normalize import unnormalize_bi
from utils.pifpaf import get_input_data


class PredictMonoLoco:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    output_size = 2
    input_size = 17 * 2

    def __init__(self, boxes, keypoints, image_path, output_path, args):
        self.boxes = boxes
        self.keypoints = keypoints
        self.image_path = image_path
        self.output_path = output_path
        self.device = args.device
        self.draw_kps = args.draw_kps
        self.z_max = args.z_max
        self.output_types = args.output_types
        self.path_gt = args.path_gt
        self.show = args.show
        self.n_samples = 100
        self.n_dropout = args.n_dropout
        if self.n_dropout > 0:
            self.epistemic = True
        else:
            self.epistemic = False
        self.iou_min = 0.25

        # load the model parameters
        self.model = LinearModel(input_size=self.input_size, output_size=self.output_size, linear_size=args.hidden_size)
        self.model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def run(self):
        # Extract calibration matrix if ground-truth file is present or use a default one
        cnt = 0
        dic_names, kk = factory_for_gt(self.path_gt, self.image_path)
        (inputs_norm, xy_kps), (uv_kps, uv_boxes, uv_centers, uv_shoulders) = \
            get_input_data(self.boxes, self.keypoints, kk, left_to_right=True)

        # Conversion into torch tensor
        if inputs_norm:
            with torch.no_grad():
                inputs = torch.from_numpy(np.array(inputs_norm)).float()
                inputs = inputs.to(self.device)
                # self.model.to("cpu")
                start = time.time()
                # Manually reactivate dropout in eval
                self.model.dropout.training = True
                total_outputs = torch.empty((0, len(xy_kps))).to(self.device)

                if self.n_dropout > 0:
                    for _ in range(self.n_dropout):
                        outputs = self.model(inputs)
                        outputs = unnormalize_bi(outputs)
                        samples = laplace_sampling(outputs, self.n_samples)
                        total_outputs = torch.cat((total_outputs, samples), 0)
                    varss = total_outputs.std(0)
                else:
                    varss = [0] * len(inputs_norm)

                # # Don't use dropout for the mean prediction
                start_single = time.time()
                self.model.dropout.training = False
                outputs = self.model(inputs)
                outputs = unnormalize_bi(outputs)
                end = time.time()
                print("Total Forward pass time with {} forward passes = {:.2f} ms"
                      .format(self.n_dropout, (end-start) * 1000))
                print("Single forward pass time = {:.2f} ms".format((end - start_single) * 1000))

        # Print image and save json
        dic_out = defaultdict(list)
        if dic_names:
            name = os.path.basename(self.image_path)
            boxes_gt, dds_gt = dic_names[name]['boxes'], dic_names[name]['dds']

        for idx, box in enumerate(uv_boxes):
            dd_pred = float(outputs[idx][0])
            ale = float(outputs[idx][1])
            var_y = float(varss[idx])

            # Find the corresponding ground truth if available
            if dic_names:
                idx_max, iou_max = get_idx_max(box, boxes_gt)
                if iou_max > self.iou_min:
                    dd_real = dds_gt[idx_max]
                    boxes_gt.pop(idx_max)
                    dds_gt.pop(idx_max)
                # In case of no matching
                else:
                    dd_real = 0
            # In case of no ground truth
            else:
                dd_real = dd_pred

            uv_center = uv_centers[idx]
            xyz_real = get_depth(uv_center, kk, dd_real)
            xyz_pred = get_depth(uv_center, kk, dd_pred)
            dic_out['boxes'].append(box)
            dic_out['dds_real'].append(dd_real)
            dic_out['dds_pred'].append(dd_pred)
            dic_out['stds_ale'].append(ale)
            dic_out['stds_epi'].append(var_y)
            dic_out['xyz_real'].append(xyz_real)
            dic_out['xyz_pred'].append(xyz_pred)
            dic_out['xy_kps'].append(xy_kps[idx])
            dic_out['uv_kps'].append(uv_kps[idx])
            dic_out['uv_centers'].append(uv_center)
            dic_out['uv_shoulders'].append(uv_shoulders[idx])

        if any((xx in self.output_types for xx in ['front', 'bird', 'combined'])):
            printer = Printer(self.image_path, self.output_path, dic_out, kk,  output_types=self.output_types,
                              show=self.show, z_max=self.z_max, epistemic=self.epistemic)
            printer.print()

        if 'json' in self.output_types:
            with open(os.path.join(self.output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)

        sys.stdout.write('\r' + 'Saving image {}'.format(cnt) + '\t')


def factory_for_gt(path_gt, image_path):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        print('-' * 120 + "\nMonoloco: Ground-truth file opened\n")
    except FileNotFoundError:
        print('-' * 120 + "\nMonoloco: ground-truth file not found\n")
        dic_names = {}

    try:
        name = os.path.basename(image_path)
        kk = dic_names[name]['K']
        print("Monoloco: matched ground-truth file!\n" + '-' * 120)
    except KeyError:
        dic_names = None
        with open(image_path, 'rb') as f:
            im = Image.open(f)
            x_factor = im.size[0] / 1600
            y_factor = im.size[1] / 900
            pixel_factor = (x_factor + y_factor) / 2
            if im.size[0] / im.size[1] > 2.5:
                kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
            else:
                kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                      [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                      [0., 0., 1.]]  # nuScenes calibration
        print("Ground-truth annotations for the image not found\n"
              "Using a standard calibration matrix...\n" + '-' * 120)

    return dic_names, kk
