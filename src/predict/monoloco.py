
"""
From a json file output images and json annotations
"""

from collections import defaultdict
import os
import logging
import time

import numpy as np
import torch

from models.architectures import LinearModel
from utils.camera import get_depth
from utils.misc import laplace_sampling, get_idx_max
from utils.normalize import unnormalize_bi
from utils.pifpaf import get_input_data, preprocess_pif


class MonoLoco:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    output_size = 2
    input_size = 17 * 2
    linear_size = 256
    iou_min = 0.25
    n_samples = 100

    def __init__(self, model, device, n_dropout=0):

        self.device = device
        self.n_dropout = n_dropout
        if self.n_dropout > 0:
            self.epistemic = True
        else:
            self.epistemic = False

        # load the model parameters
        self.model = LinearModel(input_size=self.input_size, output_size=self.output_size, linear_size=self.linear_size)
        self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def forward(self, pifpaf_out, im_size, kk, dic_names, image_name):

        # Preprocess pifpaf outputs
        boxes, keypoints = preprocess_pif(pifpaf_out, im_size)

        (inputs_norm, xy_kps), (uv_kps, uv_boxes, uv_centers, uv_shoulders) = \
            get_input_data(boxes, keypoints, kk, left_to_right=True)

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

        # Create output files
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

        return dic_out
