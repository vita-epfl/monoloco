# pylint: disable=too-many-statements, too-many-branches

"""
Loco super class for MonStereo, MonoLoco, MonoLoco++ nets.
From 2D joints to real-world distances with monocular &/or stereo cameras
"""

import math
import logging
from collections import defaultdict

import numpy as np
import torch

from ..utils import get_iou_matches, reorder_matches, get_keypoints, pixel_to_camera, xyz_from_distance,  \
    mask_joint_disparity
from .process import preprocess_monstereo, preprocess_monoloco, extract_outputs, extract_outputs_mono,\
    filter_outputs, cluster_outputs, unnormalize_bi, laplace_sampling
from ..activity import social_interactions, is_raising_hand
from .architectures import MonolocoModel, LocoModel


class Loco:
    """Class for both MonoLoco and MonStereo"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    LINEAR_SIZE_MONO = 256
    N_SAMPLES = 100

    def __init__(self, model, mode, net=None, device=None, n_dropout=0, p_dropout=0.2, linear_size=1024):

        # Select networks
        assert mode in ('mono', 'stereo'), "mode not recognized"
        self.mode = mode
        if net is None:
            if mode == 'mono':
                self.net = 'monoloco_pp'
            else:
                self.net = 'monstereo'
        else:
            assert self.net in ('monstereo', 'monoloco', 'monoloco_p', 'monoloco_pp')
            if self.net != 'monstereo':
                assert mode == 'stereo', "Assert arguments mode and net are in conflict"
            self.net = net

        if self.net == 'monstereo':
            input_size = 68
            output_size = 10
        elif self.net == 'monoloco_p':
            input_size = 34
            output_size = 9
            linear_size = 256
        elif self.net == 'monoloco_pp':
            input_size = 34
            output_size = 9
        else:
            input_size = 34
            output_size = 2

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.n_dropout = n_dropout
        self.epistemic = bool(self.n_dropout > 0)

        # if the path is provided load the model parameters
        if isinstance(model, str):
            model_path = model
            if net in ('monoloco', 'monoloco_p'):
                self.model = MonolocoModel(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                           output_size=output_size)
            else:
                self.model = LocoModel(p_dropout=p_dropout, input_size=input_size, output_size=output_size,
                                            linear_size=linear_size, device=self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            self.model = model
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def forward(self, keypoints, kk, keypoints_r=None):
        """
        Forward pass of MonSter or monoloco network
        It includes preprocessing and postprocessing of data
        """
        if not keypoints:
            return None

        with torch.no_grad():
            keypoints = torch.tensor(keypoints).to(self.device)
            kk = torch.tensor(kk).to(self.device)

            if self.net == 'monoloco':
                inputs = preprocess_monoloco(keypoints, kk, zero_center=True)
                outputs = self.model(inputs)
                bi = unnormalize_bi(outputs)
                dic_out = {'d': outputs[:, 0:1], 'bi': bi}
                dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}

            elif self.net == 'monoloco_p':
                inputs = preprocess_monoloco(keypoints, kk)
                outputs = self.model(inputs)
                dic_out = extract_outputs_mono(outputs)

            elif self.net == 'monoloco_pp':
                inputs = preprocess_monoloco(keypoints, kk)
                outputs = self.model(inputs)
                dic_out = extract_outputs(outputs)

            else:
                if keypoints_r:
                    keypoints_r = torch.tensor(keypoints_r).to(self.device)
                else:
                    keypoints_r = keypoints[0:1, :].clone()
                inputs, _ = preprocess_monstereo(keypoints, keypoints_r, kk)
                outputs = self.model(inputs)

                outputs = cluster_outputs(outputs, keypoints_r.shape[0])
                outputs_fin, _ = filter_outputs(outputs)
                dic_out = extract_outputs(outputs_fin)

                # For Median baseline
                # dic_out = median_disparity(dic_out, keypoints, keypoints_r, mask)
            if self.n_dropout > 0 and self.net != 'monstereo':
                varss = self.epistemic_uncertainty(inputs)
                dic_out['epi'] = varss
            else:
                dic_out['epi'] = [0.] * outputs.shape[0]
                # Add in the dictionary

        return dic_out

    def epistemic_uncertainty(self, inputs):
        """
        Apply dropout at test time to obtain combined aleatoric + epistemic uncertainty
        """
        assert self.net in ('monoloco', 'monoloco_p', 'monoloco_pp'), "Not supported for MonStereo"

        self.model.dropout.training = True  # Manually reactivate dropout in eval
        total_outputs = torch.empty((0, inputs.size()[0])).to(self.device)

        for _ in range(self.n_dropout):
            outputs = self.model(inputs)

            # Extract localization output
            if self.net == 'monoloco':
                db = outputs[:, 0:2]
            else:
                db = outputs[:, 2:4]

            # Unnormalize b and concatenate
            bi = unnormalize_bi(db)
            outputs = torch.cat((db[:, 0:1], bi), dim=1)

            samples = laplace_sampling(outputs, self.N_SAMPLES)
            total_outputs = torch.cat((total_outputs, samples), 0)
        varss = total_outputs.std(0)
        self.model.dropout.training = False
        return varss

    @staticmethod
    def post_process(dic_in, boxes, keypoints, kk, dic_gt=None, iou_min=0.3, reorder=True, verbose=False):
        """Post process monoloco to output final dictionary with all information for visualizations"""

        dic_out = defaultdict(list)
        if dic_in is None:
            return dic_out

        if dic_gt:
            boxes_gt = dic_gt['boxes']
            dds_gt = [el[3] for el in dic_gt['ys']]
            matches = get_iou_matches(boxes, boxes_gt, iou_min=iou_min)
            dic_out['gt'] = [True]
            if verbose:
                print("found {} matches with ground-truth".format(len(matches)))

            # Keep track of instances non-matched
            idxs_matches = [el[0] for el in matches]
            not_matches = [idx for idx, _ in enumerate(boxes) if idx not in idxs_matches]

        else:
            matches = []
            not_matches = list(range(len(boxes)))
            if verbose:
                print("NO ground-truth associated")

        if reorder and matches:
            matches = reorder_matches(matches, boxes, mode='left_right')

        all_idxs = [idx for idx, _ in matches] + not_matches
        dic_out['gt'] = [True]*len(matches) + [False]*len(not_matches)

        uv_shoulders = get_keypoints(keypoints, mode='shoulder')
        uv_heads = get_keypoints(keypoints, mode='head')
        uv_centers = get_keypoints(keypoints, mode='center')
        xy_centers = pixel_to_camera(uv_centers, kk, 1)

        # Add all the predicted annotations, starting with the ones that match a ground-truth
        for idx in all_idxs:
            kps = keypoints[idx]
            box = boxes[idx]
            dd_pred = float(dic_in['d'][idx])
            bi = float(dic_in['bi'][idx])
            var_y = float(dic_in['epi'][idx])
            uu_s, vv_s = uv_shoulders.tolist()[idx][0:2]
            uu_c, vv_c = uv_centers.tolist()[idx][0:2]
            uu_h, vv_h = uv_heads.tolist()[idx][0:2]
            uv_shoulder = [round(uu_s), round(vv_s)]
            uv_center = [round(uu_c), round(vv_c)]
            uv_head = [round(uu_h), round(vv_h)]
            xyz_pred = xyz_from_distance(dd_pred, xy_centers[idx])[0]
            distance = math.sqrt(float(xyz_pred[0])**2 + float(xyz_pred[1])**2 + float(xyz_pred[2])**2)
            conf = 0.035 * (box[-1]) / (bi / distance)

            dic_out['boxes'].append(box)
            dic_out['confs'].append(conf)
            dic_out['dds_pred'].append(dd_pred)
            dic_out['stds_ale'].append(bi)
            dic_out['stds_epi'].append(var_y)

            dic_out['xyz_pred'].append(xyz_pred.squeeze().tolist())
            dic_out['uv_kps'].append(kps)
            dic_out['uv_centers'].append(uv_center)
            dic_out['uv_shoulders'].append(uv_shoulder)
            dic_out['uv_heads'].append(uv_head)

            # For MonStereo / MonoLoco++
            try:
                dic_out['angles'].append(float(dic_in['yaw'][0][idx]))  # Predicted angle
                dic_out['angles_egocentric'].append(float(dic_in['yaw'][1][idx]))  # Egocentric angle
            except KeyError:
                continue

            # Only for MonStereo
            try:
                dic_out['aux'].append(float(dic_in['aux'][idx]))
            except KeyError:
                continue

        for idx, idx_gt in matches:
            dd_real = dds_gt[idx_gt]
            xyz_real = xyz_from_distance(dd_real, xy_centers[idx])
            dic_out['dds_real'].append(dd_real)
            dic_out['boxes_gt'].append(boxes_gt[idx_gt])
            dic_out['xyz_real'].append(xyz_real.squeeze().tolist())
        return dic_out

    @staticmethod
    def social_distance(dic_out, args):

        angles = dic_out['angles']
        dds = dic_out['dds_pred']
        stds = dic_out['stds_ale']
        xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

        # Prepare color for social distancing
        dic_out['social_distance'] = [bool(social_interactions(idx, xz_centers, angles, dds,
                                                               stds=stds,
                                                               threshold_prob=args.threshold_prob,
                                                               threshold_dist=args.threshold_dist,
                                                               radii=args.radii))
                                      for idx, _ in enumerate(dic_out['xyz_pred'])]
        return dic_out


    @staticmethod
    def raising_hand(dic_out, keypoints):
        dic_out['raising_hand'] = [is_raising_hand(keypoint) for keypoint in keypoints]
        return dic_out


def median_disparity(dic_out, keypoints, keypoints_r, mask):
    """
    Ablation study: whenever a matching is found, compute depth by median disparity instead of using MonSter
    Filters are applied to masks nan joints and remove outlier disparities with iqr
    The mask input is used to filter the all-vs-all approach
    """

    keypoints = keypoints.cpu().numpy()
    keypoints_r = keypoints_r.cpu().numpy()
    mask = mask.cpu().numpy()
    avg_disparities, _, _ = mask_joint_disparity(keypoints, keypoints_r)
    BF = 0.54 * 721
    for idx, aux in enumerate(dic_out['aux']):
        if aux > 0.5:
            idx_r = np.argmax(mask[idx])
            z = BF / avg_disparities[idx][idx_r]
            if 1 < z < 80:
                dic_out['xyzd'][idx][2] = z
                dic_out['xyzd'][idx][3] = torch.norm(dic_out['xyzd'][idx][0:3])
    return dic_out
