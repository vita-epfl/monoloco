# pylint: disable=too-many-statements, import-error


"""Extract joints annotations and match with nuScenes ground truths
"""

import os
import sys
import time
import math
import copy
import json
import logging
from collections import defaultdict
import datetime

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

from ..utils import get_iou_matches, append_cluster, select_categories, project_3d, correct_angle, normalize_hwl, \
    to_spherical
from ..network.process import preprocess_pifpaf, preprocess_monoloco


class PreprocessNuscenes:
    """Preprocess Nuscenes dataset"""
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1
    social = False

    CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list)))
              }
    dic_names = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, dir_nuscenes, dataset, iou_min):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.iou_min = iou_min
        self.dir_ann = dir_ann
        dir_out = os.path.join('data', 'arrays')
        assert os.path.exists(dir_nuscenes), "Nuscenes directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-' + dataset + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + dataset + '-' + now_time + '.json')

        self.nusc, self.scenes, self.split_train, self.split_val = factory(dataset, dir_nuscenes)

    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes = cnt_samples = cnt_sd = cnt_ann = 0
        start = time.time()
        for ii, scene in enumerate(self.scenes):
            end_scene = time.time()
            current_token = scene['first_sample_token']
            cnt_scenes += 1
            time_left = str((end_scene - start_scene) / 60 * (len(self.scenes) - ii))[:4] if ii != 0 else "NaN"

            sys.stdout.write('\r' + 'Elaborating scene {}, remaining time {} minutes'
                             .format(cnt_scenes, time_left) + '\t\n')
            start_scene = time.time()
            if scene['name'] in self.split_train:
                phase = 'train'
            elif scene['name'] in self.split_val:
                phase = 'val'
            else:
                print("phase name not in training or validation split")
                continue

            while not current_token == "":
                sample_dic = self.nusc.get('sample', current_token)
                cnt_samples += 1
                # if (cnt_samples % 4 == 0) and (cnt_ann < 3000):
                # Extract all the sample_data tokens for each sample
                for cam in self.CAMERAS:
                    sd_token = sample_dic['data'][cam]
                    cnt_sd += 1

                    # Extract all the annotations of the person
                    path_im, boxes_obj, kk = self.nusc.get_sample_data(sd_token, box_vis_level=1)  # At least one corner
                    boxes_gt, boxes_3d, ys = extract_ground_truth(boxes_obj, kk)
                    kk = kk.tolist()
                    name = os.path.basename(path_im)
                    basename, _ = os.path.splitext(name)

                    self.dic_names[basename + '.jpg']['boxes'] = copy.deepcopy(boxes_gt)
                    self.dic_names[basename + '.jpg']['ys'] = copy.deepcopy(ys)
                    self.dic_names[basename + '.jpg']['K'] = copy.deepcopy(kk)

                    # Run IoU with pifpaf detections and save
                    path_pif = os.path.join(self.dir_ann, name + '.predictions.json')
                    exists = os.path.isfile(path_pif)

                    if exists:
                        with open(path_pif, 'r') as file:
                            annotations = json.load(file)
                            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1600, 900))
                    else:
                        continue
                    if keypoints:
                        matches = get_iou_matches(boxes, boxes_gt, self.iou_min)
                        for (idx, idx_gt) in matches:
                            keypoint = keypoints[idx:idx + 1]
                            inp = preprocess_monoloco(keypoint, kk).view(-1).tolist()
                            lab = ys[idx_gt]
                            lab = normalize_hwl(lab)
                            self.dic_jo[phase]['kps'].append(keypoint)
                            self.dic_jo[phase]['X'].append(inp)
                            self.dic_jo[phase]['Y'].append(lab)
                            self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                            self.dic_jo[phase]['boxes_3d'].append(boxes_3d[idx_gt])
                            append_cluster(self.dic_jo, phase, inp, lab, keypoint)
                            cnt_ann += 1
                            sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')
                current_token = sample_dic['next']

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        # extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} annotations for {} samples in {} scenes. Total time: {:.1f} minutes"
              .format(cnt_ann, cnt_samples, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))


def extract_ground_truth(boxes_obj, kk, spherical=True):

    boxes_gt = []
    boxes_3d = []
    ys = []

    for box_obj in boxes_obj:
        # Select category
        if box_obj.name[:6] != 'animal':
            general_name = box_obj.name.split('.')[0] + '.' + box_obj.name.split('.')[1]
        else:
            general_name = 'animal'
        if general_name in select_categories('all'):

            # Obtain 2D & 3D box
            boxes_gt.append(project_3d(box_obj, kk))
            boxes_3d.append(box_obj.center.tolist() + box_obj.wlh.tolist())

            # Angle
            yaw = quaternion_yaw(box_obj.orientation)
            assert - math.pi <= yaw <= math.pi
            sin, cos, _ = correct_angle(yaw, box_obj.center)
            hwl = [float(box_obj.wlh[i]) for i in (2, 0, 1)]

            # Spherical coordinates
            xyz = list(box_obj.center)
            dd = np.linalg.norm(box_obj.center)
            if spherical:
                rtp = to_spherical(xyz)
                loc = rtp[1:3] + xyz[2:3] + rtp[0:1]  # [theta, psi, z, r]
            else:
                loc = xyz + [dd]

            output = loc + hwl + [sin, cos, yaw]
            ys.append(output)

    return boxes_gt, boxes_3d, ys


def factory(dataset, dir_nuscenes):
    """Define dataset type and split training and validation"""

    assert dataset in ['nuscenes', 'nuscenes_mini', 'nuscenes_teaser']
    if dataset == 'nuscenes_mini':
        version = 'v1.0-mini'
    else:
        version = 'v1.0-trainval'

    nusc = NuScenes(version=version, dataroot=dir_nuscenes, verbose=True)
    scenes = nusc.scene

    if dataset == 'nuscenes_teaser':
        with open("splits/nuscenes_teaser_scenes.txt", "r") as file:
            teaser_scenes = file.read().splitlines()
        scenes = [scene for scene in scenes if scene['token'] in teaser_scenes]
        with open("splits/split_nuscenes_teaser.json", "r") as file:
            dic_split = json.load(file)
        split_train = [scene['name'] for scene in scenes if scene['token'] in dic_split['train']]
        split_val = [scene['name'] for scene in scenes if scene['token'] in dic_split['val']]
    else:
        split_scenes = splits.create_splits_scenes()
        split_train, split_val = split_scenes['train'], split_scenes['val']

    return nusc, scenes, split_train, split_val


def quaternion_yaw(q: Quaternion, in_image_frame: bool = True) -> float:
    if in_image_frame:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
    return float(yaw)


def extract_box_average(boxes_3d):
    boxes_np = np.array(boxes_3d)
    means = np.mean(boxes_np[:, 3:], axis=0)
    stds = np.std(boxes_np[:, 3:], axis=0)
    print(means)
    print(stds)


def extract_social(inputs, ys, keypoints, idx, matches):
    """Output a (padded) version with all the 5 neighbours
    - Take the ground feet and the output z
    - make relative to the person (as social LSTM)"""
    all_inputs = []

    # Find the lowest relative ground foot
    ground_foot = np.max(np.array(inputs)[:, [31, 33]], axis=1)
    rel_ground_foot = ground_foot - ground_foot[idx]
    rel_ground_foot = rel_ground_foot.tolist()

    # Order the people based on their distance
    base = np.array([np.mean(np.array(keypoints[idx][0])), np.mean(np.array(keypoints[idx][1]))])
    # delta_input = [abs((inp[31] + inp[33]) / 2 - base) for inp in inputs]
    delta_input = [np.linalg.norm(base - np.array([np.mean(np.array(kp[0])), np.mean(np.array(kp[1]))]))
                   for kp in keypoints]
    sorted_indices = sorted(range(len(delta_input)), key=lambda k: delta_input[k])  # Return a list of sorted indices
    all_inputs.extend(inputs[idx])

    indices_idx = [idx for (idx, idx_gt) in matches]
    for ii in range(1, 3):
        try:
            index = sorted_indices[ii]

            # Extract the idx_gt corresponding to the input we are attaching if it exists
            try:
                idx_idx_gt = indices_idx.index(index)
                idx_gt = matches[idx_idx_gt][1]
                all_inputs.append(rel_ground_foot[index])  # Relative lower ground foot
                all_inputs.append(float(ys[idx_gt][3]))  # Output Z
            except ValueError:
                all_inputs.extend([0.] * 2)
        except IndexError:
            all_inputs.extend([0.] * 2)
    assert len(all_inputs) == 34 + 2 * 2
    return all_inputs
