# pylint: disable=too-many-statements, import-error


"""Extract joints annotations and match with nuScenes ground truths
"""

import os
import sys
import time
import math
import json
import logging
from collections import defaultdict, namedtuple
import datetime

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pyquaternion import Quaternion
import torch

from ..utils import get_iou_matches, append_cluster, select_categories, project_3d, correct_angle, normalize_hwl, \
    to_spherical, average
from ..network.process import preprocess_pifpaf, preprocess_monoloco
from .. import __version__

Annotation = namedtuple('Annotation', 'kps ys kk i_tokens name')
empty_annotations = Annotation([], [], [], [], '')


class PreprocessNuscenes:
    """Preprocess Nuscenes dataset"""
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1
    social = False

    # CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    CAMERAS = ('CAM_FRONT', )
    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list))),
              'version': __version__,
              }
    dic_names = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(int)
    stats = defaultdict(int)

    def __init__(self, dir_ann, dir_nuscenes, dataset, iou_min):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.mode = 'mono'
        self.iou_min = iou_min
        self.dir_ann = dir_ann
        dir_out = os.path.join('data', 'arrays')
        assert os.path.exists(dir_nuscenes), "Nuscenes directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-' + dataset + '-' + self.mode + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + dataset + '-' + now_time + '.json')

        self.nusc, self.scenes, self.split_train, self.split_val = factory(dataset, dir_nuscenes)
        self.nusc_can = NuScenesCanBus(dataroot=dir_nuscenes)

    def run(self):
        """
        Prepare arrays for training
        """
        start = time.time()
        for ii, scene in enumerate(self.scenes):
            end_scene = time.time()
            previous_token = scene['first_sample_token']
            sample_dic = self.nusc.get('sample', previous_token)
            current_token = sample_dic['next']
            i_s = 1  # speed counter
            annotations_p = None
            self.stats['scenes'] += 1
            time_left = str((end_scene - start_scene) / 60 * (len(self.scenes) - ii))[:4] if ii != 0 else "NaN"

            sys.stdout.write('\r' + 'Elaborating scene {}, remaining time {} minutes'
                             .format(self.stats['scenes'], time_left) + '\t\n')
            start_scene = time.time()
            if scene['name'] in self.split_train:
                self.phase = 'train'
            elif scene['name'] in self.split_val:
                self.phase = 'val'
            else:
                print("phase name not in training or validation split")
                continue
            speeds, flag = self.get_vehicle_speed(scene)
            if flag:  # Skip scene for lack of CAN data
                continue

            while not current_token == "" and i_s < 40:
                sample_dic = self.nusc.get('sample', current_token)  # metadata of the sample
                sample_dic_p = self.nusc.get('sample', previous_token)
                speed = [speeds[i_s-1], speeds[i_s]]

                for cam in self.CAMERAS:
                    sd_token = sample_dic['data'][cam]
                    annotations = self.match_annotations(sd_token)
                    sd_token_p = sample_dic_p['data'][cam]
                    if not annotations.kps:
                        continue
                    if annotations_p is None:  # at the beginning
                        annotations_p = self.match_annotations(sd_token_p)
                    kk = annotations.kk
                    name = annotations.name
                    for idx, i_token in enumerate(annotations.i_tokens):
                        kp = annotations.kps[idx]
                        label = annotations.ys[idx]
                        self.stats['ann'] += 1
                        if self.mode == 'mono':
                            inp = preprocess_monoloco(kp, kk).view(-1).tolist()
                            self.dic_jo[self.phase]['kps'].append(kp.tolist())
                            self.dic_jo[self.phase]['X'].append(inp)
                            self.dic_jo[self.phase]['Y'].append(label)
                            self.dic_jo[self.phase]['names'].append(name)  # One image name for each annotation
                            append_cluster(self.dic_jo, self.phase, inp, label, kp.tolist())
                            assert len(inp) == 34
                            s_match = 0
                        else:
                            s_matches = token_matching(i_token, annotations_p.i_tokens)
                            for (idx_r, s_match) in s_matches:
                                kp_r = annotations_p.kps[idx_r]
                                label_s = label + [s_match]  # add flag to distinguish "true pairs and false pairs"
                                input_l = preprocess_monoloco(kp, kk).view(-1)
                                input_r = preprocess_monoloco(kp_r, kk).view(-1)
                                keypoint = torch.cat((kp, kp_r), dim=2).tolist()
                                inp = torch.cat((input_l, input_l - input_r)).tolist()
                                inp.extend(speed)
                                assert len(inp) == 70
                                self.dic_jo[self.phase]['kps'].append(keypoint)
                                self.dic_jo[self.phase]['X'].append(inp)
                                self.dic_jo[self.phase]['Y'].append(label_s)
                                self.dic_jo[self.phase]['names'].append(name)  # One image name for each annotation
                                append_cluster(self.dic_jo, self.phase, inp, label_s, keypoint)

                            self.stats['true_pair'] += 1 if s_match > 0.9 else 0
                            self.stats['pair'] += 1
                            sys.stdout.write('\r' + 'Saved annotations {}'.format(self.stats['ann']) + '\t')

                previous_token = current_token
                current_token = sample_dic['next']
                annotations_p = annotations
                i_s += 1
        print(f"Initial annotations: {self.stats['ann']}")
        print(f"Stereo pairs: {self.stats['true_pair']}")
        print(f"All pairs: {self.stats['pair']}")

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        # extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} pairs for {} annotations in {} scenes. Total time: {:.1f} minutes"
              .format(self.stats['pair'], self.stats['ann'], self.stats['scenes'], (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))

    def match_annotations(self, sd_token):

        kps, inputs, ys, i_tokens = [], [], [], []
        # Extract all the annotations of the person
        path_im, boxes_obj, kk = self.nusc.get_sample_data(sd_token, box_vis_level=1)  # At least one corner
        boxes_gt, boxes_3d, ys, tokens = self.extract_ground_truth(boxes_obj, kk)
        kk = kk.tolist()
        name = os.path.basename(path_im)
        basename, _ = os.path.splitext(name)

        # Run IoU with pifpaf detections and save
        path_pif = os.path.join(self.dir_ann, name + '.predictions.json')
        exists = os.path.isfile(path_pif)
        if exists:
            with open(path_pif, 'r') as file:
                annotations = json.load(file)
                boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1600, 900))
        else:
            return empty_annotations
        if keypoints:
            matches = get_iou_matches(boxes, boxes_gt, self.iou_min)
            kk = torch.tensor(kk)
            for (idx, idx_gt) in matches:
                keypoint = keypoints[idx:idx + 1]
                label = ys[idx_gt]
                label = normalize_hwl(label)
                instance_token = tokens[idx_gt]
                kps.append(torch.tensor(keypoint))
                ys.append(label)
                i_tokens.append(instance_token)

            annotations = Annotation(kps, ys, kk, i_tokens, name)
            return annotations
        else:
            return empty_annotations

    def extract_ground_truth(self, boxes_obj, kk, spherical=True):
        boxes_gt, boxes_3d, ys, i_tokens = [], [], [], []

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
                ann_metadata = self.nusc.get('sample_annotation', box_obj.token)
                i_tokens.append(ann_metadata['instance_token'])
        return boxes_gt, boxes_3d, ys, i_tokens

    def get_vehicle_speed(self, scene):
        flag = False
        try:
            veh_speed = self.nusc_can.get_messages(scene['name'], 'vehicle_monitor')
        except Exception:
            print('no CAN bus data for the scene. Skipping it')
            flag = True
            return None, flag
        veh_speed = np.array([(m['utime'], m['vehicle_speed']) for m in veh_speed])
        veh_speed[:, 1] *= 1 / 3.6
        veh_speed[:, 0] = (veh_speed[:, 0] - veh_speed[0, 0]) / 1e6
        times, speeds = veh_speed[:, 0].tolist(), veh_speed[:, 1].tolist()
        while len(speeds) > 40:
            speeds.pop(-1)
            times.pop(-1)
        while len(times) < 40:
            if times[-1] < 19.3:
                times.append(19.5)
                speeds.append(speeds[-1])
            else:  # Interpolate missing values
                for idx, time in enumerate(times[1:], 1):
                    time_p = times[idx-1]
                    speed = speeds[idx]
                    speed_p = speeds[idx-1]

                    if time-time_p > 1.2:
                        assert time-time_p < 1.6, "case not included"
                        delta_t = (time-time_p) / 3
                        delta_s = (speed - speed_p) / 3
                        time_1 = time_p + delta_t
                        time_2 = time_p + 2*delta_t
                        speed_1 = speed_p + delta_s
                        speed_2 = speed_p + 2*delta_s
                        times.insert(idx, time_2)
                        speeds.insert(idx, speed_2)
                        times.insert(idx, time_1)
                        speeds.insert(idx, speed_1)
                        print(f"Check scene {scene['name']}")
                        break

                    elif time-time_p > 0.6:
                        time_new = average([time_p, time])
                        speed_new = average([speed_p, speed])
                        times.insert(idx, time_new)
                        speeds.insert(idx, speed_new)
                        break
        if len(speeds) != 40:
            aa = 5
        assert len(speeds) == 40
        return speeds, flag


def token_matching(token, tokens_r):
    """match annotations based on their tokens"""
    s_matches = []
    for idx_r, token_r in enumerate(tokens_r):
        if token == token_r:
            s_matches.append((idx_r, 1))
        elif len(s_matches) < 2:
            s_matches.append((idx_r, 0))
    return s_matches


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
