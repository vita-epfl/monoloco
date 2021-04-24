# pylint: disable=too-many-statements, too-many-branches, too-many-nested-blocks

"""Preprocess annotations with KITTI ground-truth"""

import os
import glob
import copy
import math
import logging
from collections import defaultdict
import json
import warnings
import datetime
from PIL import Image

import torch

from .. import __version__
from ..utils import split_training, get_iou_matches, append_cluster, get_calibration, open_annotations, \
    extract_stereo_matches, make_new_directory, \
    check_conditions, to_spherical, correct_angle
from ..network.process import preprocess_pifpaf, preprocess_monoloco
from .transforms import flip_inputs, flip_labels, height_augmentation


class PreprocessKitti:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    # KITTI Dataset files
    dir_gt = os.path.join('data', 'kitti', 'gt')
    dir_images = os.path.join('data', 'kitti', 'images')
    dir_kk = os.path.join('data', 'kitti', 'calib')

    # SOCIAL DISTANCING PARAMETERS
    THRESHOLD_DIST = 2  # Threshold to check distance of people
    RADII = (0.3, 0.5, 1)  # expected radii of the o-space
    SOCIAL_DISTANCE = True

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dic_jo = {
        'train': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
        'val': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
        'test': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
        'version': __version__,
    }
    dic_names = defaultdict(lambda: defaultdict(list))
    dic_std = defaultdict(lambda: defaultdict(list))
    categories_gt = dict(train=['Pedestrian', 'Person_sitting'], val=['Pedestrian'])

    def __init__(self, dir_ann, mode='mono', iou_min=0.3, sample=False):

        self.dir_ann = dir_ann
        self.mode = mode
        self.iou_min = iou_min
        self.sample = sample

        assert os.path.isdir(self.dir_ann), "Annotation directory not found"
        assert any(os.scandir(self.dir_ann)), "Annotation directory empty"
        assert os.path.isdir(self.dir_gt), "Ground truth directory not found"
        assert any(os.scandir(self.dir_gt)), "Ground-truth directory empty"
        if self.mode == 'stereo':
            assert os.path.isdir(self.dir_ann + '_right'), "Annotation directory for right images not found"
            assert any(os.scandir(self.dir_ann + '_right')), "Annotation directory for right images empty"
        elif not os.path.isdir(self.dir_ann + '_right') or not any(os.scandir(self.dir_ann + '_right')):
            warnings.warn('Horizontal flipping not applied as annotation directory for right images not found/empty')
        assert self.mode in ('mono', 'stereo'), "modality not recognized"

        self.names_gt = tuple(os.listdir(self.dir_gt))
        self.list_gt = glob.glob(self.dir_gt + '/*.txt')
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        dir_out = os.path.join('data', 'arrays')
        self.path_joints = os.path.join(dir_out, 'joints-kitti-' + self.mode + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-kitti-' + self.mode + '-' + now_time + '.json')
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        self.set_train, self.set_val = split_training(self.names_gt, path_train, path_val)
        self.phase, self.name = None, None
        self.stats = defaultdict(int)
        self.stats_stereo = defaultdict(int)

    def run(self):
        # self.names_gt = ('002282.txt',)
        for self.name in self.names_gt:
            # Extract ground truth
            path_gt = os.path.join(self.dir_gt, self.name)
            basename, _ = os.path.splitext(self.name)
            self.phase, file_not_found = self._factory_phase(self.name)
            category = 'all' if self.phase == 'train' else 'pedestrian'
            if file_not_found:
                self.stats['fnf'] += 1
                continue

            boxes_gt, labels, _, _, _ = parse_ground_truth(path_gt, category=category, spherical=True)
            self.stats['gt_' + self.phase] += len(boxes_gt)
            self.stats['gt_files'] += 1
            self.stats['gt_files_ped'] += min(len(boxes_gt), 1)  # if no boxes 0 else 1
            self.dic_names[basename + '.png']['boxes'] = copy.deepcopy(boxes_gt)
            self.dic_names[basename + '.png']['ys'] = copy.deepcopy(labels)

            # Extract annotations
            dic_boxes, dic_kps, dic_gt = self.parse_annotations(boxes_gt, labels, basename)
            if dic_boxes is None:  # No annotations
                continue
            self.dic_names[basename + '.png']['K'] = copy.deepcopy(dic_gt['K'])
            self.dic_jo[self.phase]['K'].append(dic_gt['K'])

            # Match each set of keypoint with a ground truth
            for ii, boxes_gt in enumerate(dic_boxes['gt']):
                kps, kps_r = torch.tensor(dic_kps['left'][ii]), torch.tensor(dic_kps['right'][ii])
                matches = get_iou_matches(dic_boxes['left'][ii], boxes_gt, self.iou_min)
                self.stats['flipping_match'] += len(matches) if ii == 1 else 0
                for (idx, idx_gt) in matches:
                    cat_gt = dic_gt['labels'][ii][idx_gt][-1]
                    if cat_gt not in self.categories_gt[self.phase]: # only for training as cyclists are also extracted
                        continue
                    kp = kps[idx:idx + 1]
                    kk = dic_gt['K']
                    label = dic_gt['labels'][ii][idx_gt][:-1]
                    self.stats['match'] += 1
                    assert len(label) == 10, 'dimensions of monocular label is wrong'

                    if self.mode == 'mono':
                        self._process_annotation_mono(kp, kk, label)
                    else:
                        self._process_annotation_stereo(kp, kk, label, kps_r)

        with open(self.path_joints, 'w') as file:
            json.dump(self.dic_jo, file)
        with open(os.path.join(self.path_names), 'w') as file:
            json.dump(self.dic_names, file)
        self._cout()

    def parse_annotations(self, boxes_gt, labels, basename):

        path_im = os.path.join(self.dir_images, basename + '.png')
        path_calib = os.path.join(self.dir_kk, basename + '.txt')
        min_conf = 0 if self.phase == 'train' else 0.1

        # Check image size
        with Image.open(path_im) as im:
            width, height = im.size

        # Extract left keypoints
        annotations, kk, _ = factory_file(path_calib, self.dir_ann, basename)
        boxes, keypoints = preprocess_pifpaf(annotations, im_size=(width, height), min_conf=min_conf)
        if not keypoints:
            return None, None, None

        # Stereo-based horizontal flipping for training (obtaining ground truth for right images)
        self.stats['instances'] += len(keypoints)
        annotations_r, _, _ = factory_file(path_calib, self.dir_ann, basename, ann_type='right')
        boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(width, height), min_conf=min_conf)

        if not keypoints_r:  # Duplicate the left one(s)
            all_boxes_gt, all_labels = [boxes_gt], [labels]
            boxes_r, keypoints_r = boxes[0:1].copy(), keypoints[0:1].copy()
            all_boxes, all_keypoints = [boxes], [keypoints]
            all_keypoints_r = [keypoints_r]

        elif self.phase == 'train':
            # GT)
            boxes_gt_flip, ys_flip = flip_labels(boxes_gt, labels, im_w=width)
            # New left
            boxes_flip = flip_inputs(boxes_r, im_w=width, mode='box')
            keypoints_flip = flip_inputs(keypoints_r, im_w=width)

            # New right
            keypoints_r_flip = flip_inputs(keypoints, im_w=width)

            # combine the 2 modes
            all_boxes_gt = [boxes_gt, boxes_gt_flip]
            all_labels = [labels, ys_flip]
            all_boxes = [boxes, boxes_flip]
            all_keypoints = [keypoints, keypoints_flip]
            all_keypoints_r = [keypoints_r, keypoints_r_flip]

        else:
            all_boxes_gt, all_labels = [boxes_gt], [labels]
            all_boxes, all_keypoints = [boxes], [keypoints]
            all_keypoints_r = [keypoints_r]

        dic_boxes = dict(left=all_boxes, gt=all_boxes_gt)
        dic_kps = dict(left=all_keypoints, right=all_keypoints_r)
        dic_gt = dict(K=kk, labels=all_labels)
        return dic_boxes, dic_kps, dic_gt

    def _process_annotation_mono(self, kp, kk, label):
        """For a single annotation, process all the labels and save them"""
        kp = kp.tolist()
        inp = preprocess_monoloco(kp, kk).view(-1).tolist()

        # Save
        self.dic_jo[self.phase]['kps'].append(kp)
        self.dic_jo[self.phase]['X'].append(inp)
        self.dic_jo[self.phase]['Y'].append(label)
        self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
        append_cluster(self.dic_jo, self.phase, inp, label, kp)
        self.stats['total_' + self.phase] += 1

    def _process_annotation_stereo(self, kp, kk, label, kps_r):
        """For a reference annotation, combine it with some (right) annotations and save it"""

        zz = label[2]
        stereo_matches, cnt_amb = extract_stereo_matches(kp, kps_r, zz,
                                                         phase=self.phase,
                                                         seed=self.stats_stereo['pair'])
        self.stats_stereo['ambiguous'] += cnt_amb

        for idx_r, s_match in stereo_matches:
            label_s = label + [s_match]  # add flag to distinguish "true pairs and false pairs"
            self.stats_stereo['true_pair'] += 1 if s_match > 0.9 else 0
            self.stats_stereo['pair'] += 1  # before augmentation

            # ---> Remove noise of very far instances for validation
            # if (self.phase == 'val') and (label[3] >= 50):
            #     continue

            #  ---> Save only positives unless there is no positive (keep positive flip and augm)
            # if num > 0 and s_match < 0.9:
            #     continue

            # Height augmentation
            flag_aug = False
            if self.phase == 'train' and 3 < label[2] < 30 and (s_match > 0.9 or self.stats_stereo['pair'] % 2 == 0):
                flag_aug = True

            # Remove height augmentation
            # flag_aug = False

            if flag_aug:
                kps_aug, labels_aug = height_augmentation(kp, kps_r[idx_r:idx_r + 1], label_s,
                                                          seed=self.stats_stereo['pair'])
            else:
                kps_aug = [(kp, kps_r[idx_r:idx_r + 1])]
                labels_aug = [label_s]

            for i, lab in enumerate(labels_aug):
                assert len(lab) == 11, 'dimensions of stereo label is wrong'
                self.stats_stereo['pair_aug'] += 1
                (kp_aug, kp_aug_r) = kps_aug[i]
                input_l = preprocess_monoloco(kp_aug, kk).view(-1)
                input_r = preprocess_monoloco(kp_aug_r, kk).view(-1)
                keypoint = torch.cat((kp_aug, kp_aug_r), dim=2).tolist()
                inp = torch.cat((input_l, input_l - input_r)).tolist()
                self.dic_jo[self.phase]['kps'].append(keypoint)
                self.dic_jo[self.phase]['X'].append(inp)
                self.dic_jo[self.phase]['Y'].append(lab)
                self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
                append_cluster(self.dic_jo, self.phase, inp, lab, keypoint)
                self.stats_stereo['total_' + self.phase] += 1  # including height augmentation

    def _cout(self):
        print('-' * 100)
        print(f"Number of GT files: {self.stats['gt_files']} ")
        print(f"Files with at least one pedestrian/cyclist: {self.stats['gt_files_ped']}")
        print(f"Files not found: {self.stats['fnf']}")
        print('-' * 100)
        our = self.stats['match'] - self.stats['flipping_match']
        gt = self.stats['gt_train'] + self.stats['gt_val']
        print(f"Ground truth matches: {100 * our  / gt:.1f} for left images (train and val)")
        print(f"Parsed instances: {self.stats['instances']}")
        print(f"Ground truth instances: {gt}")
        print(f"Matched instances: {our}")
        print(f"Including horizontal flipping: {self.stats['match']}")

        if self.mode == 'stereo':
            print('-' * 100)
            print(f"Ambiguous instances removed: {self.stats_stereo['ambiguous']}")
            print(f"True pairs ratio: {100 * self.stats_stereo['true_pair'] / self.stats_stereo['pair']:.1f}% ")
            print(f"Height augmentation pairs: {self.stats_stereo['pair_aug'] - self.stats_stereo['pair']} ")
            print('-' * 100)
        total_train = self.stats_stereo['total_train'] if self.mode == 'stereo' else self.stats['total_train']
        total_val = self.stats_stereo['total_val'] if self.mode == 'stereo' else self.stats['total_val']
        print(f"Total annotations for TRAINING: {total_train}")
        print(f"Total annotations for VALIDATION: {total_val}")
        print('-' * 100)
        print(f"\nOutput files:\n{self.path_names}\n{self.path_joints}")
        print('-' * 100)

    def process_activity(self):
        """Augment ground-truth with flag activity"""

        from monoloco.activity import social_interactions  # pylint: disable=import-outside-toplevel
        main_dir = os.path.join('data', 'kitti')
        dir_gt = os.path.join(main_dir, 'gt')
        dir_out = os.path.join(main_dir, 'gt_activity')
        make_new_directory(dir_out)
        cnt_tp, cnt_tn = 0, 0

        # Extract validation images for evaluation
        category = 'pedestrian'

        for name in self.set_val:
            # Read
            path_gt = os.path.join(dir_gt, name)
            _, ys, _, _, lines = parse_ground_truth(path_gt, category, spherical=False)
            angles = [y[10] for y in ys]
            dds = [y[4] for y in ys]
            xz_centers = [[y[0], y[2]] for y in ys]

            # Write
            path_out = os.path.join(dir_out, name)
            with open(path_out, "w+") as ff:
                for idx, line in enumerate(lines):
                    if social_interactions(idx, xz_centers, angles, dds,
                                           n_samples=1,
                                           threshold_dist=self.THRESHOLD_DIST,
                                           radii=self.RADII,
                                           social_distance=self.SOCIAL_DISTANCE):
                        activity = '1'
                        cnt_tp += 1
                    else:
                        activity = '0'
                        cnt_tn += 1

                    line_new = line[:-1] + ' ' + activity + line[-1]
                    ff.write(line_new)

        print(f'Written {len(self.set_val)} new files in {dir_out}')
        print(f'Saved {cnt_tp} positive and {cnt_tn} negative annotations')

    def _factory_phase(self, name):
        """Choose the phase"""
        phase = None
        flag = False
        if name in self.set_train:
            phase = 'train'
        elif name in self.set_val:
            phase = 'val'
        else:
            flag = True
        return phase, flag


def parse_ground_truth(path_gt, category, spherical=False):
    """Parse KITTI ground truth files"""

    boxes_gt = []
    labels = []
    truncs_gt = []  # Float from 0 to 1
    occs_gt = []  # Either 0,1,2,3 fully visible, partly occluded, largely occluded, unknown
    lines = []

    with open(path_gt, "r") as f_gt:
        for line_gt in f_gt:
            line = line_gt.split()
            if not check_conditions(line_gt, category, method='gt'):
                continue
            truncs_gt.append(float(line[1]))
            occs_gt.append(int(line[2]))
            boxes_gt.append([float(x) for x in line[4:8]])
            xyz = [float(x) for x in line[11:14]]
            hwl = [float(x) for x in line[8:11]]
            dd = float(math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))
            yaw = float(line[14])
            assert - math.pi <= yaw <= math.pi
            alpha = float(line[3])
            sin, cos, yaw_corr = correct_angle(yaw, xyz)
            assert min(abs(-yaw_corr - alpha), (abs(yaw_corr - alpha))) < 0.15, "more than 10 degrees of error"
            if spherical:
                rtp = to_spherical(xyz)
                loc = rtp[1:3] + xyz[2:3] + rtp[0:1]  # [theta, psi, z, r]
            else:
                loc = xyz + [dd]
            cat = line[0]  # 'Pedestrian', or 'Person_sitting' for people
            output = loc + hwl + [sin, cos, yaw, cat]
            labels.append(output)
            lines.append(line_gt)
    return boxes_gt, labels, truncs_gt, occs_gt, lines


def factory_file(path_calib, dir_ann, basename, ann_type='left'):
    """Choose the annotation and the calibration files"""

    assert ann_type in ('left', 'right')
    p_left, p_right = get_calibration(path_calib)

    if ann_type == 'left':
        kk, tt = p_left[:]
        path_ann = os.path.join(dir_ann, basename + '.png.predictions.json')

    # The right folder is called <NameOfLeftFolder>_right
    else:
        kk, tt = p_right[:]
        path_ann = os.path.join(dir_ann + '_right', basename + '.png.predictions.json')

    annotations = open_annotations(path_ann)

    return annotations, kk, tt
