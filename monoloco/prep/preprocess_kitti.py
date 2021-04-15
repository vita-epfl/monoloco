# pylint: disable=too-many-statements, too-many-branches, too-many-nested-blocks

"""Preprocess annotations with KITTI ground-truth"""

import os
import glob
import copy
import math
import logging
from collections import defaultdict
import json
import datetime
from PIL import Image

import torch

from ..utils import split_training, get_iou_matches, append_cluster, get_calibration, open_annotations, \
    extract_stereo_matches, get_category, normalize_hwl, make_new_directory, \
    check_conditions, to_spherical, correct_angle
from ..network.process import preprocess_pifpaf, preprocess_monoloco
from .transforms import flip_inputs, flip_labels, height_augmentation


class PreprocessKitti:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    # KITTI Dataset files
    dir_gt = os.path.join('data', 'kitti', 'gt')
    dir_images = os.path.join('data', 'kitti', 'images')
    dir_kk = os.path.join('data', 'kitti', 'calib')
    dir_byc_l = '/data/lorenzo-data/kitti/object_detection/left'

    # SOCIAL DISTANCING PARAMETERS
    THRESHOLD_DIST = 2  # Threshold to check distance of people
    RADII = (0.3, 0.5, 1)  # expected radii of the o-space
    SOCIAL_DISTANCE = True

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list)))}
    dic_names = defaultdict(lambda: defaultdict(list))
    dic_std = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, mode='mono', iou_min=0.3):

        self.dir_ann = dir_ann
        self.iou_min = iou_min
        self.mode = mode
        assert self.mode in ('mono', 'stereo'), "modality not recognized"
        self.names_gt = tuple(os.listdir(self.dir_gt))
        self.list_gt = glob.glob(self.dir_gt + '/*.txt')
        assert os.path.exists(self.dir_gt), "Ground truth dir does not exist"
        assert os.path.exists(self.dir_ann), "Annotation dir does not exist"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        dir_out = os.path.join('data', 'arrays')
        self.path_joints = os.path.join(dir_out, 'joints-kitti-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-kitti-' + now_time + '.json')
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        self.set_train, self.set_val = split_training(self.names_gt, path_train, path_val)
        self.phase, self.name = None, None
        self.stats_files = defaultdict(int)
        self.stats_gt = defaultdict(int)
        self.stats_mono = defaultdict(int)
        self.stats_stereo = defaultdict(int)

    def run(self):
        # self.names_gt = ('003603.txt',)
        for self.name in self.names_gt:
            # Extract ground truth
            path_gt = os.path.join(self.dir_gt, self.name)
            basename, _ = os.path.splitext(self.name)
            self.phase, file_not_found = self._factory_phase(self.name)
            category = 'all' if self.phase == 'train' else 'pedestrian'
            if file_not_found:
                self.stats_files['fnf'] += 1
                continue

            boxes_gt, labels, _, _ = parse_ground_truth(path_gt, category=category, spherical=True)
            self.stats_gt[self.phase] += len(boxes_gt)
            self.stats_files['files'] += 1
            self.stats_files['files_ped'] += min(len(boxes_gt), 1)  # if no boxes 0 else 1
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
                self.stats_stereo['flipped_pair'] += 1 if ii == 1 else 0
                for (idx, idx_gt) in matches:
                    kp = kps[idx:idx + 1]
                    label = dic_gt['labels'][ii][idx_gt]
                    kk = dic_gt['K']
                    cat = dic_gt['cat'][idx] if self.phase == 'val' else None
                    self._process_location(kp, kps_r, label, kk, cat)

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
        annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)
        boxes, keypoints = preprocess_pifpaf(annotations, im_size=(width, height), min_conf=min_conf)
        cat = get_category(keypoints, os.path.join(self.dir_byc_l, basename + '.json'))
        if not keypoints:
            return None, None, None

        # Stereo-based horizontal flipping for training (obtaining ground truth for right images)
        annotations_r, kk_r, tt_r = factory_file(path_calib, self.dir_ann, basename, ann_type='right')
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
        dic_gt = dict(K=kk, labels=all_labels, cat=cat)
        return dic_boxes, dic_kps, dic_gt

    def _process_location(self, kp, kps_r, label, kk, cat):

        if self.mode == 'mono':
            inp = preprocess_monoloco(kp, kk).view(-1).tolist()
            label = normalize_hwl(label)
            if label[10] < 0.5:
                self.dic_jo[self.phase]['kps'].append(kp.tolist())
                self.dic_jo[self.phase]['X'].append(inp)
                self.dic_jo[self.phase]['Y'].append(label[:-1])
                self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
                append_cluster(self.dic_jo, self.phase, inp, label, kp.tolist())
                self.stats_mono[self.phase] += 1
                self.stats_files['tot'] += 1

        # Preprocess MonStereo
        else:
            zz = label[2]
            stereo_matches, flag_stereo, cnt_amb = extract_stereo_matches(kp, kps_r, zz,
                                                                          phase=self.phase,
                                                                          seed=self.stats_stereo['pair_tot'])
            self.stats_stereo['match_r'] += 1 if flag_stereo else 0
            self.stats_stereo['ambiguous'] += cnt_amb

            # Monitor precision of classes
            if self.phase == 'val':
                if label[10] == cat == 1:
                    self.stats_stereo['correct_byc'] += 1
                elif label[10] == cat == 0:
                    self.stats_stereo['correct_ped'] += 1
                elif label[10] != cat and label[10] == 1:
                    self.stats_stereo['wrong_byc'] += 1
                elif label[10] != cat and label[10] == 0:
                    self.stats_stereo['wrong_ped'] += 1

            self.stats_stereo['cyclists'] += 1 if label[10] == 1 else 0

            for num, (idx_r, s_match) in enumerate(stereo_matches):
                label = label[:-1] + [s_match]
                if s_match > 0.9:
                    self.stats_stereo['pair'] += 1

                # Remove noise of very far instances for validation
                # if (phase == 'val') and (ys[idx_gt][3] >= 50):
                #     continue

                #  ---> Save only positives unless there is no positive (keep positive flip and augm)
                # if num > 0 and s_match < 0.9:
                #     continue

                # Height augmentation
                self.stats_stereo['pair_tot'] += 1
                flag_aug = False
                if self.phase == 'train' and 3 < label[2] < 30 and s_match > 0.9:
                    flag_aug = True
                elif self.phase == 'train' and 3 < label[2] < 30 and self.stats_stereo['pair_tot'] % 2 == 0:
                    flag_aug = True

                # Remove height augmentation
                # flag_aug = False

                if flag_aug:
                    kps_aug, labels_aug = height_augmentation(
                        kp, kps_r[idx_r:idx_r + 1], label, s_match,
                        seed=self.stats_stereo['pair_tot'])
                else:
                    kps_aug = [(kp, kps_r[idx_r:idx_r + 1])]
                    labels_aug = [label]

                for i, lab in enumerate(labels_aug):
                    (kps, kps_r) = kps_aug[i]
                    input_l = preprocess_monoloco(kps, kk).view(-1)
                    input_r = preprocess_monoloco(kps_r, kk).view(-1)
                    keypoint = torch.cat((kps, kps_r), dim=2).tolist()
                    inp = torch.cat((input_l, input_l - input_r)).tolist()

                    # Only relative distances
                    # inp_x = input[::2]
                    # inp = torch.cat((inp_x, input - input_r)).tolist()

                    # lab = normalize_hwl(lab)
                    if label[10] < 0.5:
                        self.dic_jo[self.phase]['kps'].append(keypoint)
                        self.dic_jo[self.phase]['X'].append(inp)
                        self.dic_jo[self.phase]['Y'].append(lab)
                        self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
                        append_cluster(self.dic_jo, self.phase, inp, lab, keypoint)
                        self.stats_stereo['tot'] += 1
                        if s_match > 0.9:
                            self.stats_stereo[self.phase] += 1
                        else:
                            self.stats_mono[self.phase] += 1

    def _cout(self):
        print(
            f"Number of GT files: {self.stats_files['files']}. "
            f"Files with at least one pedestrian: {self.stats_files['files_ped']}.  "
            f"Files not found: {self.stats_files['fnf']}")
        print(
            f"Ground truth matches : "
            f"{100 * self.stats_stereo['match_l'] / (self.stats_gt['train'] + self.stats_gt['train']):.1f} "
            f"% for left images (train and val)")

        print(
            f"Ground truth matches : "
            f"{100 * self.stats_stereo['match_r'] / self.stats_gt['train']:.1f} "
            f"% for right images (train)")

        print(f"Total annotations: {self.stats_files['tot']}")
        print(f"Total number of cyclists: {self.stats_stereo['cyclists']}\n")
        print(f"Ambiguous instances removed: {self.stats_stereo['ambiguous']}")
        print(f"Extra pairs created with horizontal flipping: {self.stats_stereo['flipped_pair']}\n")

        if self.mode == 'stereo':
            print('Instances with stereo correspondence: {:.1f}% '
                  .format(100 * self.stats_stereo['pair'] / self.stats_stereo['pair_tot']))
            for phase in ['train', 'val']:
                cnt = self.stats_mono[self.phase] + self.stats_stereo[self.phase]
                print("{}: annotations: {}. Stereo pairs {:.1f}% "
                      .format(phase.upper(), cnt, 100 * self.stats_stereo[self.phase] / cnt))

        print("\nOutput files:\n{}\n{}".format(self.path_names, self.path_joints))
        print('-' * 120)

    def process_activity(self):
        """Augment ground-truth with flag activity"""

        from monoloco.activity import social_interactions
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
            boxes_gt, ys, truncs_gt, occs_gt, lines = parse_ground_truth(path_gt, category, spherical=False,
                                                                         verbose=True)
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


def parse_ground_truth(path_gt, category, spherical=False, verbose=False):
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
            # cat = 0 if line[0] in ('Pedestrian', 'Person_sitting') else 1
            if line[0] in ('Pedestrian', 'Person_sitting'):
                cat = 0
            else:
                cat = 1
            output = loc + hwl + [sin, cos, yaw, cat]
            labels.append(output)
            if verbose:
                lines.append(line_gt)
    if verbose:
        return boxes_gt, labels, truncs_gt, occs_gt, lines
    return boxes_gt, labels, truncs_gt, occs_gt


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