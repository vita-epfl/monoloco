
# pylint: disable=too-many-statements, too-many-branches, too-many-nested-blocks

"""Preprocess annotations with KITTI ground-truth"""

import os
import glob
import copy
import logging
from collections import defaultdict
import json
import datetime
from PIL import Image

import torch
import cv2

from ..utils import split_training, parse_ground_truth, get_iou_matches, append_cluster, factory_file, \
    extract_stereo_matches, get_category, normalize_hwl, make_new_directory
from ..network.process import preprocess_pifpaf, preprocess_monoloco
from .transforms import flip_inputs, flip_labels, height_augmentation


class PreprocessKitti:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    # AV_W = 0.68
    # AV_L = 0.75
    # AV_H = 1.72
    # WLH_STD = 0.1

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

    def __init__(self, dir_ann, iou_min, monocular=False):

        self.dir_ann = dir_ann
        self.iou_min = iou_min
        self.monocular = monocular
        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.dir_images = '/data/lorenzo-data/kitti/original_images/training/image_2'
        self.dir_byc_l = '/data/lorenzo-data/kitti/object_detection/left'
        self.names_gt = tuple(os.listdir(self.dir_gt))
        self.dir_kk = os.path.join('data', 'kitti', 'calib')
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

    def run(self):

        cnt_match_l, cnt_match_r, cnt_pair, cnt_pair_tot, cnt_extra_pair, cnt_files, cnt_files_ped, cnt_fnf, \
        cnt_tot, cnt_ambiguous, cnt_cyclist = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        cnt_mono = {'train': 0, 'val': 0, 'test': 0}
        cnt_gt = cnt_mono.copy()
        cnt_stereo = cnt_mono.copy()
        correct_ped, correct_byc, wrong_ped, wrong_byc = 0, 0, 0, 0
        cnt_30, cnt_less_30 = 0, 0

        # self.names_gt = ('002282.txt',)
        for name in self.names_gt:
            path_gt = os.path.join(self.dir_gt, name)
            basename, _ = os.path.splitext(name)
            path_im = os.path.join(self.dir_images, basename + '.png')
            phase, flag = self._factory_phase(name)
            if flag:
                cnt_fnf += 1
                continue

            if phase == 'train':
                min_conf = 0
                category = 'all'
            else:  # Remove for original results
                min_conf = 0.1
                category = 'pedestrian'

            # Extract ground truth
            boxes_gt, ys, _, _ = parse_ground_truth(path_gt, category=category, spherical=True)
            cnt_gt[phase] += len(boxes_gt)
            cnt_files += 1
            cnt_files_ped += min(len(boxes_gt), 1)  # if no boxes 0 else 1

            # Extract keypoints
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)

            self.dic_names[basename + '.png']['boxes'] = copy.deepcopy(boxes_gt)
            self.dic_names[basename + '.png']['ys'] = copy.deepcopy(ys)
            self.dic_names[basename + '.png']['K'] = copy.deepcopy(kk)

            # Check image size
            with Image.open(path_im) as im:
                width, height = im.size

            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(width, height), min_conf=min_conf)

            if keypoints:
                annotations_r, kk_r, tt_r = factory_file(path_calib, self.dir_ann, basename, mode='right')
                boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(width, height), min_conf=min_conf)
                cat = get_category(keypoints, os.path.join(self.dir_byc_l, basename + '.json'))

                if not keypoints_r:  # Case of no detection
                    all_boxes_gt, all_ys = [boxes_gt], [ys]
                    boxes_r, keypoints_r = boxes[0:1].copy(), keypoints[0:1].copy()
                    all_boxes, all_keypoints = [boxes], [keypoints]
                    all_keypoints_r = [keypoints_r]
                else:

                    # Horizontal Flipping for training
                    if phase == 'train':
                        # GT)
                        boxes_gt_flip, ys_flip = flip_labels(boxes_gt, ys, im_w=width)
                        # New left
                        boxes_flip = flip_inputs(boxes_r, im_w=width, mode='box')
                        keypoints_flip = flip_inputs(keypoints_r, im_w=width)

                        # New right
                        keypoints_r_flip = flip_inputs(keypoints, im_w=width)

                        # combine the 2 modes
                        all_boxes_gt = [boxes_gt, boxes_gt_flip]
                        all_ys = [ys, ys_flip]
                        all_boxes = [boxes, boxes_flip]
                        all_keypoints = [keypoints, keypoints_flip]
                        all_keypoints_r = [keypoints_r, keypoints_r_flip]

                    else:
                        all_boxes_gt, all_ys = [boxes_gt], [ys]
                        all_boxes, all_keypoints = [boxes], [keypoints]
                        all_keypoints_r = [keypoints_r]

                # Match each set of keypoint with a ground truth
                self.dic_jo[phase]['K'].append(kk)
                for ii, boxes_gt in enumerate(all_boxes_gt):
                    keypoints, keypoints_r = torch.tensor(all_keypoints[ii]), torch.tensor(all_keypoints_r[ii])
                    ys = all_ys[ii]
                    matches = get_iou_matches(all_boxes[ii], boxes_gt, self.iou_min)
                    for (idx, idx_gt) in matches:
                        keypoint = keypoints[idx:idx + 1]
                        lab = ys[idx_gt][:-1]

                        # Preprocess MonoLoco++
                        if self.monocular:
                            inp = preprocess_monoloco(keypoint, kk).view(-1).tolist()
                            lab = normalize_hwl(lab)
                            if ys[idx_gt][10] < 0.5:
                                self.dic_jo[phase]['kps'].append(keypoint.tolist())
                                self.dic_jo[phase]['X'].append(inp)
                                self.dic_jo[phase]['Y'].append(lab)
                                self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                                append_cluster(self.dic_jo, phase, inp, lab, keypoint)
                                cnt_mono[phase] += 1
                                cnt_tot += 1

                        # Preprocess MonStereo
                        else:
                            zz = ys[idx_gt][2]
                            stereo_matches, cnt_amb = extract_stereo_matches(keypoint, keypoints_r, zz,
                                                                             phase=phase, seed=cnt_pair_tot)
                            cnt_match_l += 1 if ii < 0.1 else 0  # matched instances
                            cnt_match_r += 1 if ii > 0.9 else 0
                            cnt_ambiguous += cnt_amb

                            # Monitor precision of classes
                            if phase == 'val':
                                if ys[idx_gt][10] == cat[idx] == 1:
                                    correct_byc += 1
                                elif ys[idx_gt][10] == cat[idx] == 0:
                                    correct_ped += 1
                                elif ys[idx_gt][10] != cat[idx] and ys[idx_gt][10] == 1:
                                    wrong_byc += 1
                                elif ys[idx_gt][10] != cat[idx] and ys[idx_gt][10] == 0:
                                    wrong_ped += 1

                            cnt_cyclist += 1 if ys[idx_gt][10] == 1 else 0

                            for num, (idx_r, s_match) in enumerate(stereo_matches):
                                label = ys[idx_gt][:-1] + [s_match]
                                if s_match > 0.9:
                                    cnt_pair += 1

                                # Remove noise of very far instances for validation
                                # if (phase == 'val') and (ys[idx_gt][3] >= 50):
                                #     continue

                                #  ---> Save only positives unless there is no positive (keep positive flip and augm)
                                # if num > 0 and s_match < 0.9:
                                #     continue

                                # Height augmentation
                                cnt_pair_tot += 1
                                cnt_extra_pair += 1 if ii == 1 else 0
                                flag_aug = False
                                if phase == 'train' and 3 < label[2] < 30 and s_match > 0.9:
                                    flag_aug = True
                                elif phase == 'train' and 3 < label[2] < 30 and cnt_pair_tot % 2 == 0:
                                    flag_aug = True

                                # Remove height augmentation
                                # flag_aug = False

                                if flag_aug:
                                    kps_aug, labels_aug = height_augmentation(
                                        keypoints[idx:idx+1], keypoints_r[idx_r:idx_r+1], label, s_match,
                                        seed=cnt_pair_tot)
                                else:
                                    kps_aug = [(keypoints[idx:idx+1], keypoints_r[idx_r:idx_r+1])]
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
                                    if ys[idx_gt][10] < 0.5:
                                        self.dic_jo[phase]['kps'].append(keypoint)
                                        self.dic_jo[phase]['X'].append(inp)
                                        self.dic_jo[phase]['Y'].append(lab)
                                        self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                                        append_cluster(self.dic_jo, phase, inp, lab, keypoint)
                                        cnt_tot += 1
                                        if s_match > 0.9:
                                            cnt_stereo[phase] += 1
                                        else:
                                            cnt_mono[phase] += 1

        with open(self.path_joints, 'w') as file:
            json.dump(self.dic_jo, file)
        with open(os.path.join(self.path_names), 'w') as file:
            json.dump(self.dic_names, file)

        # cout
        print(cnt_30)
        print(cnt_less_30)
        print('-' * 120)

        print("Number of GT files: {}. Files with at least one pedestrian: {}.  Files not found: {}"
              .format(cnt_files, cnt_files_ped, cnt_fnf))
        print("Ground truth matches : {:.1f} % for left images (train and val) and {:.1f} % for right images (train)"
              .format(100*cnt_match_l / (cnt_gt['train'] + cnt_gt['val']), 100*cnt_match_r / cnt_gt['train']))
        print("Total annotations: {}".format(cnt_tot))
        print("Total number of cyclists: {}\n".format(cnt_cyclist))
        print("Ambiguous instances removed: {}".format(cnt_ambiguous))
        print("Extra pairs created with horizontal flipping: {}\n".format(cnt_extra_pair))

        if not self.monocular:
            print('Instances with stereo correspondence: {:.1f}% '.format(100 * cnt_pair / cnt_pair_tot))
            for phase in ['train', 'val']:
                cnt = cnt_mono[phase] + cnt_stereo[phase]
                print("{}: annotations: {}. Stereo pairs {:.1f}% "
                      .format(phase.upper(), cnt, 100 * cnt_stereo[phase] / cnt))

        print("\nOutput files:\n{}\n{}".format(self.path_names, self.path_joints))
        print('-' * 120)

    def prep_activity(self):
        """Augment ground-truth with flag activity"""

        from monstereo.activity import social_interactions
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


def crop_and_draw(im, box, keypoint):

    box = [round(el) for el in box[:-1]]
    center = (int((keypoint[0][0])), int((keypoint[1][0])))
    radius = round((box[3]-box[1]) / 20)
    im = cv2.circle(im, center, radius, color=(0, 255, 0), thickness=1)
    crop = im[box[1]:box[3], box[0]:box[2]]
    h_crop = crop.shape[0]
    w_crop = crop.shape[1]

    return crop, h_crop, w_crop
