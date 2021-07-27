# pylint: disable=too-many-statements, too-many-branches, too-many-nested-blocks

"""Preprocess annotations with wv ground-truth"""

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
from ..utils import split_training, get_iou_matches, append_cluster, open_annotations, \
    check_conditions, to_spherical, correct_angle
from ..network.process import preprocess_pifpaf, preprocess_monoloco, load_calibration
from .preprocess_kitti import parse_ground_truth


class Preprocess:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""
    width = 1920
    height = 1200
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ext = '.jpeg'
    load = True
    joints_path = 'data/arrays/joints-nuscenes_teaser-210727-1137_copy.json'
    if not load:
        dic_jo = {
            'train': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
            'val': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
            'test': dict(X=[], Y=[], names=[], kps=[], K=[], clst=defaultdict(lambda: defaultdict(list))),
            'version': __version__,
            }
    else:
        # dic_names = defaultdict(lambda: defaultdict(list))
        with open(joints_path, 'r') as ff:
            dic_jo = json.load(ff)
    categories_gt = dict(
        train=['pedestrian', 'person_on_bike', 'person_on_motorcycle'],
        val=['pedestrian', 'person_on_bike', 'person_on_motorcycle'])

    def __init__(self, args):

        assert args.dir_ann is not None
        self.dir_ann = args.dir_ann
        self.iou_min = args.iou_min
        assert args.dir_gt is not None
        self.dir_gt = args.dir_gt
        assert args.dir_images is not None or self.width is not None
        self.dir_images = args.dir_images
        assert args.calibration is not None
        self.calibration = args.calibration

        assert os.path.isdir(self.dir_ann), "Annotation directory not found"
        assert any(os.scandir(self.dir_ann)), "Annotation directory empty"

        self.names_gt = tuple(os.listdir(self.dir_gt))
        self.list_gt = glob.glob(self.dir_gt + '/*.txt')
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        dir_out = os.path.join('data', 'arrays')
        self.path_joints = os.path.join(dir_out, 'joints-' + args.calibration + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + args.calibration + '-' + now_time + '.json')

        path_train = os.path.join('data', 'wv', 'train.txt')
        path_val = os.path.join('data', 'wv', 'val.txt')
        self.set_train, self.set_val = split_training(self.names_gt, path_train, path_val)

        self.kk = load_calibration(args.calibration)
        self.phase, self.name = None, None
        self.stats = defaultdict(int)

    def run(self):
        # self.names_gt = ('002282.txt',)
        for self.name in self.names_gt:
            # Extract ground truth
            path_gt = os.path.join(self.dir_gt, self.name)
            basename, _ = os.path.splitext(self.name)
            self.phase, file_not_found = self._factory_phase(self.name)
            category = 'all'
            if file_not_found:
                self.stats['fnf'] += 1
                continue

            boxes_gt, labels, _, _, _ = parse_ground_truth(path_gt, category=category, spherical=True)
            self.stats['gt_' + self.phase] += len(boxes_gt)
            self.stats['gt_files'] += 1
            self.stats['gt_files_ped'] += min(len(boxes_gt), 1)  # if no boxes 0 else 1
            # self.dic_names[basename + self.ext]['boxes'] = copy.deepcopy(boxes_gt)
            # self.dic_names[basename + self.ext]['ys'] = copy.deepcopy(labels)

            # Extract annotations
            dic_boxes, dic_kps, dic_gt = self.parse_annotations(boxes_gt, labels, basename)
            if dic_boxes is None:  # No annotations
                continue
            # self.dic_names[basename + self.ext]['K'] = copy.deepcopy(dic_gt['K'])
            self.dic_jo[self.phase]['K'].append(dic_gt['K'])

            # Match each set of keypoint with a ground truth
            kps = torch.tensor(dic_kps['left'])
            matches = get_iou_matches(dic_boxes['left'], boxes_gt, self.iou_min)
            for (idx, idx_gt) in matches:
                cat_gt = dic_gt['labels'][idx_gt][-1]
                if cat_gt not in self.categories_gt[self.phase]:  # only for training as cyclists are also extracted
                    continue
                kp = kps[idx:idx + 1]
                kk = dic_gt['K']
                label = dic_gt['labels'][idx_gt][:-1]
                self.stats['match'] += 1
                assert len(label) == 11, 'dimensions of monocular label is wrong'
                self._process_annotation_mono(kp, kk, label)

        with open(self.path_joints, 'w') as file:
            json.dump(self.dic_jo, file)
        # with open(os.path.join(self.path_names), 'w') as file:
        #     json.dump(self.dic_names, file)
        self._cout()

    def parse_annotations(self, boxes_gt, labels, basename):

        min_conf = 0 if self.phase == 'train' else 0.1
        if self.dir_images is None:
            width = self.width
            height = self.height
        else:
            # Check image size
            path_im = os.path.join(self.dir_images, basename + self.ext)
            with Image.open(path_im) as im:
                width, height = im.size
                print(im.size)

        # Extractkeypoints
        path_ann = os.path.join(self.dir_ann, basename + self.ext + '.predictions.json')
        annotations = open_annotations(path_ann)
        boxes, keypoints = preprocess_pifpaf(annotations, im_size=(width, height), min_conf=min_conf)
        if not keypoints:
            return None, None, None

        dic_boxes = dict(left=boxes, gt=boxes_gt)
        dic_kps = dict(left=keypoints)
        dic_gt = dict(K=self.kk, labels=labels)
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

        total_train = self.stats['total_train']
        total_val = self.stats['total_val']
        print(f"Total annotations for TRAINING: {total_train}")
        print(f"Total annotations for VALIDATION: {total_val}")
        print('-' * 100)
        print(f"\nOutput files:\n{self.path_names}\n{self.path_joints}")
        print('-' * 100)

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
