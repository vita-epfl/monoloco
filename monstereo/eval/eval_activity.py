
import os
import glob
import csv
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

from monstereo.network import Loco
from monstereo.network.process import factory_for_gt, preprocess_pifpaf
from monstereo.activity import social_interactions
from monstereo.utils import open_annotations, get_iou_matches, get_difficulty


class ActivityEvaluator:
    """Evaluate talking activity for Collective Activity Dataset & KITTI"""

    dic_cnt = dict(fp=0, fn=0, det=0)
    cnt = {'pred': defaultdict(int), 'gt': defaultdict(int)}  # pred is for matched instances

    def __init__(self, args):

        # COLLECTIVE ACTIVITY DATASET (talking)
        # -------------------------------------------------------------------------------------------------------------
        if args.dataset == 'collective':
            self.folders_collective = ['seq02', 'seq14', 'seq12', 'seq13', 'seq11', 'seq36']
            # folders_collective = ['seq02']
            self.path_collective = ['data/activity/' + fold for fold in self.folders_collective]
            self.THRESHOLD_PROB = 0.25  # Concordance for samples
            self.THRESHOLD_DIST = 2  # Threshold to check distance of people
            self.RADII = (0.3, 0.5)  # expected radii of the o-space
            self.PIFPAF_CONF = 0.4
            self.SOCIAL_DISTANCE = False
        # -------------------------------------------------------------------------------------------------------------

        # KITTI DATASET (social distancing)
        # ------------------------------------------------------------------------------------------------------------
        else:
            self.dir_ann_kitti = '/data/lorenzo-data/annotations/kitti/scale_2_july'
            self.dir_gt_kitti = 'data/kitti/gt_activity'
            self.dir_kk = os.path.join('data', 'kitti', 'calib')
            self.THRESHOLD_PROB = 0.25  # Concordance for samples
            self.THRESHOLD_DIST = 2  # Threshold to check distance of people
            self.RADII = (0.3, 0.5, 1)  # expected radii of the o-space
            self.PIFPAF_CONF = 0.3
            self.SOCIAL_DISTANCE = True
            # ---------------------------------------------------------------------------------------------------------

        # Load model
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        self.monoloco = Loco(model=args.model, net=args.net,
                             device=device, n_dropout=args.n_dropout, p_dropout=args.dropout)
        self.all_pred = defaultdict(list)
        self.all_gt = defaultdict(list)
        assert args.dataset in ('collective', 'kitti')

    def eval_collective(self):
        """Parse Collective Activity Dataset and predict if people are talking or not"""

        for fold in self.path_collective:
            images = glob.glob(fold + '/*.jpg')
            initial_path = os.path.join(fold, 'frame0001.jpg')
            with open(initial_path, 'rb') as f:
                image = Image.open(f).convert('RGB')
                im_size = image.size

            for idx, im_path in enumerate(images):

                # Collect PifPaf files and calibration
                basename = os.path.basename(im_path)
                extension = '.pifpaf.json'
                path_pif = os.path.join(fold, basename + extension)
                annotations = open_annotations(path_pif)
                kk, _ = factory_for_gt(im_size, verbose=False)

                # Collect corresponding gt files (ys_gt: 1 or 0)
                boxes_gt, ys_gt = parse_gt_collective(fold, path_pif)

                # Run Monoloco
                dic_out, boxes = self.run_monoloco(annotations, kk, im_size=im_size)

                # Match and update stats
                matches = get_iou_matches(boxes, boxes_gt, iou_min=0.3)

                # Estimate activity
                categories = [os.path.basename(fold)] * len(boxes_gt)
                self.estimate_activity(dic_out, matches, ys_gt, categories=categories)

        # Print Results
        cout_results(self.cnt, self.all_gt, self.all_pred, categories=self.folders_collective)

    def eval_kitti(self):
        """Parse KITTI Dataset and predict if people are talking or not"""

        from ..utils import factory_file
        files = glob.glob(self.dir_gt_kitti + '/*.txt')
        # files = [self.dir_gt_kitti + '/001782.txt']
        assert files, "Empty directory"

        for file in files:

            # Collect PifPaf files and calibration
            basename, _ = os.path.splitext(os.path.basename(file))
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann_kitti, basename)

            # Collect corresponding gt files (ys_gt: 1 or 0)
            path_gt = os.path.join(self.dir_gt_kitti, basename + '.txt')
            boxes_gt, ys_gt, difficulties = parse_gt_kitti(path_gt)

            # Run Monoloco
            dic_out, boxes = self.run_monoloco(annotations, kk, im_size=(1242, 374))

            # Match and update stats
            matches = get_iou_matches(boxes, boxes_gt, iou_min=0.3)

            # Estimate activity
            self.estimate_activity(dic_out, matches, ys_gt, categories=difficulties)

            # Print Results
        cout_results(self.cnt, self.all_gt, self.all_pred, categories=('easy', 'moderate', 'hard'))

    def estimate_activity(self, dic_out, matches, ys_gt, categories):

        # Calculate social interaction
        angles = dic_out['angles']
        dds = dic_out['dds_pred']
        stds = dic_out['stds_ale']
        confs = dic_out['confs']
        xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

        # Count gt statistics
        for key in categories:
            self.cnt['gt'][key] += 1
            self.cnt['gt']['all'] += 1

        for i_m, (idx, idx_gt) in enumerate(matches):

            # Select keys to update resultd for Collective or KITTI
            keys = ('all', categories[idx_gt])

            # Run social interactions rule
            flag = social_interactions(idx, xz_centers, angles, dds,
                                       stds=stds,
                                       threshold_prob=self.THRESHOLD_PROB,
                                       threshold_dist=self.THRESHOLD_DIST,
                                       radii=self.RADII,
                                       social_distance=self.SOCIAL_DISTANCE)
            # Accumulate results
            for key in keys:
                self.all_pred[key].append(flag)
                self.all_gt[key].append(ys_gt[idx_gt])
                self.cnt['pred'][key] += 1

    def run_monoloco(self, annotations, kk, im_size=None):

        boxes, keypoints = preprocess_pifpaf(annotations, im_size, enlarge_boxes=True, min_conf=self.PIFPAF_CONF)
        dic_out = self.monoloco.forward(keypoints, kk)
        dic_out = self.monoloco.post_process(dic_out, boxes, keypoints, kk, dic_gt=None, reorder=False, verbose=False)

        return dic_out, boxes


def parse_gt_collective(fold, path_pif):
    """Parse both gt and binary label (1/0) for talking or not"""

    with open(os.path.join(fold, "annotations.txt"), "r") as ff:
        reader = csv.reader(ff, delimiter='\t')
        dic_frames = defaultdict(lambda: defaultdict(list))
        for idx, line in enumerate(reader):
            box = convert_box(line[1:5])
            cat = convert_category(line[5])
            dic_frames[line[0]]['boxes'].append(box)
            dic_frames[line[0]]['y'].append(cat)

    frame = extract_frame_number(path_pif)
    boxes_gt = dic_frames[frame]['boxes']
    ys_gt = np.array(dic_frames[frame]['y'])
    return boxes_gt, ys_gt


def parse_gt_kitti(path_gt):
    """Parse both gt and binary label (1/0) for talking or not"""
    boxes_gt = []
    ys = []
    difficulties = []
    with open(path_gt, "r") as f_gt:
        for line_gt in f_gt:
            line = line_gt.split()
            box = [float(x) for x in line[4:8]]
            boxes_gt.append(box)
            y = int(line[-1])
            assert y in (1, 0), "Expected to be binary (1/0)"
            ys.append(y)
            trunc = float(line[1])
            occ = int(line[2])
            difficulties.append(get_difficulty(box, trunc, occ))
    return boxes_gt, ys, difficulties


def cout_results(cnt, all_gt, all_pred, categories=()):

    categories = list(categories)
    categories.append('all')
    print('-' * 80)

    # Split by folders for collective activity
    for key in categories:
        acc = accuracy_score(all_gt[key], all_pred[key])
        print("Accuracy of category {}: {:.2f}% , Recall: {:.2f}%, #: {}, Predicted positive: {:.2f}%"
              .format(key,
                      acc * 100,
                      cnt['pred'][key] / cnt['gt'][key]*100,
                      cnt['pred'][key],
                      sum(all_gt[key]) / len(all_gt[key]) * 100))

    # Final Accuracy
    acc = accuracy_score(all_gt['all'], all_pred['all'])
    print('-' * 80)
    print("Final Accuracy: {:.2f}%".format(acc * 100))
    print('-' * 80)


def convert_box(box_str):
    """from string with left and center to standard """
    box = [float(el) for el in box_str]  # x, y, w h
    box[2] += box[0]
    box[3] += box[1]
    return box


def convert_category(cat):
    """Talking = 6"""
    if cat == '6':
        return 1
    return 0


def extract_frame_number(path):
    """extract frame number from path"""
    name = os.path.basename(path)
    if name[5] == '0':
        frame = name[6:9]
    else:
        frame = name[5:9]
    return frame
