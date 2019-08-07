"""Evaluate Monoloco code on KITTI dataset using ALE and ALP metrics with the following baselines:
    - Mono3D
    - 3DOP
    - MonoDepth
    """

import os
import math
import logging
import datetime
from collections import defaultdict
from itertools import chain

from tabulate import tabulate

from ..utils import get_iou_matches, get_task_error, get_pixel_error, check_conditions, get_category, split_training, \
    parse_ground_truth
from ..visuals import show_results, show_spread


class EvalKitti:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    CLUSTERS = ('easy', 'moderate', 'hard', 'all', '6', '10', '15', '20', '25', '30', '40', '50', '>50')
    METHODS = ['m3d', 'md', 'geom', 'task_error', '3dop', 'our']
    HEADERS = ['method', '<0.5', '<1m', '<2m', 'easy', 'moderate', 'hard', 'all']
    CATEGORIES = ['pedestrian']

    def __init__(self, thresh_iou_our=0.3, thresh_iou_m3d=0.3, thresh_conf_m3d=0.3, thresh_conf_our=0.3,
                 verbose=False, stereo=False):

        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.dir_m3d = os.path.join('data', 'kitti', 'm3d')
        self.dir_3dop = os.path.join('data', 'kitti', '3dop')
        self.dir_md = os.path.join('data', 'kitti', 'monodepth')
        self.dir_our = os.path.join('data', 'kitti', 'monoloco')
        self.stereo = stereo
        if self.stereo:
            self.dir_our_stereo = os.path.join('data', 'kitti', 'monoloco_stereo')
            self.METHODS.extend(['our_stereo', 'pixel_error'])
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        dir_logs = os.path.join('data', 'logs')
        assert dir_logs, "No directory to save final statistics"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_results = os.path.join(dir_logs, 'eval-' + now_time + '.json')
        self.verbose = verbose

        assert os.path.exists(self.dir_m3d) and os.path.exists(self.dir_our) \
               and os.path.exists(self.dir_3dop)

        self.dic_thresh_iou = {'m3d': thresh_iou_m3d, '3dop': thresh_iou_m3d,
                               'md': thresh_iou_our, 'our': thresh_iou_our, 'our_stereo': thresh_iou_our}
        self.dic_thresh_conf = {'m3d': thresh_conf_m3d, '3dop': thresh_conf_m3d,
                                'our': thresh_conf_our, 'our_stereo': thresh_conf_our}

        # Extract validation images for evaluation
        names_gt = tuple(os.listdir(self.dir_gt))
        _, self.set_val = split_training(names_gt, path_train, path_val)

        # Define variables to save statistics
        self.errors = None
        self.dic_stds = None
        self.dic_stats = None
        self.dic_cnt = None
        self.cnt_stereo_error = None
        self.cnt_gt = 0

    def run(self):
        """Evaluate Monoloco performances on ALP and ALE metrics"""

        for category in self.CATEGORIES:

            # Initialize variables
            self.errors = defaultdict(lambda: defaultdict(list))
            self.dic_stds = defaultdict(lambda: defaultdict(list))
            self.dic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
            self.dic_cnt = defaultdict(int)
            self.cnt_gt = 0
            self.cnt_stereo_error = 0

            # Iterate over each ground truth file in the training set
            for name in self.set_val:
                path_gt = os.path.join(self.dir_gt, name)
                path_m3d = os.path.join(self.dir_m3d, name)
                path_our = os.path.join(self.dir_our, name)
                if self.stereo:
                    path_our_stereo = os.path.join(self.dir_our_stereo, name)
                path_3dop = os.path.join(self.dir_3dop, name)
                path_md = os.path.join(self.dir_md, name)

                # Iterate over each line of the gt file and save box location and distances
                out_gt = parse_ground_truth(path_gt, category)
                self.cnt_gt += len(out_gt[0])

                # Extract annotations for the same file
                if out_gt[0]:
                    out_m3d = self._parse_txts(path_m3d, category, method='m3d')
                    out_3dop = self._parse_txts(path_3dop, category, method='3dop')
                    out_md = self._parse_txts(path_md, category, method='md')
                    out_our = self._parse_txts(path_our, category, method='our')
                    out_our_stereo = self._parse_txts(path_our_stereo, category, method='our') if self.stereo else []

                    # Compute the error with ground truth
                    self._estimate_error(out_gt, out_m3d, method='m3d')
                    self._estimate_error(out_gt, out_3dop, method='3dop')
                    self._estimate_error(out_gt, out_md, method='md')
                    self._estimate_error(out_gt, out_our, method='our')
                    if self.stereo:
                        self._estimate_error(out_gt, out_our_stereo, method='our_stereo')

                    # Iterate over all the files together to find a pool of common annotations
                    self._compare_error(out_gt, out_m3d, out_3dop, out_md, out_our, out_our_stereo)

            # Update statistics of errors and uncertainty
            for key in self.errors:
                add_true_negatives(self.errors[key], self.cnt_gt)
                for clst in self.CLUSTERS[:-2]:  # M3d and pifpaf does not have annotations above 40 meters
                    get_statistics(self.dic_stats['test'][key][clst], self.errors[key][clst], self.dic_stds[clst], key)

            # Show statistics
            print('\n' + category.upper() + ':')
            self.show_statistics()

    def printer(self, show, save):
        if save or show:
            show_results(self.dic_stats, show, save)
            show_spread(self.dic_stats, show, save)

    def _parse_txts(self, path, category, method):
        boxes = []
        dds = []
        stds_ale = []
        stds_epi = []
        dds_geom = []

        # Iterate over each line of the txt file
        if method in ['3dop', 'm3d']:
            try:
                with open(path, "r") as ff:
                    for line in ff:
                        if check_conditions(line, category, method=method, thresh=self.dic_thresh_conf[method]):
                            boxes.append([float(x) for x in line.split()[4:8]])
                            loc = ([float(x) for x in line.split()[11:14]])
                            dds.append(math.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2))
                            self.dic_cnt[method] += 1
                return boxes, dds

            except FileNotFoundError:
                return [], []

        elif method == 'md':
            try:
                with open(path, "r") as ff:
                    for line in ff:
                        box = [float(x[:-1]) for x in line.split()[0:4]]
                        delta_h = (box[3] - box[1]) / 7
                        delta_w = (box[2] - box[0]) / 3.5
                        assert delta_h > 0 and delta_w > 0, "Bounding box <=0"
                        box[0] -= delta_w
                        box[1] -= delta_h
                        box[2] += delta_w
                        box[3] += delta_h
                        boxes.append(box)
                        dds.append(float(line.split()[5][:-1]))
                        self.dic_cnt[method] += 1
                return boxes, dds

            except FileNotFoundError:
                return [], []

        else:
            assert method == 'our', "method not recognized"
            try:
                with open(path, "r") as ff:
                    file_lines = ff.readlines()
                for line_our in file_lines[:-1]:
                    line_list = [float(x) for x in line_our.split()]

                    if check_conditions(line_list, category, method=method, thresh=self.dic_thresh_conf[method]):
                        boxes.append(line_list[:4])
                        dds.append(line_list[8])
                        stds_ale.append(line_list[9])
                        stds_epi.append(line_list[10])
                        dds_geom.append(line_list[11])
                        self.dic_cnt[method] += 1
                        self.dic_cnt['geom'] += 1

                # kk_list = [float(x) for x in file_lines[-1].split()]

                return boxes, dds, stds_ale, stds_epi, dds_geom

            except FileNotFoundError:
                return [], [], [], [], []

    def _estimate_error(self, out_gt, out, method):
        """Estimate localization error"""

        boxes_gt, _, dds_gt, zzs_gt, truncs_gt, occs_gt = out_gt
        if method[:3] == 'our':
            boxes, dds, stds_ale, stds_epi, dds_geom = out
        else:
            boxes, dds = out

        matches = get_iou_matches(boxes, boxes_gt, self.dic_thresh_iou[method])

        for (idx, idx_gt) in matches:
            # Update error if match is found
            cat = get_category(boxes_gt[idx_gt], truncs_gt[idx_gt], occs_gt[idx_gt])
            self.update_errors(dds[idx], dds_gt[idx_gt], cat, self.errors[method])

            if method == 'our':
                self.update_errors(dds_geom[idx], dds_gt[idx_gt], cat, self.errors['geom'])
                self.update_uncertainty(stds_ale[idx], stds_epi[idx], dds[idx], dds_gt[idx_gt], cat)
                dd_task_error = dds_gt[idx_gt] + (get_task_error(dds_gt[idx_gt]))**2
                self.update_errors(dd_task_error, dds_gt[idx_gt], cat, self.errors['task_error'])

            elif method == 'our_stereo':
                dd_pixel_error = get_pixel_error(dds_gt[idx_gt], zzs_gt[idx_gt])
                self.update_errors(dd_pixel_error, dds_gt[idx_gt], cat, self.errors['pixel_error'])

    def _compare_error(self, out_gt, out_m3d, out_3dop, out_md, out_our, out_our_stereo):
        """Compare the error for a pool of instances commonly matched by all methods"""

        # Extract outputs of each method
        boxes_gt, _, dds_gt, zzs_gt, truncs_gt, occs_gt = out_gt
        boxes_m3d, dds_m3d = out_m3d
        boxes_3dop, dds_3dop = out_3dop
        boxes_md, dds_md = out_md
        boxes_our, dds_our, _, _, dds_geom = out_our
        if self.stereo:
            boxes_our_stereo, dds_our_stereo, _, _, dds_geom_stereo = out_our_stereo

        # Find IoU matches
        matches_our = get_iou_matches(boxes_our, boxes_gt, self.dic_thresh_iou['our'])
        matches_m3d = get_iou_matches(boxes_m3d, boxes_gt, self.dic_thresh_iou['m3d'])
        matches_3dop = get_iou_matches(boxes_3dop, boxes_gt, self.dic_thresh_iou['3dop'])
        matches_md = get_iou_matches(boxes_md, boxes_gt, self.dic_thresh_iou['md'])

        # Update error of commonly matched instances
        for (idx, idx_gt) in matches_our:
            check, indices = extract_indices(idx_gt, matches_m3d, matches_3dop, matches_md)
            if check:
                cat = get_category(boxes_gt[idx_gt], truncs_gt[idx_gt], occs_gt[idx_gt])
                dd_gt = dds_gt[idx_gt]

                self.update_errors(dds_our[idx], dd_gt, cat, self.errors['our_merged'])
                self.update_errors(dds_geom[idx], dd_gt, cat, self.errors['geom_merged'])
                self.update_errors(dd_gt + get_task_error(dd_gt), dd_gt, cat, self.errors['task_error_merged'])
                self.update_errors(dds_m3d[indices[0]], dd_gt, cat, self.errors['m3d_merged'])
                self.update_errors(dds_3dop[indices[1]], dd_gt, cat, self.errors['3dop_merged'])
                self.update_errors(dds_md[indices[2]], dd_gt, cat, self.errors['md_merged'])
                if self.stereo:
                    self.update_errors(dds_our_stereo[idx], dd_gt, cat, self.errors['our_stereo_merged'])
                    dd_pixel = get_pixel_error(dd_gt, zzs_gt[idx_gt])
                    self.update_errors(dd_pixel, dd_gt, cat, self.errors['pixel_error_merged'])
                    error = abs(dds_our[idx] - dd_gt)
                    error_stereo = abs(dds_our_stereo[idx] - dd_gt)
                    if error_stereo > (error + 0.1):
                        self.cnt_stereo_error += 1

                for key in self.METHODS:
                    self.dic_cnt[key + '_merged'] += 1

    def update_errors(self, dd, dd_gt, cat, errors):
        """Compute and save errors between a single box and the gt box which match"""

        diff = abs(dd - dd_gt)
        clst = find_cluster(dd_gt, self.CLUSTERS)
        errors['all'].append(diff)
        errors[cat].append(diff)
        errors[clst].append(diff)

        # Check if the distance is less than one or 2 meters
        if diff <= 0.5:
            errors['<0.5m'].append(1)
        else:
            errors['<0.5m'].append(0)

        if diff <= 1:
            errors['<1m'].append(1)
        else:
            errors['<1m'].append(0)

        if diff <= 2:
            errors['<2m'].append(1)
        else:
            errors['<2m'].append(0)

    def update_uncertainty(self, std_ale, std_epi, dd, dd_gt, cat):

        clst = find_cluster(dd_gt, self.CLUSTERS)
        self.dic_stds['all']['ale'].append(std_ale)
        self.dic_stds[clst]['ale'].append(std_ale)
        self.dic_stds[cat]['ale'].append(std_ale)
        self.dic_stds['all']['epi'].append(std_epi)
        self.dic_stds[clst]['epi'].append(std_epi)
        self.dic_stds[cat]['epi'].append(std_epi)

        # Number of annotations inside the confidence interval
        std = std_epi if std_epi > 0 else std_ale  # consider aleatoric uncertainty if epistemic is not calculated
        if abs(dd - dd_gt) <= std:
            self.dic_stds['all']['interval'].append(1)
            self.dic_stds[clst]['interval'].append(1)
            self.dic_stds[cat]['interval'].append(1)
        else:
            self.dic_stds['all']['interval'].append(0)
            self.dic_stds[clst]['interval'].append(0)
            self.dic_stds[cat]['interval'].append(0)

        # Annotations at risk inside the confidence interval
        if dd_gt <= dd:
            self.dic_stds['all']['at_risk'].append(1)
            self.dic_stds[clst]['at_risk'].append(1)
            self.dic_stds[cat]['at_risk'].append(1)

            if abs(dd - dd_gt) <= std_epi:
                self.dic_stds['all']['at_risk-interval'].append(1)
                self.dic_stds[clst]['at_risk-interval'].append(1)
                self.dic_stds[cat]['at_risk-interval'].append(1)
            else:
                self.dic_stds['all']['at_risk-interval'].append(0)
                self.dic_stds[clst]['at_risk-interval'].append(0)
                self.dic_stds[cat]['at_risk-interval'].append(0)

        else:
            self.dic_stds['all']['at_risk'].append(0)
            self.dic_stds[clst]['at_risk'].append(0)
            self.dic_stds[cat]['at_risk'].append(0)

        # Precision of uncertainty
        eps = 1e-4
        task_error = get_task_error(dd)
        prec_1 = abs(dd - dd_gt) / (std_epi + eps)

        prec_2 = abs(std_epi - task_error)
        self.dic_stds['all']['prec_1'].append(prec_1)
        self.dic_stds[clst]['prec_1'].append(prec_1)
        self.dic_stds[cat]['prec_1'].append(prec_1)
        self.dic_stds['all']['prec_2'].append(prec_2)
        self.dic_stds[clst]['prec_2'].append(prec_2)
        self.dic_stds[cat]['prec_2'].append(prec_2)

    def show_statistics(self):

        print('-'*90)
        alp = [[str(100 * average(self.errors[key][perc]))[:5]
                for perc in ['<0.5m', '<1m', '<2m']]
               for key in self.METHODS]

        ale = [[str(self.dic_stats['test'][key + '_merged'][clst]['mean'])[:4] + ' (' +
                str(self.dic_stats['test'][key][clst]['mean'])[:4] + ')'
                for clst in self.CLUSTERS[:4]]
               for key in self.METHODS]

        results = [[key] + alp[idx] + ale[idx] for idx, key in enumerate(self.METHODS)]
        print(tabulate(results, headers=self.HEADERS))
        print('-'*90 + '\n')

        if self.verbose:
            methods_all = list(chain.from_iterable((method, method + '_merged') for method in self.METHODS))
            for key in methods_all:
                for clst in self.CLUSTERS[:4]:
                    print(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                          "for {} annotations"
                          .format(key, clst, self.dic_stats['test'][key][clst]['mean'],
                                  self.dic_stats['test'][key][clst]['max'],
                                  self.dic_stats['test'][key][clst]['cnt']))

                    if key == 'our':
                        print("% of annotation inside the confidence interval: {:.1f} %, "
                              "of which {:.1f} % at higher risk"
                              .format(self.dic_stats['test'][key][clst]['interval']*100,
                                      self.dic_stats['test'][key][clst]['at_risk']*100))

                for perc in ['<0.5m', '<1m', '<2m']:
                    print("{} Instances with error {}: {:.2f} %"
                          .format(key, perc, 100 * average(self.errors[key][perc])))

                print("\nMatched annotations: {:.1f} %".format(self.errors[key]['matched']))
                print(" Detected annotations : {}/{} ".format(self.dic_cnt[key], self.cnt_gt))
                print("-" * 100)

            print("\n Annotations inside the confidence interval: {:.1f} %"
                  .format(self.dic_stats['test']['our']['all']['interval']))
            print("precision 1: {:.2f}".format(self.dic_stats['test']['our']['all']['prec_1']))
            print("precision 2: {:.2f}".format(self.dic_stats['test']['our']['all']['prec_2']))
            if self.stereo:
                print("Stereo error greater than mono: {:.1f} %"
                      .format(100 * self.cnt_stereo_error / self.dic_cnt['our_merged']))


def get_statistics(dic_stats, errors, dic_stds, key):
    """Update statistics of a cluster"""

    try:
        dic_stats['mean'] = average(errors)
        dic_stats['max'] = max(errors)
        dic_stats['cnt'] = len(errors)
    except (ZeroDivisionError, ValueError):
        dic_stats['mean'] = 0.
        dic_stats['max'] = 0.
        dic_stats['cnt'] = 0.

    if key == 'our':
        dic_stats['std_ale'] = average(dic_stds['ale'])
        dic_stats['std_epi'] = average(dic_stds['epi'])
        dic_stats['interval'] = average(dic_stds['interval'])
        dic_stats['at_risk'] = average(dic_stds['at_risk'])
        dic_stats['prec_1'] = average(dic_stds['prec_1'])
        dic_stats['prec_2'] = average(dic_stds['prec_2'])


def add_true_negatives(err, cnt_gt):
    """Update errors statistics of a specific method with missing detections"""

    matched = len(err['all'])
    missed = cnt_gt - matched
    zeros = [0] * missed
    err['<0.5m'].extend(zeros)
    err['<1m'].extend(zeros)
    err['<2m'].extend(zeros)
    err['matched'] = 100 * matched / cnt_gt


def find_cluster(dd, clusters):
    """Find the correct cluster. The first and the last one are not numeric"""

    for clst in clusters[4: -1]:
        if dd <= int(clst):
            return clst

    return clusters[-1]


def extract_indices(idx_to_check, *args):
    """
    Look if a given index j_gt is present in all the other series of indices (_, j)
    and return the corresponding one for argument

    idx_check --> gt index to check for correspondences in other method
    idx_method --> index corresponding to the method
    idx_gt --> index gt of the method
    idx_pred --> index of the predicted box of the method
    indices --> list of predicted indices for each method corresponding to the ground truth index to check
    """

    checks = [False]*len(args)
    indices = []
    for idx_method, method in enumerate(args):
        for (idx_pred, idx_gt) in method:
            if idx_gt == idx_to_check:
                checks[idx_method] = True
                indices.append(idx_pred)
    return all(checks), indices


def average(my_list):
    """calculate mean of a list"""
    return sum(my_list) / len(my_list)
