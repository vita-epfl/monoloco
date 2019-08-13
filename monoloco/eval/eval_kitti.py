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
    METHODS_MONO = ['m3d', 'monodepth', '3dop', 'monoloco']
    METHODS_STEREO = ['monoloco_stereo']
    BASELINES = ['geom', 'task_error', 'pixel_error']
    HEADERS = ['method', '<0.5', '<1m', '<2m', 'easy', 'moderate', 'hard', 'all']
    CATEGORIES = ['pedestrian']

    def __init__(self, thresh_iou_monoloco=0.3, thresh_iou_m3d=0.3, thresh_conf_m3d=0.3, thresh_conf_monoloco=0.3,
                 verbose=False, stereo=False):

        self.main_dir = os.path.join('data', 'kitti')
        self.dir_gt = os.path.join(self.main_dir, 'gt')
        self.methods = self.METHODS_MONO
        self.all_methods = self.METHODS_MONO + self.BASELINES
        self.stereo = stereo
        if self.stereo:
            self.methods.extend(self.METHODS_STEREO)
            self.all_methods.extend(self.METHODS_STEREO)
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        dir_logs = os.path.join('data', 'logs')
        assert dir_logs, "No directory to save final statistics"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_results = os.path.join(dir_logs, 'eval-' + now_time + '.json')
        self.verbose = verbose

        self.dic_thresh_iou = {'m3d': thresh_iou_m3d, '3dop': thresh_iou_m3d,
                               'monodepth': thresh_iou_m3d, 'monoloco': thresh_iou_monoloco,
                               'monoloco_stereo': thresh_iou_monoloco}
        self.dic_thresh_conf = {'m3d': thresh_conf_m3d, '3dop': thresh_conf_m3d,
                                'monoloco': thresh_conf_m3d, 'monoloco_stereo': thresh_conf_monoloco}

        # Extract validation images for evaluation
        names_gt = tuple(os.listdir(self.dir_gt))
        _, self.set_val = split_training(names_gt, path_train, path_val)

        # Define variables to save statistics
        self.dic_methods = self.errors = self.dic_stds = self.dic_stats = self.dic_cnt = self.cnt_stereo_error = None
        self.cnt_gt = 0

    def run(self):
        """Evaluate Monoloco performances on ALP and ALE metrics"""

        for category in self.CATEGORIES:

            # Initialize variables
            self.errors = self.dic_stds = defaultdict(lambda: defaultdict(list))
            self.dic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
            self.dic_cnt = defaultdict(int)
            self.cnt_gt = self.cnt_stereo_error = 0

            # Iterate over each ground truth file in the training set
            for name in self.set_val:
                path_gt = os.path.join(self.dir_gt, name)

                # Iterate over each line of the gt file and save box location and distances
                out_gt = parse_ground_truth(path_gt, category)
                methods_out = defaultdict(tuple)  # Save all methods for comparison
                self.cnt_gt += len(out_gt[0])

                if out_gt[0]:
                    for method in self.methods:
                        # Extract annotations
                        dir_method = os.path.join(self.main_dir, method)
                        assert os.path.exists(dir_method), "directory of the method %s does not exists" % method
                        path_method = os.path.join(dir_method, name)
                        methods_out[method] = self._parse_txts(path_method, category, method=method)

                        # Compute the error with ground truth
                        self._estimate_error(out_gt, methods_out[method], method=method)

                    # Iterate over all the files together to find a pool of common annotations
                    self._compare_error(out_gt, methods_out)

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

        elif method == 'monodepth':
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

        elif method == 'monoloco':
            stds_ale = []
            stds_epi = []
            dds_geom = []
            try:
                with open(path, "r") as ff:
                    file_lines = ff.readlines()
                for line_monoloco in file_lines[:-1]:
                    line_list = [float(x) for x in line_monoloco.split()]
                    if check_conditions(line_list, category, method=method, thresh=self.dic_thresh_conf[method]):
                        boxes.append(line_list[:4])
                        dds.append(line_list[8])
                        stds_ale.append(line_list[9])
                        stds_epi.append(line_list[10])
                        dds_geom.append(line_list[11])
                        self.dic_cnt['geom'] += 1
                        self.dic_cnt[method] += 1
                return boxes, dds, stds_ale, stds_epi, dds_geom
            except FileNotFoundError:
                return [], [], [], [], []

        else:
            assert method == 'monoloco_stereo', "method not recognized"  # TODO Remove
            stds_ale = []
            stds_epi = []
            dds_geom = []
            try:
                with open(path, "r") as ff:
                    file_lines = ff.readlines()
                for line_monoloco in file_lines[:-1]:
                    line_list = [float(x) for x in line_monoloco.split()]
                    if check_conditions(line_list, category, method=method, thresh=self.dic_thresh_conf[method]):
                        boxes.append(line_list[:4])
                        dds.append(line_list[8])
                        stds_ale.append(line_list[9])
                        stds_epi.append(line_list[10])
                        dds_geom.append(line_list[11])
                        self.dic_cnt['geom'] += 1
                        self.dic_cnt[method] += 1
                return boxes, dds, stds_ale, stds_epi, dds_geom
            except FileNotFoundError:
                return [], [], [], [], []

    def _estimate_error(self, out_gt, out, method):
        """Estimate localization error"""

        boxes_gt, _, dds_gt, zzs_gt, truncs_gt, occs_gt = out_gt
        if method in ('monoloco', 'monoloco_stereo'):
            boxes, dds, stds_ale, stds_epi, dds_geom = out
        else:
            boxes, dds = out

        matches = get_iou_matches(boxes, boxes_gt, self.dic_thresh_iou[method])

        for (idx, idx_gt) in matches:
            # Update error if match is found
            cat = get_category(boxes_gt[idx_gt], truncs_gt[idx_gt], occs_gt[idx_gt])
            self.update_errors(dds[idx], dds_gt[idx_gt], cat, self.errors[method])

            if method == 'monoloco':
                self.update_errors(dds_geom[idx], dds_gt[idx_gt], cat, self.errors['geom'])
                self.update_uncertainty(stds_ale[idx], stds_epi[idx], dds[idx], dds_gt[idx_gt], cat)
                dd_task_error = dds_gt[idx_gt] + (get_task_error(dds_gt[idx_gt]))**2
                self.update_errors(dd_task_error, dds_gt[idx_gt], cat, self.errors['task_error'])

            elif method == 'monoloco_stereo':
                dd_pixel_error = get_pixel_error(dds_gt[idx_gt], zzs_gt[idx_gt])
                self.update_errors(dd_pixel_error, dds_gt[idx_gt], cat, self.errors['pixel_error'])

    def _compare_error(self, out_gt, methods_out):
        """Compare the error for a pool of instances commonly matched by all methods"""

        # Extract outputs of each method
        boxes_gt, _, dds_gt, zzs_gt, truncs_gt, occs_gt = out_gt
        # boxes_m3d, dds_m3d = out_m3d
        # boxes_3dop, dds_3dop = out_3dop
        # boxes_md, dds_md = out_md
        # boxes_our, dds_our, _, _, dds_geom = out_our
        # if self.stereo:
        #     boxes_our_stereo, dds_our_stereo, _, _, dds_geom_stereo = out_our_stereo
        matches = defaultdict(list)
        for method in self.methods:
            # Find IoU matches
            boxes = methods_out[method][0]
            matches[method] = get_iou_matches(boxes, boxes_gt, self.dic_thresh_iou[method])
        # matches_our = get_iou_matches(boxes_our, boxes_gt, self.dic_thresh_iou['our'])
        # matches_m3d = get_iou_matches(boxes_m3d, boxes_gt, self.dic_thresh_iou['m3d'])
        # matches_3dop = get_iou_matches(boxes_3dop, boxes_gt, self.dic_thresh_iou['3dop'])
        # matches_md = get_iou_matches(boxes_md, boxes_gt, self.dic_thresh_iou['md'])

        # Update error of commonly matched instances
        for (idx, idx_gt) in matches['monoloco']:
            methods_matches = [matches[method] for method in self.methods if method != 'monoloco']
            # check, indices = extract_indices(idx_gt, matches_m3d, matches_3dop, matches_md)
            check, indices = extract_indices(idx_gt, *methods_matches)
            if check:
                cat = get_category(boxes_gt[idx_gt], truncs_gt[idx_gt], occs_gt[idx_gt])
                dd = methods_out[method][1][idx]
                dd_gt = dds_gt[idx_gt]
                for method in self.methods:
                    self.update_errors(dd, dd_gt, cat, self.errors[method + '_merged'])
                    # self.update_errors(dds_our[idx], dd_gt, cat, self.errors['monoloco_merged'])
                    # self.update_errors(dds_geom[idx], dd_gt, cat, self.errors['geom_merged'])
                    # self.update_errors(dd_gt + get_task_error(dd_gt), dd_gt, cat, self.errors['task_error_merged'])
                    # self.update_errors(dds_m3d[indices[0]], dd_gt, cat, self.errors['m3d_merged'])
                    # self.update_errors(dds_3dop[indices[1]], dd_gt, cat, self.errors['3dop_merged'])
                    # self.update_errors(dds_md[indices[2]], dd_gt, cat, self.errors['monodepth_merged'])
                self.update_errors(dd_gt + get_task_error(dd_gt), dd_gt, cat, self.errors['task_error_merged'])
                dd_pixel = get_pixel_error(dd_gt, zzs_gt[idx_gt])
                self.update_errors(dd_pixel, dd_gt, cat, self.errors['pixel_error_merged'])

                # if self.stereo:
                #     error = abs(dds_our[idx] - dd_gt)
                #     error_stereo = abs(dds_our_stereo[idx] - dd_gt)
                #     if error_stereo > (error + 0.1):
                #         self.cnt_stereo_error += 1
                #     for key in self.METHODS_STEREO:
                #         self.dic_cnt[key + '_merged'] += 1

                for key in self.methods:
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
        self.summary_table()

        if self.verbose:
            methods_all = list(chain.from_iterable((method, method + '_merged') for method in self.METHODS))
            for key in methods_all:
                for clst in self.CLUSTERS[:4]:
                    print(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                          "for {} annotations"
                          .format(key, clst, self.dic_stats['test'][key][clst]['mean'],
                                  self.dic_stats['test'][key][clst]['max'],
                                  self.dic_stats['test'][key][clst]['cnt']))

                    if key == 'monoloco':
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
                  .format(self.dic_stats['test']['monoloco']['all']['interval']))
            print("precision 1: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_1']))
            print("precision 2: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_2']))
            if self.stereo:
                print("Stereo error greater than mono: {:.1f} %"
                      .format(100 * self.cnt_stereo_error / self.dic_cnt['monoloco_merged']))

    def summary_table(self):

        alp = [[str(100 * average(self.errors[key][perc]))[:5]
                for perc in ['<0.5m', '<1m', '<2m']]
               for key in self.all_methods]

        ale = [[str(self.dic_stats['test'][key + '_merged'][clst]['mean'])[:4] + ' (' +
                str(self.dic_stats['test'][key][clst]['mean'])[:4] + ')'
                for clst in self.CLUSTERS[:4]]
               for key in self.all_methods]

        results = [[key] + alp[idx] + ale[idx] for idx, key in enumerate(self.all_methods)]
        print(tabulate(results, headers=self.HEADERS))
        print('-' * 90 + '\n')


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

    if key == 'monoloco':
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
