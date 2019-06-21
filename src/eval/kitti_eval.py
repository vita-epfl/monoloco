"""Evaluate Monoloco code on KITTI dataset using ALE and ALP metrics"""

import os
import math
import logging
from collections import defaultdict
import copy
import datetime
from utils.misc import get_idx_max
from utils.kitti import check_conditions, get_category, split_training, parse_ground_truth
from visuals.results import print_results


class KittiEval:
    """
    Evaluate Monoloco code and compare it with the following baselines:
    - Mono3D
    - 3DOP
    - MonoDepth
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    CLUSTERS = ('easy', 'moderate', 'hard', 'all', '6', '10', '15', '20', '25', '30', '40', '50', '>50')
    dic_stds = defaultdict(lambda: defaultdict(list))
    dic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    dic_cnt = defaultdict(int)
    errors = defaultdict(lambda: defaultdict(list))

    def __init__(self, thresh_iou_our=0.3, thresh_iou_m3d=0.5, thresh_conf_m3d=0.5, thresh_conf_our=0.3):
        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.dir_m3d = os.path.join('data', 'kitti', 'm3d')
        self.dir_3dop = os.path.join('data', 'kitti', '3dop')
        self.dir_md = os.path.join('data', 'kitti', 'monodepth')
        self.dir_our = os.path.join('data', 'kitti', 'monoloco')
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        dir_logs = os.path.join('data', 'logs')
        assert dir_logs, "No directory to save final statistics"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_results = os.path.join(dir_logs, 'eval-' + now_time + '.json')

        assert os.path.exists(self.dir_m3d) and os.path.exists(self.dir_our) \
               and os.path.exists(self.dir_3dop)

        self.dic_thresh_iou = {'m3d': thresh_iou_m3d, '3dop': thresh_iou_m3d, 'md': thresh_iou_our, 'our': thresh_iou_our}
        self.dic_thresh_conf = {'m3d': thresh_conf_m3d, '3dop': thresh_conf_m3d, 'our': thresh_conf_our}

        # Extract validation images for evaluation
        names_gt = tuple(os.listdir(self.dir_gt))
        _, self.set_val = split_training(names_gt, path_train, path_val)

    def run(self):
        """Evaluate Monoloco performances on ALP and ALE metrics"""

        # Iterate over each ground truth file in the training set
        cnt_gt = 0
        for name in self.set_val:
            path_gt = os.path.join(self.dir_gt, name)
            path_m3d = os.path.join(self.dir_m3d, name)
            path_our = os.path.join(self.dir_our, name)
            path_3dop = os.path.join(self.dir_3dop, name)
            path_md = os.path.join(self.dir_md, name)

            # Iterate over each line of the gt file and save box location and distances
            boxes_gt, _, dds_gt, truncs_gt, occs_gt = parse_ground_truth(path_gt)
            cnt_gt += len(boxes_gt)

            # Extract annotations for the same file
            if boxes_gt:
                boxes_m3d, dds_m3d = self._parse_txts(path_m3d, method='m3d')
                boxes_3dop, dds_3dop = self._parse_txts(path_3dop, method='3dop')
                boxes_md, dds_md = self._parse_txts(path_md, method='md')
                boxes_our, dds_our, stds_ale, stds_epi, _, dds_geom, _, _ = \
                    self._parse_txts(path_our, method='our')

                # Compute the error with ground truth
                self._estimate_error_base(boxes_m3d, dds_m3d, boxes_gt, dds_gt, truncs_gt, occs_gt, method='m3d')
                self._estimate_error_base(boxes_3dop, dds_3dop, boxes_gt, dds_gt, truncs_gt, occs_gt, method='3dop')
                self._estimate_error_base(boxes_md, dds_md, boxes_gt, dds_gt, truncs_gt, occs_gt, method='md')
                self._estimate_error_mloco(boxes_our, dds_our, stds_ale, stds_epi, dds_geom,
                                           boxes_gt, dds_gt, truncs_gt, occs_gt)

                # Iterate over all the files together to find a pool of common annotations
                self._compare_error(boxes_m3d, dds_m3d, boxes_3dop, dds_3dop, boxes_md, dds_md, boxes_our, dds_our,
                                    boxes_gt, dds_gt, truncs_gt, occs_gt, dds_geom)

        # Update statistics of errors and uncertainty
        for key in self.errors:
            add_true_negatives(self.errors[key], cnt_gt)
            for clst in self.CLUSTERS[:-2]:  # M3d and pifpaf does not have annotations above 40 meters
                get_statistics(self.dic_stats['test'][key][clst], self.errors[key][clst], self.dic_stds[clst], key)

        # Show statistics
        print(" Number of GT annotations: {} ".format(cnt_gt))
        for key in self.errors:
            if key in ['our', 'm3d', '3dop']:
                print(" Number of {} annotations with confidence >= {} : {} "
                      .format(key, self.dic_thresh_conf[key], self.dic_cnt[key]))

            for clst in self.CLUSTERS[:-9]:
                print(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                      "for {} annotations"
                      .format(key, clst, self.dic_stats['test'][key][clst]['mean'],
                              self.dic_stats['test'][key][clst]['max'],
                              self.dic_stats['test'][key][clst]['cnt']))

                if key == 'our':
                    print("% of annotation inside the confidence interval: {:.1f} %, "
                          "of which {:.1f} % at higher risk"
                          .format(100 * self.dic_stats['test'][key][clst]['interval'],
                                  100 * self.dic_stats['test'][key][clst]['at_risk']))

            for perc in ['<0.5m', '<1m', '<2m']:
                print("{} Instances with error {}: {:.2f} %"
                      .format(key, perc, 100 * sum(self.errors[key][perc])/len(self.errors[key][perc])))

            print("\n Number of matched annotations: {:.1f} %".format(self.errors[key]['matched']))
            print("-"*100)

    def printer(self, show):
        print_results(self.dic_stats, show)

    def _parse_txts(self, path, method):
        boxes = []
        dds = []
        stds_ale = []
        stds_epi = []
        dds_geom = []
        xyzs = []
        xy_kps = []

        # Iterate over each line of the txt file
        if method in ['3dop', 'm3d']:
            try:
                with open(path, "r") as ff:
                    for line in ff:
                        if check_conditions(line, thresh=self.dic_thresh_conf[method], mode=method):
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
                        delta_h = (box[3] - box[1]) / 10
                        delta_w = (box[2] - box[0]) / 10
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

        elif method == 'psm':
            try:
                with open(path, "r") as ff:
                    for line in ff:
                        box = [float(x[:-1]) for x in line[1:-1].split(',')[0:4]]
                        delta_h = (box[3] - box[1]) / 10
                        delta_w = (box[2] - box[0]) / 10
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
                    if check_conditions(line_list, thresh=self.dic_thresh_conf[method], mode=method):
                        boxes.append(line_list[:4])
                        xyzs.append(line_list[4:7])
                        dds.append(line_list[7])
                        stds_ale.append(line_list[8])
                        stds_epi.append(line_list[9])
                        dds_geom.append(line_list[11])
                        xy_kps.append(line_list[12:])

                        self.dic_cnt[method] += 1

                kk_list = [float(x) for x in file_lines[-1].split()]

                return boxes, dds, stds_ale, stds_epi, kk_list, dds_geom, xyzs, xy_kps

            except FileNotFoundError:
                return [], [], [], [], [], [], [], []

    def _estimate_error_base(self, boxes, dds, boxes_gt, dds_gt, truncs_gt, occs_gt, method):

        # Compute error (distance) and save it
        boxes_gt = copy.deepcopy(boxes_gt)
        dds_gt = copy.deepcopy(dds_gt)
        truncs_gt = copy.deepcopy(truncs_gt)
        occs_gt = copy.deepcopy(occs_gt)

        for idx, box in enumerate(boxes):
            if len(boxes_gt) >= 1:
                dd = dds[idx]
                idx_max, iou_max = get_idx_max(box, boxes_gt)
                cat = get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])
                # Update error if match is found
                if iou_max > self.dic_thresh_iou[method]:
                    dd_gt = dds_gt[idx_max]
                    self.update_errors(dd, dd_gt, cat, self.errors[method])

                    boxes_gt.pop(idx_max)
                    dds_gt.pop(idx_max)
                    truncs_gt.pop(idx_max)
                    occs_gt.pop(idx_max)
            else:
                break

    def _estimate_error_mloco(self, boxes, dds, stds_ale, stds_epi, dds_geom, boxes_gt, dds_gt, truncs_gt, occs_gt):

        # Compute error (distance) and save it
        boxes_gt = copy.deepcopy(boxes_gt)
        dds_gt = copy.deepcopy(dds_gt)
        truncs_gt = copy.deepcopy(truncs_gt)
        occs_gt = copy.deepcopy(occs_gt)

        for idx, box in enumerate(boxes):
            if len(boxes_gt) >= 1:
                dd = dds[idx]
                dd_geom = dds_geom[idx]
                ale = stds_ale[idx]
                epi = stds_epi[idx]
                idx_max, iou_max = get_idx_max(box, boxes_gt)
                cat = get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])

                # Update error if match is found
                if iou_max > self.dic_thresh_iou['our']:
                    dd_gt = dds_gt[idx_max]
                    self.update_errors(dd, dd_gt, cat, self.errors['our'])
                    self.update_errors(dd_geom, dd_gt, cat, self.errors['geom'])
                    self.update_uncertainty(ale, epi, dd, dd_gt, cat)

                    boxes_gt.pop(idx_max)
                    dds_gt.pop(idx_max)
                    truncs_gt.pop(idx_max)
                    occs_gt.pop(idx_max)

    def _compare_error(self, boxes_m3d, dds_m3d, boxes_3dop, dds_3dop, boxes_md, dds_md, boxes_our, dds_our,
                       boxes_gt, dds_gt, truncs_gt, occs_gt, dds_geom):

        boxes_gt = copy.deepcopy(boxes_gt)
        dds_gt = copy.deepcopy(dds_gt)
        truncs_gt = copy.deepcopy(truncs_gt)
        occs_gt = copy.deepcopy(occs_gt)

        for idx, box in enumerate(boxes_our):
            if len(boxes_gt) >= 1:
                dd_our = dds_our[idx]
                dd_geom = dds_geom[idx]
                idx_max, iou_max = get_idx_max(box, boxes_gt)
                cat = get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])

                idx_max_3dop, iou_max_3dop = get_idx_max(box, boxes_3dop)
                idx_max_m3d, iou_max_m3d = get_idx_max(box, boxes_m3d)
                idx_max_md, iou_max_md = get_idx_max(box, boxes_md)

                iou_min = min(iou_max_3dop, iou_max_m3d, iou_max_md)

                if iou_max >= self.dic_thresh_iou['our'] and iou_min >= self.dic_thresh_iou['m3d']:
                    dd_gt = dds_gt[idx_max]
                    dd_3dop = dds_3dop[idx_max_3dop]
                    dd_m3d = dds_m3d[idx_max_m3d]
                    dd_md = dds_md[idx_max_md]

                    self.update_errors(dd_3dop, dd_gt, cat, self.errors['3dop_merged'])
                    self.update_errors(dd_our, dd_gt, cat, self.errors['our_merged'])
                    self.update_errors(dd_m3d, dd_gt, cat, self.errors['m3d_merged'])
                    self.update_errors(dd_geom, dd_gt, cat, self.errors['geom_merged'])
                    self.update_errors(dd_md, dd_gt, cat, self.errors['md_merged'])
                    self.dic_cnt['merged'] += 1

                    boxes_gt.pop(idx_max)
                    dds_gt.pop(idx_max)
                    truncs_gt.pop(idx_max)
                    occs_gt.pop(idx_max)
            else:
                break

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
        if dd_gt <= dd:  # Particularly dangerous instances
            self.dic_stds['all']['at_risk'].append(1)
            self.dic_stds[clst]['at_risk'].append(1)
            self.dic_stds[cat]['at_risk'].append(1)

            if abs(dd - dd_gt) <= std_epi:
                self.dic_stds['all']['interval'].append(1)
                self.dic_stds[clst]['interval'].append(1)
                self.dic_stds[cat]['interval'].append(1)

            else:
                self.dic_stds['all']['interval'].append(0)
                self.dic_stds[clst]['interval'].append(0)
                self.dic_stds[cat]['interval'].append(0)

        else:
            self.dic_stds['all']['at_risk'].append(0)
            self.dic_stds[clst]['at_risk'].append(0)
            self.dic_stds[cat]['at_risk'].append(0)


def get_statistics(dic_stats, errors, dic_stds, key):
    """Update statistics of a cluster"""

    dic_stats['mean'] = sum(errors) / float(len(errors))
    dic_stats['max'] = max(errors)
    dic_stats['cnt'] = len(errors)

    if key == 'our':
        dic_stats['std_ale'] = sum(dic_stds['ale']) / float(len(dic_stds['ale']))
        dic_stats['std_epi'] = sum(dic_stds['epi']) / float(len(dic_stds['epi']))
        dic_stats['interval'] = sum(dic_stds['interval']) / float(len(dic_stds['interval']))
        dic_stats['at_risk'] = sum(dic_stds['at_risk']) / float(len(dic_stds['at_risk']))


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
