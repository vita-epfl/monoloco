
import os
import math
import logging
from collections import defaultdict
import json
import copy
import datetime


class KittiEval:
    """
    Evaluate Monoloco code on KITTI dataset and compare it with:
    - Mono3D
    - 3DOP
    - MonoDepth
    """

    def __init__(self, show=False, thresh_iou_our=0.3, thresh_iou_m3d=0.5, thresh_conf_m3d=0.5, thresh_conf_our=0.3):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.show = show
        from utils.misc import get_idx_max
        self.get_idx_max = get_idx_max
        from utils.kitti import check_conditions, get_category
        self.check_conditions = check_conditions
        self.get_category = get_category
        from visuals.results import print_results
        self.print_results = print_results

        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.dir_m3d = os.path.join('data', 'kitti', 'm3d')
        self.dir_3dop = os.path.join('data', 'kitti', '3dop')
        self.dir_md = os.path.join('data', 'kitti', 'monodepth')
        self.dir_psm = os.path.join('data', 'kitti', 'psm')
        self.dir_our = os.path.join('data', 'kitti', 'monoloco')
        path_val = os.path.join('splits', 'kitti_val.txt')
        dir_logs = os.path.join('data', 'logs')
        assert dir_logs, "No directory to save final statistics"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_results = os.path.join(dir_logs, 'eval-' + now_time + '.json')

        assert os.path.exists(self.dir_m3d) and os.path.exists(self.dir_our) \
               and os.path.exists(self.dir_3dop)

        self.clusters = ['easy', 'moderate', 'hard', 'all', '6', '10', '15', '20', '25', '30', '40', '50', '>50']

        self.dic_thresh_iou = {'m3d': thresh_iou_m3d, '3dop': thresh_iou_m3d, 'md': thresh_iou_our,
                               'psm': thresh_iou_our, 'our': thresh_iou_our}
        self.dic_thresh_conf = {'m3d': thresh_conf_m3d, '3dop': thresh_conf_m3d, 'our': thresh_conf_our}

        self.dic_cnt = defaultdict(int)
        self.errors = defaultdict(lambda: defaultdict(list))

        # Only consider validation images
        set_gt = set(os.listdir(self.dir_gt))
        set_val = set()

        with open(path_val, "r") as f_val:
            for line in f_val:
                set_val.add(line[:-1] + '.txt')
        self.list_gt = list(set_gt.intersection(set_val))
        assert self.list_gt, "No images in the folder"

    def run(self):

        self.dic_stds = defaultdict(lambda: defaultdict(list))
        dic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        cnt_gt = 0

        # Iterate over each ground truth file in the training set
        for name in self.list_gt:
            if name == '004647.txt':
                aa = 5
            path_gt = os.path.join(self.dir_gt, name)
            path_m3d = os.path.join(self.dir_m3d, name)
            path_our = os.path.join(self.dir_our, name)
            path_3dop = os.path.join(self.dir_3dop, name)
            path_md = os.path.join(self.dir_md, name)
            path_psm = os.path.join(self.dir_psm, name)
            boxes_gt = []
            truncs_gt = [] # Float from 0 to 1
            occs_gt = []  # Either 0,1,2,3 fully visible, partly occluded, largely occluded, unknown
            dds_gt = []
            dic_fin = defaultdict(list)

            # Iterate over each line of the gt file and save box location and distances
            with open(path_gt, "r") as f_gt:
                for line_gt in f_gt:
                    if self.check_conditions(line_gt, mode='gt'):
                        truncs_gt.append(float(line_gt.split()[1]))
                        occs_gt.append(int(line_gt.split()[2]))
                        boxes_gt.append([float(x) for x in line_gt.split()[4:8]])
                        loc_gt = [float(x) for x in line_gt.split()[11:14]]
                        dds_gt.append(math.sqrt(loc_gt[0] ** 2 + loc_gt[1] ** 2 + loc_gt[2] ** 2))
                        cnt_gt += 1

            # Extract annotations for the same file
            if len(boxes_gt) > 0:
                boxes_m3d, dds_m3d = self.parse_txts(path_m3d,  method='m3d')
                boxes_3dop, dds_3dop = self.parse_txts(path_3dop, method='3dop')
                boxes_md, dds_md = self.parse_txts(path_md, method='md')
                boxes_psm, dds_psm = self.parse_txts(path_psm, method='psm')
                boxes_our, dds_our, stds_ale, stds_epi, kk_list, dds_geom, xyzs, xy_kps = \
                    self.parse_txts(path_our, method='our')

                if len(boxes_our) > 0 and len(boxes_psm) == 0:
                    aa = 5

                # Compute the error with ground truth

                self.estimate_error_base(boxes_m3d, dds_m3d, boxes_gt, dds_gt, truncs_gt, occs_gt, method='m3d')
                self.estimate_error_base(boxes_3dop, dds_3dop, boxes_gt, dds_gt, truncs_gt, occs_gt,  method='3dop')
                self.estimate_error_base(boxes_md, dds_md, boxes_gt, dds_gt, truncs_gt, occs_gt, method='md')
                self.estimate_error_base(boxes_psm, dds_psm, boxes_gt, dds_gt, truncs_gt, occs_gt, method='psm')
                self.estimate_error_our(boxes_our, dds_our, stds_ale, stds_epi, kk_list, dds_geom, xyzs, xy_kps,
                                        boxes_gt, dds_gt, truncs_gt, occs_gt, dic_fin, name)

                # Iterate over all the files together to find a pool of common annotations
                self.compare_error(boxes_m3d, dds_m3d, boxes_3dop, dds_3dop, boxes_md, dds_md, boxes_our, dds_our,
                                   boxes_psm, dds_psm, boxes_gt, dds_gt, truncs_gt, occs_gt, dds_geom)

        # Save statistics
        for key in self.errors:
            for clst in self.clusters[:-2]:  # M3d and pifpaf does not have annotations above 40 meters
                dic_stats['test'][key]['mean'][clst] = sum(self.errors[key][clst]) / float(len(self.errors[key][clst]))
                dic_stats['test'][key]['max'][clst] = max(self.errors[key][clst])
                dic_stats['test'][key]['cnt'][clst] = len(self.errors[key][clst])

            if key == 'our':
                for clst in self.clusters[:-2]:
                    dic_stats['test'][key]['std_ale'][clst] = \
                        sum(self.dic_stds['ale'][clst]) / float(len(self.dic_stds['ale'][clst]))
                    dic_stats['test'][key]['std_epi'][clst] = \
                        sum(self.dic_stds['epi'][clst]) / float(len(self.dic_stds['epi'][clst]))
                    dic_stats['test'][key]['interval'][clst] = \
                        sum(self.dic_stds['interval'][clst]) / float(len(self.dic_stds['interval'][clst]))
                    dic_stats['test'][key]['at_risk'][clst] = \
                        sum(self.dic_stds['at_risk'][clst]) / float(len(self.dic_stds['at_risk'][clst]))

        # Print statistics
        print(" Number of GT annotations: {} ".format(cnt_gt))
        for key in self.errors:
            if key in ['our', 'm3d', '3dop']:
                print(" Number of {} annotations with confidence >= {} : {} "
                      .format(key, self.dic_thresh_conf[key], self.dic_cnt[key]))

            # Include also missed annotations in the statistics
            matched = len(self.errors[key]['all'])
            missed = cnt_gt - matched
            zeros = [0] * missed
            self.errors[key]['<0.5m'].extend(zeros)
            self.errors[key]['<1m'].extend(zeros)
            self.errors[key]['<2m'].extend(zeros)

            for clst in self.clusters[:-9]:
                print(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                      "for {} annotations"
                      .format(key, clst, dic_stats['test'][key]['mean'][clst], dic_stats['test'][key]['max'][clst],
                              dic_stats['test'][key]['cnt'][clst]))

                if key == 'our':
                    print("% of annotation inside the confidence interval: {:.1f} %, "
                          "of which {:.1f} % at higher risk"
                          .format(100 * dic_stats['test'][key]['interval'][clst],
                                  100 * dic_stats['test'][key]['at_risk'][clst]))

            for perc in ['<0.5m', '<1m', '<2m']:
                print("{} Instances with error {}: {:.2f} %"
                      .format(key, perc, 100 * sum(self.errors[key][perc])/len(self.errors[key][perc])))

            print("\n Number of matched annotations: {:.1f} %".format(100 * matched/cnt_gt))
            print("-"*100)

        # Print images
        self.print_results(dic_stats, self.show)

    def parse_txts(self, path, method):
        boxes = []
        dds = []
        stds_ale = []
        stds_epi = []
        confs = []
        dds_geom = []
        xyzs = []
        xy_kps = []

        # Iterate over each line of the txt file
        if method == '3dop' or method == 'm3d':
            try:
                with open(path, "r") as ff:
                    for line in ff:
                        if self.check_conditions(line, thresh=self.dic_thresh_conf[method], mode=method):
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

        elif method == 'our':
            try:
                with open(path, "r") as ff:
                    file_lines = ff.readlines()
                for line_our in file_lines[:-1]:
                    line_list = [float(x) for x in line_our.split()]
                    if self.check_conditions(line_list, thresh=self.dic_thresh_conf[method], mode=method):
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

    def estimate_error_base(self, boxes, dds, boxes_gt, dds_gt, truncs_gt, occs_gt, method):

        # Compute error (distance) and save it
        boxes_gt = copy.deepcopy(boxes_gt)
        dds_gt = copy.deepcopy(dds_gt)
        truncs_gt = copy.deepcopy(truncs_gt)
        occs_gt = copy.deepcopy(occs_gt)

        for idx, box in enumerate(boxes):
            if len(boxes_gt) >= 1:
                dd = dds[idx]
                idx_max, iou_max = self.get_idx_max(box, boxes_gt)
                cat = self.get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])
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

    def estimate_error_our(self, boxes, dds, stds_ale, stds_epi, kk_list, dds_geom, xyzs, xy_kps,
                           boxes_gt, dds_gt, truncs_gt, occs_gt, dic_fin, name):

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
                xyz = xyzs[idx]
                xy_kp = xy_kps[idx]
                idx_max, iou_max = self.get_idx_max(box, boxes_gt)
                cat = self.get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])

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

                    # Extract K and save it everything in a json file
                    dic_fin['boxes'].append(box)
                    dic_fin['dds_gt'].append(dd_gt)
                    dic_fin['dds_pred'].append(dd)
                    dic_fin['stds_ale'].append(ale)
                    dic_fin['stds_epi'].append(epi)
                    dic_fin['dds_geom'].append(dd_geom)
                    dic_fin['xyz'].append(xyz)
                    dic_fin['xy_kps'].append(xy_kp)
            else:
                break

            # kk_fin = np.array(kk_list).reshape(3, 3).tolist()
            # dic_fin['K'] = kk_fin
            # path_json = os.path.join(self.dir_fin, name[:-4] + '.json')
            # with open(path_json, 'w') as ff:
            #     json.dump(dic_fin, ff)

    def compare_error(self, boxes_m3d, dds_m3d, boxes_3dop, dds_3dop, boxes_md, dds_md, boxes_our, dds_our,
                      boxes_psm, dds_psm, boxes_gt, dds_gt, truncs_gt, occs_gt, dds_geom):

        boxes_gt = copy.deepcopy(boxes_gt)
        dds_gt = copy.deepcopy(dds_gt)
        truncs_gt = copy.deepcopy(truncs_gt)
        occs_gt = copy.deepcopy(occs_gt)

        for idx, box in enumerate(boxes_our):
            if len(boxes_gt) >= 1:
                dd_our = dds_our[idx]
                dd_geom = dds_geom[idx]
                idx_max, iou_max = self.get_idx_max(box, boxes_gt)
                cat = self.get_category(boxes_gt[idx_max], truncs_gt[idx_max], occs_gt[idx_max])

                idx_max_3dop, iou_max_3dop = self.get_idx_max(box, boxes_3dop)
                idx_max_m3d, iou_max_m3d = self.get_idx_max(box, boxes_m3d)
                idx_max_md, iou_max_md = self.get_idx_max(box, boxes_md)
                # idx_max_psm, iou_max_psm = self.get_idx_max(box, boxes_psm)
                iou_max_psm = 1

                iou_min = min(iou_max_3dop, iou_max_m3d, iou_max_md, iou_max_psm)

                if iou_max >= self.dic_thresh_iou['our'] and iou_min >= self.dic_thresh_iou['m3d']:

                    dd_gt = dds_gt[idx_max]
                    dd_3dop = dds_3dop[idx_max_3dop]
                    dd_m3d = dds_m3d[idx_max_m3d]
                    dd_md = dds_md[idx_max_md]
                    # dd_psm = dds_psm[idx_max_psm]

                    self.update_errors(dd_3dop, dd_gt, cat, self.errors['3dop_merged'])
                    self.update_errors(dd_our, dd_gt, cat, self.errors['our_merged'])
                    self.update_errors(dd_m3d, dd_gt, cat, self.errors['m3d_merged'])
                    self.update_errors(dd_geom, dd_gt, cat, self.errors['geom_merged'])
                    self.update_errors(dd_md, dd_gt, cat, self.errors['md_merged'])
                    # self.update_errors(dd_psm, dd_gt, cat, self.errors['psm_merged'])
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
        clst = self.find_cluster(dd_gt, self.clusters)
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

        clst = self.find_cluster(dd_gt, self.clusters)
        self.dic_stds['ale']['all'].append(std_ale)
        self.dic_stds['ale'][clst].append(std_ale)
        self.dic_stds['ale'][cat].append(std_ale)
        self.dic_stds['epi']['all'].append(std_epi)
        self.dic_stds['epi'][clst].append(std_epi)
        self.dic_stds['epi'][cat].append(std_epi)

        # Number of annotations inside the confidence interval
        if dd_gt <= dd:  # Particularly dangerous instances
            self.dic_stds['at_risk']['all'].append(1)
            self.dic_stds['at_risk'][clst].append(1)
            self.dic_stds['at_risk'][cat].append(1)

            if abs(dd - dd_gt) <= (std_epi):
                self.dic_stds['interval']['all'].append(1)
                self.dic_stds['interval'][clst].append(1)
                self.dic_stds['interval'][cat].append(1)

            else:
                self.dic_stds['interval']['all'].append(0)
                self.dic_stds['interval'][clst].append(0)
                self.dic_stds['interval'][cat].append(0)


        else:
            self.dic_stds['at_risk']['all'].append(0)
            self.dic_stds['at_risk'][clst].append(0)
            self.dic_stds['at_risk'][cat].append(0)


            # self.dic_stds['at_risk']['all'].append(0)
            # self.dic_stds['at_risk'][clst].append(0)
            # self.dic_stds['at_risk'][cat].append(0)

    @staticmethod
    def find_cluster(dd, clusters):
        """Find the correct cluster. The first and the last one are not numeric"""

        for clst in clusters[4: -1]:
            if dd <= int(clst):
                return clst

        return clusters[-1]
