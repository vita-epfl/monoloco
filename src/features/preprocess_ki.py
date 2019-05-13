
import os
import glob
import math
import logging
from collections import defaultdict
import json
import datetime


class PreprocessKitti:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    def __init__(self, dir_ann, iou_thresh=0.3):

        self.dir_ann = dir_ann
        self.iou_thresh = iou_thresh
        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.names_gt = os.listdir(self.dir_gt)
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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        from utils.kitti import get_calibration, check_conditions
        self.get_calibration = get_calibration
        self.check_conditions = check_conditions

        from utils.pifpaf import get_input_data, preprocess_pif
        self.get_input_data = get_input_data
        self.preprocess_pif = preprocess_pif

        from utils.misc import get_idx_max, append_cluster
        self.get_idx_max = get_idx_max
        self.append_cluster = append_cluster

        # self.clusters = ['all', '6', '10', '15', '20', '25', '30', '40', '50', '>50'
        self.cnt_gt = 0
        self.cnt_fnf = 0
        self.dic_cnt = {'train': 0, 'val': 0, 'test': 0}

        # Split training and validation images
        set_gt = set(self.names_gt)
        set_train = set()
        set_val = set()

        with open(path_train, "r") as f_train:
            for line in f_train:
                set_train.add(line[:-1] + '.txt')
        with open(path_val, "r") as f_val:
            for line in f_val:
                set_val.add(line[:-1] + '.txt')

        self.set_train = set_gt.intersection(set_train)
        self.set_val = set_gt.intersection(set_val)
        assert self.set_train and self.set_val, "No validation or training annotations"

        self.dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], K=[],
                                     clst=defaultdict(lambda: defaultdict(list))),
                       'val': dict(X=[], Y=[], names=[], kps=[], K=[],
                                   clst=defaultdict(lambda: defaultdict(list))),
                       'test': dict(X=[], Y=[], names=[], kps=[], K=[],
                                    clst=defaultdict(lambda: defaultdict(list)))}

        self.dic_names = defaultdict(lambda: defaultdict(list))

    def run(self):

        for name in self.names_gt:
            # Extract ground truth
            if name == '004223.txt':
                aa = 5
            path_gt = os.path.join(self.dir_gt, name)
            basename, _ = os.path.splitext(name)
            boxes_gt = []
            dds = []

            if name in self.set_train:
                phase = 'train'
            elif name in self.set_val:
                phase = 'val'
            else:
                self.cnt_fnf += 1
                continue

            # Extract keypoints
            path_txt = os.path.join(self.dir_kk, basename + '.txt')
            kk, tt = self.get_calibration(path_txt)

            # Iterate over each line of the gt file and save box location and distances
            with open(path_gt, "r") as f_gt:
                for line_gt in f_gt:
                    if self.check_conditions(line_gt, mode='gt'):
                        box = [float(x) for x in line_gt.split()[4:8]]
                        boxes_gt.append(box)
                        loc_gt = [float(x) for x in line_gt.split()[11:14]]
                        dd = math.sqrt(loc_gt[0] ** 2 + loc_gt[1] ** 2 + loc_gt[2] ** 2)
                        dds.append(dd)
                        self.dic_names[basename + '.png']['boxes'].append(box)
                        self.dic_names[basename + '.png']['dds'].append(dd)
                        self.dic_names[basename + '.png']['K'] = kk.tolist()
                        self.cnt_gt += 1

            # Find the annotations if exists
            try:
                with open(os.path.join(self.dir_ann, basename + '.png.pifpaf.json'), 'r') as f:
                    annotations = json.load(f)
                boxes, keypoints = self.preprocess_pif(annotations)
                (inputs, xy_kps), (uv_kps, uv_boxes, _, _) = self.get_input_data(boxes, keypoints, kk)

            except FileNotFoundError:
                uv_boxes = []

            # Match each set of keypoint with a ground truth
            for ii, box in enumerate(uv_boxes):
                idx_max, iou_max = self.get_idx_max(box, boxes_gt)

                if iou_max >= self.iou_thresh:

                    self.dic_jo[phase]['kps'].append(uv_kps[ii])
                    self.dic_jo[phase]['X'].append(inputs[ii])
                    self.dic_jo[phase]['Y'].append([dds[idx_max]])  # Trick to make it (nn,1)
                    self.dic_jo[phase]['K'] = kk.tolist()
                    self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                    self.append_cluster(self.dic_jo, phase, inputs[ii], dds[idx_max], uv_kps[ii])
                    self.dic_cnt[phase] += 1
                    boxes_gt.pop(idx_max)
                    dds.pop(idx_max)

        with open(self.path_joints, 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        for phase in ['train', 'val', 'test']:
            print("Saved {} annotations for phase {}"
                  .format(self.dic_cnt[phase], phase))
        print("Number of GT files: {}. Files not found: {}"
              .format(self.cnt_gt, self.cnt_fnf))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))



