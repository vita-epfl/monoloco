import numpy as np
import os
import sys
import time
import json
import logging
from collections import defaultdict
import datetime


class PreprocessNuscenes:
    """
    Preprocess Nuscenes dataset
    """
    def __init__(self, dir_ann, dir_nuscenes, dataset, iou_min=0.3):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dir_ann = dir_ann
        dir_out = os.path.join('data', 'arrays')
        assert os.path.exists(dir_nuscenes), "Nuscenes directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-' + dataset + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + dataset + '-' + now_time + '.json')

        self.iou_min = iou_min

        # Import functions
        from utils.misc import get_idx_max, append_cluster
        self.get_idx_max = get_idx_max
        self.append_cluster = append_cluster
        from utils.nuscenes import select_categories
        self.select_categories = select_categories
        from utils.camera import project_3d
        self.project_3d = project_3d
        from utils.pifpaf import get_input_data, preprocess_pif
        self.get_input_data = get_input_data
        self.preprocess_pif = preprocess_pif
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils import splits
        self.splits = splits

        # Initialize dicts to save joints for training
        self.dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                                     clst=defaultdict(lambda: defaultdict(list))),
                       'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[],  K=[],
                                   clst=defaultdict(lambda: defaultdict(list))),
                       'test': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                                    clst=defaultdict(lambda: defaultdict(list)))
                       }
        # Names as keys to retrieve it easily
        self.dic_names = defaultdict(lambda: defaultdict(list))

        self.cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        # Split training and validation base on the dataset type
        if dataset == 'nuscenes':
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dir_nuscenes, verbose=True)
            self.scenes = self.nusc.scene
            split_scenes = self.splits.create_splits_scenes()
            self.split_train, self.split_val = split_scenes['train'], split_scenes['val']

        elif dataset == 'nuscenes_mini':
            self.nusc = NuScenes(version='v1.0-mini', dataroot=dir_nuscenes, verbose=True)
            self.scenes = self.nusc.scene
            split_scenes = self.splits.create_splits_scenes()
            self.split_train, self.split_val = split_scenes['train'], split_scenes['val']

        elif dataset == 'nuscenes_teaser':
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dir_nuscenes, verbose=True)
            with open("splits/nuscenes_teaser_scenes.txt", "r") as ff:
                teaser_scenes = ff.read().splitlines()
            self.scenes = self.nusc.scene
            self.scenes = [scene for scene in self.scenes if scene['token'] in teaser_scenes]
            with open("splits/split_nuscenes_teaser.json", "r") as ff:
                dic_split = json.load(ff)
            self.split_train = [scene['name'] for scene in self.scenes if scene['token'] in dic_split['train']]
            self.split_val = [scene['name'] for scene in self.scenes if scene['token'] in dic_split['val']]

    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes = 0
        cnt_samples = 0
        cnt_sd = 0
        cnt_ann = 0
        start = time.time()

        for ii, scene in enumerate(self.scenes):
            end_scene = time.time()
            current_token = scene['first_sample_token']
            cnt_scenes += 1
            if ii == 0:
                time_left = "Nan"
            else:
                time_left = str((end_scene-start_scene)/60 * (len(self.scenes) - ii))[:4]

            sys.stdout.write('\r' + 'Elaborating scene {}, remaining time {} minutes'.format(cnt_scenes, time_left) + '\t\n')
            start_scene = time.time()
            if scene['name'] in self.split_train:
                phase = 'train'
            elif scene['name'] in self.split_val:
                phase = 'val'
            else:
                print("phase name not in training or validation split")
                continue

            while not current_token == "":
                sample_dic = self.nusc.get('sample', current_token)
                cnt_samples += 1

                # Extract all the sample_data tokens for each sample
                for cam in self.cameras:
                    sd_token = sample_dic['data'][cam]
                    cnt_sd += 1
                    path_im, boxes_obj, kk = self.nusc.get_sample_data(sd_token, box_vis_level=1)  # At least one corner

                    # Extract all the annotations of the person
                    boxes_gt = []
                    dds = []
                    boxes_3d = []
                    name = os.path.basename(path_im)
                    for box_obj in boxes_obj:
                        if box_obj.name[:6] != 'animal':
                            general_name = box_obj.name.split('.')[0] + '.' + box_obj.name.split('.')[1]
                        else:
                            general_name = 'animal'
                        if general_name in self.select_categories('all'):
                            box = self.project_3d(box_obj, kk)
                            dd = np.linalg.norm(box_obj.center)
                            boxes_gt.append(box)
                            dds.append(dd)
                            box_3d = box_obj.center.tolist() + box_obj.wlh.tolist()
                            boxes_3d.append(box_3d)
                            self.dic_names[name]['boxes'].append(box)
                            self.dic_names[name]['dds'].append(dd)
                            self.dic_names[name]['K'] = kk.tolist()

                    # Run IoU with pifpaf detections and save
                    path_pif = os.path.join(self.dir_ann, name + '.pifpaf.json')
                    exists = os.path.isfile(path_pif)

                    if exists:
                        with open(path_pif, 'r') as f:
                            annotations = json.load(f)

                        boxes, keypoints = self.preprocess_pif(annotations, im_size=None)
                        (inputs, xy_kps), (uv_kps, uv_boxes, _, _) = self.get_input_data(boxes, keypoints, kk)

                        for ii, box in enumerate(uv_boxes):
                            idx_max, iou_max = self.get_idx_max(box, boxes_gt)

                            if iou_max > self.iou_min:

                                self.dic_jo[phase]['kps'].append(uv_kps[ii])
                                self.dic_jo[phase]['X'].append(inputs[ii])
                                self.dic_jo[phase]['Y'].append([dds[idx_max]])  # Trick to make it (nn,1)
                                self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                                self.dic_jo[phase]['boxes_3d'].append(boxes_3d[idx_max])
                                self.dic_jo[phase]['K'] = kk.tolist()
                                self.append_cluster(self.dic_jo, phase, inputs[ii], dds[idx_max], uv_kps[ii])
                                boxes_gt.pop(idx_max)
                                dds.pop(idx_max)
                                boxes_3d.pop(idx_max)
                                cnt_ann += 1
                                sys.stdout.write('\r' + 'Saved annotations {}'
                                                 .format(cnt_ann) + '\t')

                current_token = sample_dic['next']

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        print("\nSaved {} annotations for {} samples in {} scenes. Total time: {:.1f} minutes"
              .format(cnt_ann, cnt_samples, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))
