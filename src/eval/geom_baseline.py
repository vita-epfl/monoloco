
import glob
import json
import logging
import os
import numpy as np
import math
from collections import defaultdict
from utils.camera import pixel_to_camera


class GeomBaseline:

    def __init__(self, joints):

        # Initialize directories
        self.clusters = ['10', '20', '30', '>30', 'all']
        self.average_y = 0.48
        self.joints = joints

        from utils.misc import calculate_iou
        self.calculate_iou = calculate_iou
        from utils.nuscenes import get_unique_tokens, split_scenes
        self.get_unique_tokens = get_unique_tokens
        self.split_scenes = split_scenes

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        List of json files --> 2 lists with mean and std for each segment and the total count of instances

        For each annotation:
        1. From gt boxes calculate the height (deltaY) for the segments head, shoulder, hip, ankle
        2. From mask boxes calculate distance of people using average height of people and real pixel height

        For left-right ambiguities we chose always the average of the joints

        The joints are mapped from 0 to 16 in the following order:
        ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
        'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        'right_ankle']

        """
        cnt_tot = 0
        dic_dist = defaultdict(lambda: defaultdict(list))

        # Access the joints file
        with open(self.joints, 'r') as ff:
            dic_joints = json.load(ff)

        # Calculate distances for all the segments
        for phase in ['train', 'val']:
            cnt = update_distances(dic_joints[phase], dic_dist, phase, self.average_y)
            cnt_tot += cnt

        dic_h_means = calculate_heights(dic_dist['heights'], mode='mean')
        dic_h_stds = calculate_heights(dic_dist['heights'], mode='std')

        self.logger.info("Computed distance of {} annotations".format(cnt_tot))

        for key in dic_h_means:
            self.logger.info("Average height of segment {} is {:.2f} with a std of {:.2f}".
                             format(key, dic_h_means[key], dic_h_stds[key]))

        errors = calculate_error(dic_dist['error'])

        for clst in self.clusters:
            self.logger.info("Average distance over the val set for clst {}: {:.2f}".format(clst, errors[clst]))

        self.logger.info("Joints used: {}".format(self.joints))


def update_distances(dic_fin, dic_dist, phase, average_y):

    # Loop over each annotation in the json file corresponding to the image

    cnt = 0
    for idx, kps in enumerate(dic_fin['kps']):
        # Extract pixel coordinates of head, shoulder, hip, ankle and and save them
        dic_uv = extract_pixel_coord(kps)

        # Convert segments from pixel coordinate to camera coordinate
        kk = dic_fin['K'][idx]
        z_met = dic_fin['boxes_3d'][idx][2]

        # Create a dict with all annotations in meters
        dic_xyz = {key: pixel_to_camera(dic_uv[key], kk, z_met) for key in dic_uv}

        # Compute real height
        dy_met = abs(dic_xyz['hip'][1] - dic_xyz['shoulder'][1])

        # Estimate distance for a single annotation
        z_met_real, _ = compute_distance_single(dic_uv['shoulder'], dic_uv['hip'], kk, average_y,
                                                mode='real', dy_met=dy_met)
        z_met_approx, _ = compute_distance_single(dic_uv['shoulder'], dic_uv['hip'], kk, average_y,
                                                  mode='average')

        # Compute distance with respect to the center of the 3D bounding box
        xyz_met = np.array(dic_fin['boxes_3d'][idx][0:3])
        d_met = np.linalg.norm(xyz_met)
        d_real = math.sqrt(z_met_real ** 2 + dic_fin['boxes_3d'][idx][0] ** 2 + dic_fin['boxes_3d'][idx][1] ** 2)
        d_approx = math.sqrt(z_met_approx ** 2 +
                             dic_fin['boxes_3d'][idx][0] ** 2 + dic_fin['boxes_3d'][idx][1] ** 2)

        # if abs(d_qmet - d_real) > 1e-1:  # "Error in computing distance with real height in pixels"
        #     aa = 5

        # Update the dictionary with distance and heights metrics
        dic_dist = update_dic_dist(dic_dist, dic_xyz, d_real, d_approx, phase)
        cnt += 1

    return cnt


def compute_distance_single(uv_1, uv_2, kk, average_y, mode='average', dy_met=0):

    """
    Compute distance Z of a mask annotation (solving a linear system) for 2 possible cases:
    1. knowing specific height of the annotation (head-ankle) dy_met
    2. using mean height of people (average_y)
    """
    assert mode == 'average' or mode == 'real'
    # Trasform into normalized camera coordinates (plane at 1m)
    xyz_met_norm_1 = pixel_to_camera(uv_1, kk, 1)
    xyz_met_norm_2 = pixel_to_camera(uv_2, kk, 1)

    x1 = xyz_met_norm_1[0]
    y1 = xyz_met_norm_1[1]
    x2 = xyz_met_norm_2[0]
    y2 = xyz_met_norm_2[1]
    xx = (x1 + x2) / 2

    # Choose if solving for provided height or average one.
    if mode == 'average':
        cc = - average_y  # Y axis goes down
    else:
        cc = -dy_met

    # if - 3 * average_y <= cc <= -2:
    #     aa = 5

    # Solving the linear system Ax = b
    Aa = np.array([[y1, 0, -xx],
                  [0, -y1, 1],
                  [y2, 0, -xx],
                  [0, -y2, 1]])

    bb = np.array([cc * xx, -cc, 0, 0]).reshape(4, 1)
    xx = np.linalg.lstsq(Aa, bb, rcond=None)
    z_met = abs(np.float(xx[0][1]))  # Abs take into account specularity behind the observer

    # Compute the absolute x and y coordinates in meters
    xyz_met_1 = xyz_met_norm_1 * z_met
    xyz_met_2 = xyz_met_norm_2 * z_met

    return z_met, (xyz_met_1, xyz_met_2)


def extract_pixel_coord(kps):

    """Extract uv coordinates from keypoints and save them in a dict """
    # For each level of height (e.g. 5 points in the head), take the average of them

    uv_head = np.array([np.average(kps[0][0:5]), np.average(kps[1][0:5]), 1])
    uv_shoulder = np.array([np.average(kps[0][5:7]), np.average(kps[1][5:7]), 1])
    uv_hip = np.array([np.average(kps[0][11:13]), np.average(kps[1][11:13]), 1])
    uv_ankle = np.array([np.average(kps[0][15:17]), np.average(kps[1][15:17]), 1])

    dic_uv = {'head': uv_head, 'shoulder': uv_shoulder, 'hip': uv_hip, 'ankle': uv_ankle}

    return dic_uv


def update_dic_dist(dic_dist, dic_xyz, d_real, d_approx, phase):
    """ For every annotation in a single image, update the final dictionary"""

    # Update the dict with heights metric
    if phase == 'train':
        dic_dist['heights']['head'].append(np.float(dic_xyz['head'][1]))
        dic_dist['heights']['shoulder'].append(np.float(dic_xyz['shoulder'][1]))
        dic_dist['heights']['hip'].append(np.float(dic_xyz['hip'][1]))
        dic_dist['heights']['ankle'].append(np.float(dic_xyz['ankle'][1]))

    # Update the dict with distance metrics for the test phase
    if phase == 'val':
        error = abs(d_real - d_approx)

        if d_real <= 10:
            dic_dist['error']['10'].append(error)
        elif d_real <= 20:
            dic_dist['error']['20'].append(error)
        elif d_real <= 30:
            dic_dist['error']['30'].append(error)
        else:
            dic_dist['error']['>30'].append(error)

        dic_dist['error']['all'].append(error)

    return dic_dist


def calculate_heights(heights, mode):
    """
     Compute statistics of heights based on the distance
     """

    assert mode == 'mean' or mode == 'std' or mode == 'max'
    heights_fin = {}

    head_shoulder = np.array(heights['shoulder']) - np.array(heights['head'])
    shoulder_hip = np.array(heights['hip']) - np.array(heights['shoulder'])
    hip_ankle = np.array(heights['ankle']) - np.array(heights['hip'])

    if mode == 'mean':
        heights_fin['head_shoulder'] = np.float(np.mean(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.mean(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.mean(hip_ankle)) * 100

    elif mode == 'std':
        heights_fin['head_shoulder'] = np.float(np.std(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.std(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.std(hip_ankle)) * 100

    elif mode == 'max':
        heights_fin['head_shoulder'] = np.float(np.max(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.max(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.max(hip_ankle)) * 100

    return heights_fin


def calculate_error(dic_errors):
    """
     Compute statistics of distances based on the distance
     """

    errors = {}
    for clst in dic_errors:

        errors[clst] = np.float(np.mean(np.array(dic_errors[clst])))

    return errors

