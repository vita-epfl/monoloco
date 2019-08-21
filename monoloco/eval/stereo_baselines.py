
""""Generate stereo baselines for kitti evaluation"""

import copy
import warnings
from collections import defaultdict

import numpy as np

from ..utils import get_keypoints


def baselines_association(baselines, zzs, keypoints, keypoints_right, reid_features):
    """compute stereo depth for each of the given stereo baselines"""

    # Initialize variables
    zzs_stereo = defaultdict(list)
    cnt_stereo = defaultdict(int)

    features, features_r, keypoints, keypoints_r = factory_features(
        keypoints, keypoints_right, baselines, reid_features)

    # Iterate over each left pose
    for idx, zz_mono in enumerate(zzs):
        keypoint = keypoints[idx]

        for key in baselines:
            if keypoints_r[key]:

                # Filter joints disparity and calculate avg disparity
                avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoint, keypoints_r[key])

                # Extract features of the baseline
                similarity = features_similarity(features[key][idx], features_r[key], key, avg_disparities, zz_mono)

                # Compute the association based on features minimization and calculate depth
                zz_stereo, idx_min, flag = similarity_to_depth(similarity, avg_disparities)

                # Filter stereo depth
                if flag and verify_stereo(zz_stereo, zz_mono, disparities_x[idx_min], disparities_y[idx_min]):
                    zzs_stereo[key].append(zz_stereo)
                    cnt_stereo[key] += 1
                    keypoints_r[key].pop(idx_min)  # Update the keypoints for the next iteration
                    features_r[key].pop(idx_min)  # update available features for the next iteration
                else:
                    zzs_stereo[key].append(zz_mono)
            else:
                zzs_stereo[key].append(zz_mono)
    return zzs_stereo, cnt_stereo


def factory_features(keypoints, keypoints_right, baselines, reid_features):

    features = defaultdict()
    keypoints_r = defaultdict(list)  # dictionaries share memory as lists!
    features_r = defaultdict(list)
    keypoints = np.array(keypoints)

    for key in baselines:
        keypoints_r[key] = copy.deepcopy(keypoints_right)
        if key == 'reid':
            features[key] = np.array(reid_features[0])
            features_r[key] = reid_features[1].tolist()
        else:
            features[key] = copy.deepcopy(keypoints)
            features_r[key] = copy.deepcopy(keypoints_right)

    return features, features_r, keypoints, keypoints_r


def features_similarity(feature, features_r, key, avg_disparities, zz_mono):

    if key == 'ml_stereo':
        expected_disparity = 0.54 * 721. / zz_mono
        similarity = np.abs(expected_disparity - avg_disparities)
        return similarity

    elif key == 'pose':
        # Zero-center the keypoints
        uv_center = np.array(get_keypoints(feature, mode='center').reshape(-1, 1))  # (1, 2) --> (2, 1)
        uv_centers_r = np.array(get_keypoints(features_r, mode='center').unsqueeze(-1))  # (m,2) --> (m, 2, 1)
        feature = feature[:2, :] - uv_center
        feature = feature.reshape(1, -1)  # (1, 34)
        features_r = np.array(features_r)[:, :2, :] - uv_centers_r
        features_r = features_r.reshape(features_r.shape[0], -1)  # (m, 34)
        similarity = np.linalg.norm(feature - features_r, axis=1)

    if key == 'reid':
        similarity = np.linalg.norm(feature - features_r, axis=1)
    return similarity


def similarity_to_depth(features, avg_disparities):

    try:
        idx_min = int(np.nanargmin(features))
        zz_stereo = 0.54 * 721. / float(avg_disparities[idx_min])
        flag = True
    except (ZeroDivisionError, ValueError):  # All nan-slices or zero division
        zz_stereo = idx_min = 0
        flag = False

    return zz_stereo, idx_min, flag


def mask_joint_disparity(kps, keypoints_r):
    """filter joints based on confidence and interquartile range of the distribution"""

    CONF_MIN = 0.3
    keypoints_r = np.array(keypoints_r)

    with warnings.catch_warnings() and np.errstate(invalid='ignore'):

        disparity_x = kps[0, :] - keypoints_r[:, 0, :]
        disparity_y = kps[1, :] - keypoints_r[:, 1, :]

        # Mask for low confidence
        mask_conf_left = kps[2, :] > CONF_MIN
        mask_conf_right = keypoints_r[:, 2, :] > CONF_MIN
        mask_conf = mask_conf_left & mask_conf_right
        disparity_x_conf = np.where(mask_conf, disparity_x, np.nan)
        disparity_y_conf = np.where(mask_conf, disparity_y, np.nan)

        # Mask outliers using iqr
        mask_outlier = interquartile_mask(disparity_x_conf)
        disparity_x_mask = np.where(mask_outlier, disparity_x_conf, np.nan)
        disparity_y_mask = np.where(mask_outlier, disparity_y_conf, np.nan)
        avg_disparity = np.nanmedian(disparity_x_mask, axis=1)  # ignore the nan

        return avg_disparity, disparity_x_mask, disparity_y_mask


def verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):
    """Verify disparities based on coefficient of variation, maximum y difference and z difference wrt monoloco"""

    COV_MIN = 0.1
    y_max_difference = (50 / zz_mono)
    z_max_difference = 0.6 * zz_mono

    # COV_MIN = 20
    # y_max_difference = (1000 / zz_mono)
    # z_max_difference = 3 * zz_mono

    cov = float(np.nanstd(disparity_x) / np.abs(np.nanmean(disparity_x)))  # Coefficient of variation
    avg_disparity_y = np.nanmedian(disparity_y)

    if abs(zz_stereo - zz_mono) < z_max_difference and \
            avg_disparity_y < y_max_difference and \
            cov < COV_MIN:
        return True
    return False


def interquartile_mask(distribution):
    quartile_1, quartile_3 = np.nanpercentile(distribution, [25, 75], axis=1)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (distribution < upper_bound.reshape(-1, 1)) & (distribution > lower_bound.reshape(-1, 1))
