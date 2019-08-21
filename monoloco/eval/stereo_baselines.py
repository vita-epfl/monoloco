
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
    keypoints_r = defaultdict(list)  # dictionaries share memory as lists!
    cnt_stereo = defaultdict(int)
    keypoints = np.array(keypoints)
    for key in baselines:
        keypoints_r[key] = copy.deepcopy(keypoints_right)

    # Iterate over each left pose
    for idx, zz_mono in enumerate(zzs):
        keypoint = keypoints[idx]

        for key in baselines:
            if keypoints_r[key]:

                # Filter joints disparity and calculate avg disparity
                avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoint, keypoints_r[key])

                # Extract features of the baseline
                if key is 'reid':
                    features = reid_features[idx]
                elif key is 'pose':
                    features = l2_distance(keypoints, keypoints_r)
                else:
                    features = ml_stereo_features(zz_mono, avg_disparities)

                # Compute the association based on features minimization and calculate depth
                zz_stereo, idx_min = features_minimization(features, avg_disparities)

                # Filter stereo depth
                if verify_stereo(zz_stereo, zz_mono, disparities_x[idx_min], disparities_y[idx_min]):
                    zzs_stereo[key].append(zz_stereo)
                    cnt_stereo[key] += 1
                    keypoints_r[key].pop(idx_min)  # Update the keypoints for the next iteration
                else:
                    zzs_stereo[key].append(zz_mono)
            else:
                zzs_stereo[key].append(zz_mono)
    return zzs_stereo


def features_minimization(features, avg_disparities):

    idx_min = int(np.argmin(features))
    try:
        zz_stereo = 0.54 * 721. / float(avg_disparities[idx_min])
    except ZeroDivisionError:
        print("Warning: ZeroDivisionError")
        zz_stereo = 0

    return zz_stereo, idx_min


def ml_stereo_features(zz_mono, avg_disparities):
    """compute distance of each average disparity from the expected disparity based on monoloco distance"""

    expected_disparity = np.array(0.54 * 721. / zz_mono).resize((1, 1))
    features = np.abs(expected_disparity - avg_disparities)
    return features


def l2_distance(keypoints, keypoints_r):
    """
    Calculate a matrix/vector of l2 similarities for left-right instances in a single image
    from representation vectors of (possibly) different dimensions
    keypoints = (m, 17, 3) or (17,2)
    keypoints_r = (m, 17, 3)
    """
    # Zero-center the keypoints
    uv_centers = np.array(get_keypoints(keypoints, mode='center').unsqueeze(-1))
    uv_centers_r = np.array(get_keypoints(keypoints_r, mode='center').unsqueeze(-1))

    keypoints = np.array(keypoints)
    if len(keypoints.shape) == 2:
        keypoints = keypoints.reshape(1, keypoints.shape[0], keypoints.shape[1])
    keypoints_0 = keypoints[:, :2, :] - uv_centers
    keypoints_r_0 = np.array(keypoints_r)[:, :2, :] - uv_centers_r

    matrix = np.empty((keypoints_0.shape[0], keypoints_r_0.shape[0]))
    for idx, kps in enumerate(keypoints_0):
        for idx_r, kps_r in enumerate(keypoints_r_0):
            l2_norm = np.linalg.norm(kps.reshape(-1) - kps_r.reshape(-1))
            matrix[idx, idx_r] = l2_norm

    return matrix


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
