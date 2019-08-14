
import copy
import warnings

import numpy as np


from ..utils import get_keypoints


def l2_distance(keypoints, keypoints_r):
    """
    Calculate a matrix of cosine similarities for left-right instances in a single image
    from representation vectors of (possibly) different dimensions
    """
    # Zero-center the keypoints
    uv_centers = np.array(get_keypoints(keypoints, mode='center').unsqueeze(-1))
    uv_centers_r = np.array(get_keypoints(keypoints_r, mode='center').unsqueeze(-1))

    keypoints_0 = np.array(keypoints)[:, :2, :] - uv_centers
    keypoints_r_0 = np.array(keypoints_r)[:, :2, :] - uv_centers_r

    matrix = np.empty((keypoints_0.shape[0], keypoints_r_0.shape[0]))
    for idx, kps in enumerate(keypoints_0):
        for idx_r, kps_r in enumerate(keypoints_r_0):
            l2_norm = np.linalg.norm(kps.reshape(-1) - kps_r.reshape(-1))
            matrix[idx, idx_r] = l2_norm

    return matrix


def similarity_to_depth(vector_similarity, avg_disparities):
    """
    from a matrix of cosine distances, calculate corresponding depths
    one depth from every left instance. If similarity too low or no similarity, z=0
    """
    idx_min = np.argmin(vector_similarity)
    disparity = avg_disparities[idx_min]
    zz_pose = 0.54 * 721. / float(avg_disparities[idx_min])
    return zz_pose


def pose_baseline(zzs, keypoints, keypoints_r):

    cnt_stereo = 0
    zzs_stereo = []
    keypoints_r_list = copy.deepcopy(keypoints_r)
    zzs = np.array(zzs)
    matrix_similarity = l2_distance(keypoints, keypoints_r)
    for idx, zz_mono in enumerate(zzs):
        avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoints[idx], keypoints_r)
        zz_stereo, idx_min = similarity_to_depth(matrix_similarity[idx], avg_disparities)

        if verify_stereo(zz_stereo, zz_mono, disparities_x[idx_min], disparities_y[idx_min]):
            zzs_stereo.append(zz_stereo)
            cnt_stereo += 1
            keypoints_r_list.pop(idx_min)
        else:
            zzs_stereo.append(zz_mono)
    else:
        zzs_stereo.append(zz_mono)
    assert len(zzs_stereo) == len(zzs)
    return zzs_stereo, cnt_stereo


def monoloco_stereo(zzs, keypoints, keypoints_r):
    """
    Associate instances in left and right images and compute disparity using monoloco prior
    """
    cnt_stereo = 0
    zzs_stereo = []
    zzs = np.array(zzs)
    keypoints = np.array(keypoints)
    keypoints_r_list = copy.deepcopy(keypoints_r)

    for idx, zz_mono in enumerate(zzs):
        if keypoints_r_list:
            avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoints[idx], keypoints_r_list)

            zz_stereo, idx_min = depth_from_monoloco_disparity(zz_mono, avg_disparities)

            if verify_stereo(zz_stereo, zz_mono, disparities_x[idx_min], disparities_y[idx_min]):
                zzs_stereo.append(zz_stereo)
                cnt_stereo += 1
                keypoints_r_list.pop(idx_min)
            else:
                zzs_stereo.append(zz_mono)
        else:
            zzs_stereo.append(zz_mono)
    assert len(zzs_stereo) == len(zzs)
    return zzs_stereo, cnt_stereo


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


def depth_from_monoloco_disparity(zz_mono, avg_disparities):
    """Use monoloco depth as prior for expected disparity"""
    expected_disparity = 0.54 * 721. / zz_mono

    try:
        diffs_x = [abs(expected_disparity - real) for real in avg_disparities]
        idx_min = diffs_x.index(min(diffs_x))
        zz_stereo = 0.54 * 721. / float(avg_disparities[idx_min])
    except ZeroDivisionError:
        zz_stereo = - 100

    return zz_stereo, idx_min


def verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):
    """Verify disparities based on coefficient of variation, maximum y difference and z difference wrt monoloco"""

    COV_MIN = 0.1
    y_max_difference = (50 / zz_mono)
    z_max_difference = 0.6 * zz_mono

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


def calculate_monoloco_disparity(zzs):

    disparities = 0.54 * 721 / np.array(zzs)
    return disparities
