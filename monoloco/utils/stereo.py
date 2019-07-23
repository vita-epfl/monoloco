
import copy
import warnings

import numpy as np


def depth_from_disparity(zzs, kps, kps_right):
    """Associate instances in left and right images and compute disparity"""
    zzs_stereo = []
    zzs = np.array(zzs)
    kps = np.array(kps)
    kps_right_list = copy.deepcopy(kps_right)
    cnt_stereo = 0
    expected_disps = 0.54 * 721 / np.array(zzs)

    for idx, zz_mono in enumerate(zzs):
        if kps_right_list:

            zz_stereo, disparity_x, disparity_y, idx_min = filter_disparities(kps, kps_right_list, idx, expected_disps)

            if verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):
                zzs_stereo.append(zz_stereo)
                cnt_stereo += 1
                kps_right_list.pop(idx_min)
            else:
                zzs_stereo.append(zz_mono)
        else:
            zzs_stereo.append(zz_mono)

    return zzs_stereo, cnt_stereo


def filter_disparities(kps, kps_right_list, idx, expected_disps):
    """filter joints based on confidence and interquartile range of the distribution"""

    CONF_MIN = 0.3
    kps_right = np.array(kps_right_list)
    with warnings.catch_warnings() and np.errstate(invalid='ignore'):
        try:
            disparity_x = kps[idx, 0, :] - kps_right[:, 0, :]
            disparity_y = kps[idx, 1, :] - kps_right[:, 1, :]

            # Mask for low confidence
            mask_conf_left = kps[idx, 2, :] > CONF_MIN
            mask_conf_right = kps_right[:, 2, :] > CONF_MIN
            mask_conf = mask_conf_left & mask_conf_right
            disparity_x_conf = np.where(mask_conf, disparity_x, np.nan)
            disparity_y_conf = np.where(mask_conf, disparity_y, np.nan)

            # Mask outliers using iqr
            mask_outlier = interquartile_mask(disparity_x_conf)
            disparity_x_mask = np.where(mask_outlier, disparity_x_conf, np.nan)
            disparity_y_mask = np.where(mask_outlier, disparity_y_conf, np.nan)
            avg_disparity_x = np.nanmedian(disparity_x_mask, axis=1)  # ignore the nan
            diffs_x = [abs(expected_disps[idx] - real) for real in avg_disparity_x]
            idx_min = diffs_x.index(min(diffs_x))
            zz_stereo = 0.54 * 721. / float(avg_disparity_x[idx_min])

        except ZeroDivisionError:
            zz_stereo = - 100

        return zz_stereo, disparity_x_mask[idx_min], disparity_y_mask[idx_min], idx_min


def verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):

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
