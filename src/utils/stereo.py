
import copy
import warnings

import numpy as np


def depth_from_disparity(zzs, kps, kps_right):
    """Associate instances in left and right images and compute disparity"""
    CONF_MIN = 0.3
    zzs_stereo = []
    zzs = np.array(zzs)
    kps = np.array(kps)
    kps_right_list = copy.deepcopy(kps_right)
    cnt_stereo = 0
    expected_disps = 0.54 * 721 / np.array(zzs)

    for idx, zz_mono in enumerate(zzs):
        if kps_right_list:
            with warnings.catch_warnings() and np.errstate(invalid='ignore'):
                try:
                    kps_right = np.array(kps_right_list)
                    disparity_x = kps[idx, 0, :] - kps_right[:, 0, :]
                    disparity_y = kps[idx, 1, :] - kps_right[:, 1, :]

                    # Mask for low confidence
                    mask_conf_left = kps[idx, 2, :] > CONF_MIN
                    mask_conf_right = kps_right[:, 2, :] > CONF_MIN
                    mask_conf = mask_conf_left & mask_conf_right
                    disparity_x_conf = np.where(mask_conf, disparity_x, np.nan)
                    disparity_y_conf = np.where(mask_conf, disparity_y, np.nan)

                    # Mask outliers using iqr
                    mask_outlier = get_iqr_mask(disparity_x_conf)
                    disparity_x_mask = np.where(mask_outlier, disparity_x_conf, np.nan)
                    disparity_y_mask = np.where(mask_outlier, disparity_y_conf, np.nan)
                    avg_disparity_x = np.nanmedian(disparity_x_mask, axis=1)  # ignore the nan
                    avg_disparity_y = np.nanmedian(disparity_y_mask, axis=1)
                    diffs_x = [abs(expected_disps[idx] - real) for real in avg_disparity_x]
                    idx_min = diffs_x.index(min(diffs_x))
                    zz_stereo = 0.54 * 721. / float(avg_disparity_x[idx_min])

                except ZeroDivisionError:
                    zz_stereo = - 100

            if abs(zz_stereo - zz_mono) <= (0.5 * zz_mono) and avg_disparity_y[idx_min] < (50 / zz_mono):
                zzs_stereo.append(zz_stereo)
                cnt_stereo += 1
                kps_right_list.pop(idx_min)
            else:
                zzs_stereo.append(zz_mono)
        else:
            zzs_stereo.append(zz_mono)

    return zzs_stereo, cnt_stereo


def get_iqr_mask(ys):
    quartile_1, quartile_3 = np.nanpercentile(ys, [25, 75], axis=1)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (ys < upper_bound.reshape(-1, 1)) & (ys > lower_bound.reshape(-1, 1))


# def depth_from_disparity_matricial(zzs, kps, kps_right):
#     """Associate instances in left and right images and compute disparity"""
#
#     zzs_stereo = copy.deepcopy(zzs)
#     zzs = np.array(zzs)
#     kps = np.array(kps)
#     kps_right = copy.deepcopy(kps_right)
#     cnt_stereo = 0
#
#     matrix_zz, matrix_diff = get_matrices_stereo(zzs, kps, kps_right)
#
#     indices = np.unravel_index(np.argmin(matrix_diff, axis=None), matrix_diff.shape)
#     z_diff = abs(zzs[indices[0]] - matrix_zz[indices])
#
#     while z_diff < 3 and matrix_diff[indices] < 100:
#
#         matrix_diff[indices[0], :] = 100
#         matrix_diff[:, indices[1]] = 100
#         zzs_stereo[indices[0]] = matrix_zz[indices]
#         indices = np.unravel_index(np.argmin(matrix_diff, axis=None), matrix_diff.shape)
#         z_diff = abs(zzs[indices[0]] - float(0.54 * 721 / matrix_diff[indices]))
#         cnt_stereo += 1
#
#     return zzs_stereo, cnt_stereo
#
#
# def get_matrices_stereo(zzs, kps, kps_right):
#
#     expected_disps = 0.54 * 721 / np.array(zzs)
#
#     matrix_diff = np.zeros((len(kps), len(kps_right)))
#     matrix_zz = np.zeros((len(kps), len(kps_right)))
#
#     for idx, zz in enumerate(zzs):
#         disparity_kp = kps[idx, 0, :] - np.array(kps_right)[:, 0, :]
#         avg_disparity = np.median(disparity_kp, axis=1)
#         matrix_zz[idx, :] = 0.54 * 721 / avg_disparity
#         matrix_diff[idx, :] = np.array([abs(expected_disps[idx] - real) for real in avg_disparity])
#
#     return matrix_zz, matrix_diff
