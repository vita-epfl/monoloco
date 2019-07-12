
import copy
import numpy as np


def depth_from_disparity(zzs, kps, kps_right):
    """Associate instances in left and right images and compute disparity"""
    MIN_CONF = 0.3
    zzs_stereo = []
    zzs = np.array(zzs)
    kps = np.array(kps)
    kps_right_list = copy.deepcopy(kps_right)
    cnt_stereo = 0
    expected_disps = 0.54 * 721 / np.array(zzs)

    for idx, zz_mono in enumerate(zzs):
        if kps_right_list:
            kps_right = np.array(kps_right_list)
            disparity_x = kps[idx, 0, :] - kps_right[:, 0, :]
            disparity_y = kps[idx, 1, :] - kps_right[:, 1, :]
            mask_left = kps[idx, 2, :] > MIN_CONF
            mask_right = kps_right[:, 2, :] > MIN_CONF
            mask = mask_left & mask_right
            disparity_x_masked = np.where(mask, disparity_x, np.nan)
            disparity_y_masked = np.where(mask, disparity_y, np.nan)
            avg_disparity_x = np.nanmedian(disparity_x_masked, axis=1)  # ignore the nan
            avg_disparity_y = np.nanmedian(disparity_y_masked, axis=1)
            diffs_x = [abs(expected_disps[idx] - real) for real in avg_disparity_x]
            idx_min = diffs_x.index(min(diffs_x))

            try:
                zz_stereo = 0.54 * 721. / float(avg_disparity_x[idx_min])
            except ZeroDivisionError:
                zz_stereo = 0

            if abs(zz_stereo - zz_mono) <= (0.4 * zz_mono) and avg_disparity_y[idx_min] < (30 / zz_mono):
                zzs_stereo.append(zz_stereo)
                cnt_stereo += 1
                kps_right_list.pop(idx_min)
            else:
                zzs_stereo.append(zz_mono)
        else:
            zzs_stereo.append(zz_mono)

    return zzs_stereo, cnt_stereo


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
