
import copy
import numpy as np


def depth_from_disparity(dds, dds_right, kps, kps_right):
    """Associate instances in left and right images and compute disparity"""

    zzs_stereo = []
    for idx, dd in enumerate(dds):

        # Find the closest human in terms of distance
        zz_stereo, idx_min, delta_d_min = calculate_disparity(dd, dds_right, kps[idx], kps_right)
        if delta_d_min < 1:
            zzs_stereo.append(zz_stereo)
            dds_right.pop(idx_min)
            kps_right.pop(idx_min)

    return zzs_stereo


def calculate_disparity(dd, dds_right, kp, kps_right):
    """From 2 sets of keypoints calculate disparity as the median of the disparities"""

    kp = np.array(copy.deepcopy(kp))
    kps_right = np.array(copy.deepcopy(kps_right))
    zz_stereo = 0
    idx_min = 0
    delta_d_min = 1

    for idx, dd_right in enumerate(dds_right):
        delta_d = abs(dd - dd_right)
        diffs = np.array(np.array(kp[0] - kps_right[idx][0]))
        diff = np.mean(diffs)

        if delta_d < delta_d_min and diff > 0:  # Check only for right instances
            delta_d_min = delta_d
            idx_min = idx
            zzs = 0.54 * 721 / diffs
            zz_stereo = np.median(zzs[kp[2] > 0])

    return zz_stereo, idx_min, delta_d_min


