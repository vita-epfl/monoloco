
import copy
import numpy as np
import warnings
warnings.filterwarnings('error')


def depth_from_disparity(zzs, zzs_right, kps, kps_right):
    """Associate instances in left and right images and compute disparity"""

    zzs_stereo = []
    cnt = 0
    for idx, zz in enumerate(zzs):

        # Find the closest human in terms of distance
        zz_stereo, idx_min, delta_d_min = calculate_disparity(zz, zzs_right, kps[idx], kps_right)
        if delta_d_min < 1:
            zzs_stereo.append(zz_stereo)
            zzs_right.pop(idx_min)
            kps_right.pop(idx_min)
            cnt += 1
        else:
            zzs_stereo.append(zz)

    return zzs_stereo, cnt


def calculate_disparity(zz, zzs_right, kp, kps_right):
    """From 2 sets of keypoints calculate disparity as the median of the disparities"""

    kp = np.array(copy.deepcopy(kp))
    kps_right = np.array(copy.deepcopy(kps_right))
    zz_stereo = 0
    idx_min = 0
    delta_z_min = 4

    for idx, zz_right in enumerate(zzs_right):
        delta_z = abs(zz - zz_right)
        diffs = np.array(np.array(kp[0] - kps_right[idx][0]))
        diff = np.mean(diffs)

        # Check only for right instances (5 pxls = 80meters)
        if delta_z < delta_z_min and diff > 5:
            delta_z_min = delta_z
            idx_min = idx
            zzs = 0.54 * 721 / diffs
            zz_stereo = np.median(zzs[kp[2] > 0])

    return zz_stereo, idx_min, delta_z_min


