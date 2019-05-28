

def distance_from_disparity(list_dds, list_kps):
    """Associate instances in left and right images and compute disparity"""

    dds = list_dds[0]
    dds_right = list_dds[1]
    kps = list_kps[0]
    kps_right = list_kps[1]
    dds_stereo = []

    for idx, dd in enumerate(dds):

        # Find the closest human in terms of distance
        idx_right, delta_min = match_distances(dd, dds_right)
        dd_stereo = calculate_disparity(kps[idx], kps_right[idx_right])
        dds_stereo.append(dd_stereo)

    stereo_dds = None
    return stereo_dds


def calculate_disparity(kp, kp_right):
    """From 2 sets of keypoints calculate disparity as the median of the disparities"""

    dd_stereo = None
    return dd_stereo


def match_distances(dd, dds_right):
    """Find index of the closest instance in the right image to the instance in the left image"""

    for idx, dd_right in enumerate(dds_right):
        delta_d_min = 1
        idx_min = 0

        delta_d = abs(dd - dd_right)
        if delta_d < delta_d_min:
            delta_d_min = delta_d
            idx_min = idx

    return idx_min, delta_d_min


