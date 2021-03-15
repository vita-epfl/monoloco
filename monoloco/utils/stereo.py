
import warnings

import numpy as np


BF = 0.54 * 721
z_min = 4
z_max = 60
D_MIN = BF / z_max
D_MAX = BF / z_min


def extract_stereo_matches(keypoint, keypoints_r, zz, phase='train', seed=0, method=None):
    """Return binaries representing the match between the pose in the left and the ones in the right"""

    stereo_matches = []
    cnt_ambiguous = 0
    if method == 'mask':
        conf_min = 0.1
    else:
        conf_min = 0.2
    avgs_x_l, avgs_x_r, disparities_x, disparities_y = average_locations(keypoint, keypoints_r, conf_min=conf_min)
    avg_disparities = [abs(float(l) - BF / zz - float(r)) for l, r in zip(avgs_x_l, avgs_x_r)]
    idx_matches = np.argsort(avg_disparities)
    # error_max_stereo = 1 * 0.0028 * zz**2 + 0.2  # 2m at 20 meters of depth + 20 cm of offset
    error_max_stereo = 0.2 * zz + 0.2  # 2m at 20 meters of depth + 20 cm of offset
    error_min_mono = 0.25 * zz + 0.2
    error_max_mono = 1 * zz + 0.5
    used = []
    # Add positive and negative samples
    for idx, idx_match in enumerate(idx_matches):
        match = avg_disparities[idx_match]
        zz_stereo, flag = disparity_to_depth(match + BF / zz)

        # Conditions to accept stereo match
        conditions = (idx == 0
                      and match < depth_to_pixel_error(zz, depth_error=error_max_stereo)
                      and flag
                      and verify_stereo(zz_stereo, zz, disparities_x[idx_match], disparities_y[idx_match]))

        # Positive matches
        if conditions:
            stereo_matches.append((idx_match, 1))
        # Ambiguous
        elif match < depth_to_pixel_error(zz, depth_error=error_min_mono):
            cnt_ambiguous += 1

        # Disparity-range negative
        # elif D_MIN < match + BF / zz < D_MAX:
        #     stereo_matches.append((idx_match, 0))

        elif phase == 'val' \
                and match < depth_to_pixel_error(zz, depth_error=error_max_mono) \
                and not stereo_matches\
                and zz < 40:
            stereo_matches.append((idx_match, 0))

        # # Hard-negative for training
        elif phase == 'train' \
                and match < depth_to_pixel_error(zz, depth_error=error_max_mono) \
                and len(stereo_matches) < 3:
            stereo_matches.append((idx_match, 0))

        # # Easy-negative
        elif phase == 'train' \
                and len(stereo_matches) < 3:
            np.random.seed(seed + idx)
            num = np.random.randint(idx, len(idx_matches))
            if idx_matches[num] not in used:
                stereo_matches.append((idx_matches[num], 0))

        # elif len(stereo_matches) < 1:
        #     stereo_matches.append((idx_match, 0))

        # Easy-negative
        # elif len(idx_matches) > len(stereo_matches):
        #         stereo_matches.append((idx_matches[-1], 0))
        #         break  # matches are ordered
        else:
            break
        used.append(idx_match)

    # Make sure each left has at least a negative match
    # if not stereo_matches:
    #     stereo_matches.append((idx_matches[0], 0))
    return stereo_matches, cnt_ambiguous


def depth_to_pixel_error(zz, depth_error=1):
    """
    Calculate the pixel error at a certain depth due to depth error according to:
    e_d = b * f * e_z / (z**2)
    """
    e_d = BF * depth_error / (zz**2)
    return e_d


def mask_joint_disparity(keypoints, keypoints_r):
    """filter joints based on confidence and interquartile range of the distribution"""
    # TODO Merge with average location
    CONF_MIN = 0.3
    with warnings.catch_warnings() and np.errstate(invalid='ignore'):
        disparity_x_mask = np.empty((keypoints.shape[0], keypoints_r.shape[0], 17))
        disparity_y_mask = np.empty((keypoints.shape[0], keypoints_r.shape[0], 17))
        avg_disparity = np.empty((keypoints.shape[0], keypoints_r.shape[0]))

        for idx, kps in enumerate(keypoints):
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
            x_mask_row = np.where(mask_outlier, disparity_x_conf, np.nan)
            y_mask_row = np.where(mask_outlier, disparity_y_conf, np.nan)
            avg_row = np.nanmedian(x_mask_row, axis=1)  # ignore the nan

            # Append
            disparity_x_mask[idx] = x_mask_row
            disparity_y_mask[idx] = y_mask_row
            avg_disparity[idx] = avg_row

        return avg_disparity, disparity_x_mask, disparity_y_mask


def average_locations(keypoint, keypoints_r, conf_min=0.2):
    """
    Extract absolute average location of keypoints
    INPUT: arrays of (1, 3, 17) & (m,3,17)
    OUTPUT: 2 arrays of (m).
    The left keypoint will have different absolute positions based on the right keypoints they are paired with
    """
    keypoint, keypoints_r = np.array(keypoint), np.array(keypoints_r)
    assert keypoints_r.shape[0] > 0, "No right keypoints"
    with warnings.catch_warnings() and np.errstate(invalid='ignore'):

        # Mask by confidence
        mask_l_conf = keypoint[0, 2, :] > conf_min
        mask_r_conf = keypoints_r[:, 2, :] > conf_min
        abs_x_l = np.where(mask_l_conf, keypoint[0, 0:1, :], np.nan)
        abs_x_r = np.where(mask_r_conf, keypoints_r[:, 0, :], np.nan)

        # Mask by iqr
        mask_l_iqr = interquartile_mask(abs_x_l)
        mask_r_iqr = interquartile_mask(abs_x_r)

        # Combine masks
        mask = mask_l_iqr & mask_r_iqr

        # Compute absolute locations and relative disparities
        x_l = np.where(mask, abs_x_l, np.nan)
        x_r = np.where(mask, abs_x_r, np.nan)
        x_disp = x_l - x_r
        y_disp = np.where(mask, keypoint[0, 1, :] - keypoints_r[:, 1, :], np.nan)
        avgs_x_l = np.nanmedian(x_l, axis=1)
        avgs_x_r = np.nanmedian(x_r, axis=1)

        return avgs_x_l, avgs_x_r, x_disp, y_disp


def interquartile_mask(distribution):
    quartile_1, quartile_3 = np.nanpercentile(distribution, [25, 75], axis=1)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (distribution < upper_bound.reshape(-1, 1)) & (distribution > lower_bound.reshape(-1, 1))


def disparity_to_depth(avg_disparity):

    try:
        zz_stereo = 0.54 * 721. / float(avg_disparity)
        flag = True
    except (ZeroDivisionError, ValueError):  # All nan-slices or zero division
        zz_stereo = np.nan
        flag = False
    return zz_stereo, flag


def verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):
    """Verify disparities based on coefficient of variation, maximum y difference and z difference wrt monoloco"""

    # COV_MIN = 0.1
    y_max_difference = (80 / zz_mono)
    z_max_difference = 1 * zz_mono

    cov = float(np.nanstd(disparity_x) / np.abs(np.nanmean(disparity_x)))  # Coefficient of variation
    avg_disparity_y = np.nanmedian(disparity_y)

    return abs(zz_stereo - zz_mono) < z_max_difference and avg_disparity_y < y_max_difference and 1 < zz_stereo < 80
# cov < COV_MIN and \
