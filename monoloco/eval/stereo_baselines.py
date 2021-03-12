
""""Generate stereo baselines for kitti evaluation"""

from collections import defaultdict

import numpy as np

from ..utils import get_keypoints, mask_joint_disparity, disparity_to_depth


def baselines_association(baselines, zzs, keypoints, keypoints_right, reid_features):
    """compute stereo depth for each of the given stereo baselines"""

    # Initialize variables
    zzs_stereo = defaultdict()
    cnt_stereo = defaultdict(int)

    features, features_r, keypoints, keypoints_r = factory_features(
        keypoints, keypoints_right, baselines, reid_features)

    # count maximum possible associations
    cnt_stereo['max'] = min(keypoints.shape[0], keypoints_r.shape[0])  # pylint: disable=E1136

    # Filter joints disparity and calculate avg disparity
    avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoints, keypoints_r)

    # Iterate over each left pose
    for key in baselines:

        # Extract features of the baseline
        similarity = features_similarity(features[key], features_r[key], key, avg_disparities, zzs)

        # Compute the association based on features minimization and calculate depth
        zzs_stereo[key] = np.empty((keypoints.shape[0]))

        indices_stereo = []  # keep track of indices
        best = np.nanmin(similarity)
        while not np.isnan(best):
            idx, arg_best = np.unravel_index(np.nanargmin(similarity), similarity.shape)  # pylint: disable=W0632
            zz_stereo, flag = disparity_to_depth(avg_disparities[idx, arg_best])
            zz_mono = zzs[idx]
            similarity[idx, :] = np.nan
            indices_stereo.append(idx)

            # Filter stereo depth
            # if flag and verify_stereo(zz_stereo, zz_mono, disparities_x[idx, arg_best], disparities_y[idx, arg_best]):
            if flag and (1 < zz_stereo < 80):  # Do not add hand-crafted verifications to stereo baselines
                zzs_stereo[key][idx] = zz_stereo
                cnt_stereo[key] += 1
                similarity[:, arg_best] = np.nan
            else:
                zzs_stereo[key][idx] = zz_mono

            best = np.nanmin(similarity)
        indices_mono = [idx for idx, _ in enumerate(zzs) if idx not in indices_stereo]
        for idx in indices_mono:
            zzs_stereo[key][idx] = zzs[idx]
        zzs_stereo[key] = zzs_stereo[key].tolist()

    return zzs_stereo, cnt_stereo


def factory_features(keypoints, keypoints_right, baselines, reid_features):

    features = defaultdict()
    features_r = defaultdict()

    for key in baselines:
        if key == 'reid':
            features[key] = np.array(reid_features[0])
            features_r[key] = np.array(reid_features[1])
        else:
            features[key] = np.array(keypoints)
            features_r[key] = np.array(keypoints_right)

    return features, features_r, np.array(keypoints), np.array(keypoints_right)


def features_similarity(features, features_r, key, avg_disparities, zzs):

    similarity = np.empty((features.shape[0], features_r.shape[0]))
    for idx, zz_mono in enumerate(zzs):
        feature = features[idx]

        if key == 'ml_stereo':
            expected_disparity = 0.54 * 721. / zz_mono
            sim_row = np.abs(expected_disparity - avg_disparities[idx])

        elif key == 'pose':
            # Zero-center the keypoints
            uv_center = np.array(get_keypoints(feature, mode='center').reshape(-1, 1))  # (1, 2) --> (2, 1)
            uv_centers_r = np.array(get_keypoints(features_r, mode='center').unsqueeze(-1))  # (m,2) --> (m, 2, 1)
            feature_0 = feature[:2, :] - uv_center
            feature_0 = feature_0.reshape(1, -1)  # (1, 34)
            features_r_0 = features_r[:, :2, :] - uv_centers_r
            features_r_0 = features_r_0.reshape(features_r_0.shape[0], -1)  # (m, 34)
            sim_row = np.linalg.norm(feature_0 - features_r_0, axis=1)

        else:
            sim_row = np.linalg.norm(feature - features_r, axis=1)

        similarity[idx] = sim_row
    return similarity
