
import math
from copy import deepcopy
import numpy as np

BASELINE = 0.54
BF = BASELINE * 721

COCO_KEYPOINTS = [
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
]

HFLIP = {
    'nose': 'nose',
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


def transform_keypoints(keypoints, mode):
    """Egocentric horizontal flip"""
    assert mode == 'flip', "mode not recognized"
    kps = np.array(keypoints)
    dic_kps = {key: kps[:, :, idx] for idx, key in enumerate(COCO_KEYPOINTS)}
    kps_hflip = np.array([dic_kps[value] for key, value in HFLIP.items()])
    kps_hflip = np.transpose(kps_hflip, (1, 2, 0))
    return kps_hflip.tolist()


def flip_inputs(keypoints, im_w, mode=None):
    """Horizontal flip the keypoints or the boxes in the image"""
    if mode == 'box':
        boxes = deepcopy(keypoints)
        for box in boxes:
            temp = box[2]
            box[2] = im_w - box[0]
            box[0] = im_w - temp
        return boxes

    keypoints = np.array(keypoints)
    keypoints[:, 0, :] = im_w - keypoints[:, 0, :]  # Shifted
    kps_flip = transform_keypoints(keypoints, mode='flip')
    return kps_flip


def flip_labels(boxes_gt, labels, im_w):
    """Correct x, d positions and angles after horizontal flipping"""
    from ..utils import correct_angle, to_cartesian, to_spherical
    boxes_flip = deepcopy(boxes_gt)
    labels_flip = deepcopy(labels)

    for idx, label_flip in enumerate(labels_flip):

        # Flip the box and account for disparity
        disp = BF / label_flip[2]
        temp = boxes_flip[idx][2]
        boxes_flip[idx][2] = im_w - boxes_flip[idx][0] + disp
        boxes_flip[idx][0] = im_w - temp + disp

        # Flip X and D
        rtp = label_flip[3:4] + label_flip[0:2]  # Originally t,p,z,r
        xyz = to_cartesian(rtp)
        xyz[0] = -xyz[0] + BASELINE  # x
        rtp_r = to_spherical(xyz)
        label_flip[3], label_flip[0], label_flip[1] = rtp_r[0], rtp_r[1], rtp_r[2]

        # FLip and correct the angle
        yaw = label_flip[9]
        yaw_n = math.copysign(1, yaw) * (np.pi - abs(yaw))  # Horizontal flipping change of angle

        sin, cos, yaw_corr = correct_angle(yaw_n, xyz)
        label_flip[7], label_flip[8], label_flip[9] = sin, cos, yaw_n

    return boxes_flip, labels_flip


def height_augmentation(kps, kps_r, label, s_match, seed=0):
    """
    label: theta, psi, z, rho, wlh, sin, cos, yaw, cat
    """
    from ..utils import to_cartesian
    n_labels = 3 if s_match > 0.9 else 1
    height_min = 1.2
    height_max = 2
    av_height = 1.71
    kps_aug = [[kps.clone(), kps_r.clone()] for _ in range(n_labels+1)]
    labels_aug = [label.copy() for _ in range(n_labels+1)]  # Maintain the original
    np.random.seed(seed)
    heights = np.random.uniform(height_min, height_max, n_labels)  # 3 samples
    zzs = heights * label[2] / av_height
    disp = BF / label[2]

    rtp = label[3:4] + label[0:2]  # Originally t,p,z,r
    xyz = to_cartesian(rtp)

    for i in range(n_labels):

        if zzs[i] < 2:
            continue
        # Update keypoints
        disp_new = BF / zzs[i]
        delta_disp = disp - disp_new
        kps_aug[i][1][0, 0, :] = kps_aug[i][1][0, 0, :] + delta_disp

        # Update labels
        labels_aug[i][2] = zzs[i]
        xyz[2] = zzs[i]
        rho = np.linalg.norm(xyz)
        labels_aug[i][3] = rho

    return kps_aug, labels_aug
