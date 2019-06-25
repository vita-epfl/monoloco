
import numpy as np
import math
import torch
import torch.nn.functional as F

def pixel_to_camera(uv1, kk, z_met):
    """
    (3,) array --> (3,) array
    Convert a point in pixel coordinate to absolute camera coordinates
    """

    kk_1 = np.linalg.inv(kk)
    xyz_met_norm = np.dot(kk_1, uv1)
    xyz_met = xyz_met_norm * z_met
    return xyz_met


def pixel_to_camera_torch(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts tensors of (m, 2) or tensors of (m, 2, x) or tensors of (m, x, 2) where x is the number of keypoints
    """
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1
    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.mm(uv_padded, kk_1)
    xyz_met = xyz_met_norm * z_met
    return xyz_met


def project_to_pixels(xyz, kk):
    """Project a single point in space into the image"""
    xx, yy, zz = np.dot(kk, xyz)
    uu = int(xx / zz)
    vv = int(yy / zz)

    return uu, vv


def project_3d(box_obj, kk):
    """
    Project a 3D bounding box into the image plane using the central corners
    """

    box_2d = []

    # Obtain the 3d points of the box
    xc, yc, zc = box_obj.center
    ww, ll, hh, = box_obj.wlh

    # Points corresponding to a box at the z of the center
    x1 = xc - ww/2
    y1 = yc - hh/2  # Y axis directed below
    x2 = xc + ww/2
    y2 = yc + hh/2
    xyz1 = np.array([x1, y1, zc])
    xyz2 = np.array([x2, y2, zc])
    corners_3d = np.array([xyz1, xyz2])

    # Project them and convert into pixel coordinates
    for xyz in corners_3d:
        xx, yy, zz = np.dot(kk, xyz)
        uu = xx / zz
        vv = yy / zz
        box_2d.append(uu)
        box_2d.append(vv)

    return box_2d


def preprocess_single(kps, kk):

    """ Preprocess input of a single annotations
    Input_kps = list of 4 elements with 0=x, 1=y, 2= confidence, 3 = ? in pixels
    Output_kps = [x0, y0, x1,...x15, y15] in meters normalized (z=1) and zero-centered using the center of the box
    """

    kps_uv = []
    kps_0c = []
    kps_orig = []

    # Create center of the bounding box using min max of the keypoints
    uu_c, vv_c = get_keypoints(kps[0], kps[1], mode='center')
    uv_center = np.array([uu_c, vv_c, 1])

    # Create a list of single arrays of (u, v, 1)
    for idx, _ in enumerate(kps[0]):
        uv_kp = np.array([kps[0][idx], kps[1][idx], 1])
        kps_uv.append(uv_kp)

    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    xy1_center = pixel_to_camera(uv_center, kk, 1) * 10
    for idx, kp in enumerate(kps_uv):
        kp_proj = pixel_to_camera(kp, kk, 1) * 10
        kp_proj_0c = kp_proj - xy1_center
        kps_0c.append(float(kp_proj_0c[0]))
        kps_0c.append(float(kp_proj_0c[1]))

        kp_orig = pixel_to_camera(kp, kk, 1)
        kps_orig.append(float(kp_orig[0]))
        kps_orig.append(float(kp_orig[1]))

    return kps_0c, kps_orig


def get_keypoints(kps_0, kps_1, mode):
    """Get the center of 2 lists"""

    assert mode == 'center' or mode == 'shoulder' or mode == 'hip'

    if mode == 'center':
        uu = (max(kps_0) - min(kps_0)) / 2 + min(kps_0)
        vv = (max(kps_1) - min(kps_1)) / 2 + min(kps_1)

    elif mode == 'shoulder':
        uu = float(np.average(kps_0[5:7]))
        vv = float(np.average(kps_1[5:7]))

    elif mode == 'hip':
        uu = float(np.average(kps_0[11:13]))
        vv = float(np.average(kps_1[11:13]))

    return uu, vv


def get_keypoints_batch(keypoints, mode):
    """Get the center of 2 lists"""

    assert mode == 'center' or mode == 'shoulder' or mode == 'hip'

    kps_np = np.array(keypoints)  # (m, 3, 17)
    kps_in = kps_np[:, 0:2, :]  # (m, 2, 17)

    if mode == 'center':
        kps_out = (np.max(kps_in, axis=2) - np.min(kps_in, axis=2)) / 2 + np.min(kps_in, axis=2)  # (m, 2, 1)

    elif mode == 'shoulder':
        kps_out = np.average(kps_in[:, :, 5:7], axis=2)

    elif mode == 'hip':
        kps_out = np.average(kps_in[:, :, 11:13], axis=2)

    return kps_out  # (m, 2, 1)


def get_keypoints_torch(keypoints, mode):
    """Get the center of 2 lists"""

    assert mode == 'center' or mode == 'shoulder' or mode == 'hip'
    kps_in = keypoints[:, 0:2, :]  # (m, 2, 17)
    if mode == 'center':
        kps_max, _ = kps_in.max(2)  # returns value, indices
        kps_min, _ = kps_in.min(2)
        kps_out = (kps_max - kps_min) / 2 + kps_min   # (m, 2) as keepdims is False

    elif mode == 'shoulder':
        kps_out = kps_in[:, :, 5:7].mean(2)

    elif mode == 'hip':
        kps_out = kps_in[:, :, 11:13].mean(2)

    return kps_out  # (m, 2)


def transform_kp(kps, tr_mode):
    """Apply different transformations to the keypoints based on the tr_mode"""

    assert tr_mode == "None" or tr_mode == "singularity" or tr_mode == "upper" or tr_mode == "lower" \
           or tr_mode == "horizontal" or tr_mode == "vertical" or tr_mode == "lateral" \
           or tr_mode == 'shoulder' or tr_mode == 'knee' or tr_mode == 'upside' or tr_mode == 'falling' \
           or tr_mode == 'random'

    uu_c, vv_c = get_keypoints(kps[0], kps[1], mode='center')

    if tr_mode == "None":
        return kps

    elif tr_mode == "singularity":
        uus = [uu_c for uu in kps[0]]
        vvs = [vv_c for vv in kps[1]]

    elif tr_mode == "vertical":
        uus = [uu_c for uu in kps[0]]
        vvs = kps[1]

    elif tr_mode == 'horizontal':
        uus = kps[0]
        vvs = [vv_c for vv in kps[1]]

    elif tr_mode == 'lower':
        uus = kps[0]
        vvs = kps[1][:9] + [vv_c for vv in kps[1][9:]]

    elif tr_mode == 'upper':
        uus = kps[0]
        vvs = [vv_c for vv in kps[1][:9]] + kps[1][9:]

    elif tr_mode == 'lateral':
        uus = []
        for idx, kp in enumerate(kps[0]):
            if idx % 2 == 1:
                uus.append(kp)
            else:
                uus.append(uu_c)
        vvs = kps[1]

    elif tr_mode == 'shoulder':
        uus = kps[0]
        vvs = kps[1][:7] + [kps[1][6] for vv in kps[1][7:]]

    elif tr_mode == 'knee':
        uus = kps[0]
        vvs = [kps[1][14] for vv in kps[1][:13]] + kps[1][13:]

    elif tr_mode == 'up':
        uus = kps[0]
        vvs = [kp - 300 for kp in kps[1]]

    elif tr_mode == 'falling':
        uus = [kps[0][16] - kp + kps[1][16] for kp in kps[1]]
        vvs = [kps[1][16] - kp + kps[0][16] for kp in kps[0]]

    elif tr_mode == 'random':
        uu_min = min(kps[0])
        uu_max = max(kps[0])
        vv_min = min(kps[1])
        vv_max = max(kps[1])
        np.random.seed(6)
        uus = np.random.uniform(uu_min, uu_max, len(kps[0])).tolist()
        vvs = np.random.uniform(vv_min, vv_max, len(kps[1])).tolist()

    return [uus, vvs, kps[2], []]


def get_depth(uv_center, kk, dd):

    if len(uv_center) == 2:
        uv_center.extend([1])
    uv_center_np = np.array(uv_center)
    xyz_norm = pixel_to_camera(uv_center, kk, 1)
    zz = dd / math.sqrt(1 + xyz_norm[0] ** 2 + xyz_norm[1] ** 2)

    xyz = pixel_to_camera(uv_center_np, kk, zz).tolist()
    return xyz


def get_depth_from_distance(outputs, xy_centers):

    list_zzs = []
    for idx, _ in enumerate(outputs):
        dd = float(outputs[idx][0])
        xx_1 = float(xy_centers[idx][0])
        yy_1 = float(xy_centers[idx][1])
        zz = dd / math.sqrt(1 + xx_1 ** 2 + yy_1 ** 2)
        list_zzs.append(zz)
    return list_zzs
