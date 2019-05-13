
import numpy as np
import copy

from utils.camera import pixel_to_camera, get_keypoints
from eval.geom_baseline import compute_distance_single


def eval_geometric(uv_kps, uv_centers, uv_shoulders, kk, average_y=0.48):
    """
    Evaluate geometric distance
    """
    xy_centers = []
    dds_geom = []
    for idx, _ in enumerate(uv_centers):
        uv_center = copy.deepcopy(uv_centers[idx])
        uv_center.append(1)
        uv_shoulder = copy.deepcopy(uv_shoulders[idx])
        uv_shoulder.append(1)
        uv_kp = uv_kps[idx]
        xy_center = pixel_to_camera(uv_center, kk, 1)
        xy_centers.append(xy_center.tolist())

        uu_2, vv_2 = get_keypoints(uv_kp[0], uv_kp[1], mode='hip')
        uv_hip = [uu_2, vv_2, 1]

        zz, _ = compute_distance_single(uv_shoulder, uv_hip, kk, average_y)
        xyz_center = np.array([xy_center[0], xy_center[1], zz])
        dd_geom = float(np.linalg.norm(xyz_center))
        dds_geom.append(dd_geom)

    return dds_geom, xy_centers


def get_calibration(path_txt):
    """Read calibration parameters from txt file:
    For the left color camera we use P2 which is K * [I|t]

    P = [fu, 0, x0, fu*t1-x0*t3
         0, fv, y0, fv*t2-y0*t3
         0, 0,  1,          t3]

    check also http://ksimek.github.io/2013/08/13/intrinsic/

    Simple case test:
    xyz = np.array([2, 3, 30, 1]).reshape(4, 1)
    xyz_2 = xyz[0:-1] + tt
    uv_temp = np.dot(kk, xyz_2)
    uv_1 = uv_temp / uv_temp[-1]
    kk_1 = np.linalg.inv(kk)
    xyz_temp2 = np.dot(kk_1, uv_1)
    xyz_new_2 = xyz_temp2 * xyz_2[2]
    xyz_fin_2 = xyz_new_2 - tt
    """

    with open(path_txt, "r") as ff:
        file = ff.readlines()
    p2_str = file[2].split()[1:]
    p2_list = [float(xx) for xx in p2_str]
    p2 = np.array(p2_list).reshape(3, 4)

    kk = p2[:, :-1]
    f_x = kk[0, 0]
    f_y = kk[1, 1]
    x0 = kk[2, 0]
    y0 = kk[2, 1]
    aa = p2[0, 3]
    bb = p2[1, 3]
    t3 = p2[2, 3]
    t1 = (aa - x0*t3) / f_x
    t2 = (bb - y0*t3) / f_y
    tt = np.array([t1, t2, t3]).reshape(3, 1)
    return kk, tt


def get_simplified_calibration(path_txt):

    with open(path_txt, "r") as ff:
        file = ff.readlines()

    for line in file:
        if line[:4] == 'K_02':
            kk_str = line[4:].split()[1:]
            kk_list = [float(xx) for xx in kk_str]
            kk = np.array(kk_list).reshape(3, 3).tolist()
            return kk

    raise ValueError('Matrix K_02 not found in the file')


def check_conditions(line, mode, thresh=0.5):

    """Check conditions of our or m3d txt file"""

    check = False
    assert mode == 'gt' or mode == 'm3d' or mode == '3dop' or mode == 'our', "Type not recognized"

    if mode == 'm3d' or mode == '3dop':
        conf = line.split()[15]
        if line[:10] == 'pedestrian' and float(conf) >= thresh:
            check = True

    elif mode == 'gt':
        if line[:10] == 'Pedestrian':
            check = True

    elif mode == 'our':
        if line[10] >= thresh:
            check = True

    return check


def get_category(box, trunc, occ):

    hh = box[3] - box[1]

    if hh >= 40 and trunc <= 0.15 and occ <= 0:
        cat = 'easy'
    elif trunc <= 0.3 and occ <= 1:
        cat = 'moderate'
    else:
        cat = 'hard'

    return cat
