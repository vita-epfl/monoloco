
import numpy as np


def correct_boxes(boxes, hwls, xyzs, yaws, path_calib):

    with open(path_calib, "r") as ff:
        file = ff.readlines()
    p2_str = file[2].split()[1:]
    p2_list = [float(xx) for xx in p2_str]
    P = np.array(p2_list).reshape(3, 4)
    boxes_new = []
    for idx, box in enumerate(boxes):
        hwl = hwls[idx]
        xyz = xyzs[idx]
        yaw = yaws[idx]
        corners_2d, corners_3d = compute_box_3d(hwl, xyz, yaw, P)
        box_new = project_8p_to_4p(corners_2d).reshape(-1).tolist()
        boxes_new.append(box_new)
    return boxes_new


def compute_box_3d(hwl, xyz, ry, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(ry)

    # 3d bounding box dimensions
    l = hwl[2]
    w = hwl[1]
    h = hwl[0]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + xyz[0]
    corners_3d[1, :] = corners_3d[1, :] + xyz[1]
    corners_3d[2, :] = corners_3d[2, :] + xyz[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)



def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])



def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.
    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]



def project_8p_to_4p(pts_2d):
    x0 = np.min(pts_2d[:, 0])
    x1 = np.max(pts_2d[:, 0])
    y0 = np.min(pts_2d[:, 1])
    y1 = np.max(pts_2d[:, 1])
    x0 = max(0, x0)
    y0 = max(0, y0)
    return np.array([x0, y0, x1, y1])
