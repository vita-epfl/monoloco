
import numpy as np
from utils.camera import preprocess_single, get_keypoints, pixel_to_camera, get_keypoints_torch, pixel_to_camera_torch


def get_network_inputs(keypoints, kk):

    """ Preprocess input of a single annotations
    Input_kps = list of 4 elements with 0=x, 1=y, 2= confidence, 3 = ? in pixels
    Output_kps = [x0, y0, x1,...x15, y15] in meters normalized (z=1) and zero-centered using the center of the box
    """

    kps_uv = []
    kps_0c = []
    kps_orig = []

    # Create center of the bounding box using min max of the keypoints
    uv_center = get_keypoints_torch(keypoints, mode='center')

    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    xy1_center = pixel_to_camera_torch(uv_center, kk, 1) * 10
    xy1_all = pixel_to_camera_torch(keypoints[:, 0:2, :], kk, 1) * 10

    for idx, kp in enumerate(kps_uv):
        kp_proj = pixel_to_camera(kp, kk, 1) * 10
        kp_proj_0c = kp_proj - xy1_center
        kps_0c.append(float(kp_proj_0c[0]))
        kps_0c.append(float(kp_proj_0c[1]))

        kp_orig = pixel_to_camera(kp, kk, 1)
        kps_orig.append(float(kp_orig[0]))
        kps_orig.append(float(kp_orig[1]))

    return kps_0c, kps_orig


def get_input_data(boxes, keypoints, kk, left_to_right=False):
    inputs = []
    xy_centers = []
    uv_boxes = []
    uv_centers = []
    uv_shoulders = []
    uv_kps = []
    xy_kps = []

    # if left_to_right:  # Order boxes from left to right
    #     ordered = np.argsort([xx[0] for xx in boxes])

    # else:  # Order boxes from most to less confident
    #     confs = []
    #     for idx, box in enumerate(boxes):
    #         confs.append(box[4])
    #     ordered = np.argsort(confs).tolist()[::-1]

    # for idx in ordered:

    for idx, kps in enumerate(keypoints):
        # kps = keypoints[idx]
        uv_kps.append(kps)
        uv_boxes.append(boxes[idx])
        if idx == 0:
            aa = 5
        uu_c, vv_c = get_keypoints(kps[0], kps[1], "center")
        uv_centers.append([round(uu_c), round(vv_c)])
        xy_center = pixel_to_camera(np.array([uu_c, vv_c, 1]), kk, 1)
        xy_centers.append(xy_center)

        uu_1, vv_1 = get_keypoints(kps[0], kps[1], "shoulder")
        uv_shoulders.append([round(uu_1), round(vv_1)])

        # 2 steps of input normalization for each instance
        kps_prep, kps_orig = preprocess_single(kps, kk)
        inputs.append(kps_prep)
        xy_kps.append(kps_orig)

    return (inputs, xy_kps), (uv_kps, uv_boxes, uv_centers, uv_shoulders)


def preprocess_pif(annotations, im_size=None):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []

    for dic in annotations:
        box = dic['bbox']
        if box[3] < 0.5:  # Check for no detections (boxes 0,0,0,0)
            return [], []

        else:
            kps = prepare_pif_kps(dic['keypoints'])
            conf = float(np.mean(np.array(kps[2])))

            # Add 10% for y
            delta_h = (box[3] - box[1]) / 10
            delta_w = (box[2] - box[0]) / 10
            assert delta_h > 0 and delta_w > 0, "Bounding box <=0"
            box[0] -= delta_w
            box[1] -= delta_h
            box[2] += delta_w
            box[3] += delta_h

            # Put the box inside the image
            if im_size is not None:
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(box[2], im_size[0])
                box[3] = min(box[3], im_size[1])

            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


