
import numpy as np
import torch 
from utils.camera import get_keypoints, pixel_to_camera


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


def get_network_inputs(keypoints, kk):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if type(keypoints) == list:
        keypoints = torch.tensor(keypoints)
    if type(kk) == list:
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 1) * 10
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 1) * 10
    kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    return kps_out


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


