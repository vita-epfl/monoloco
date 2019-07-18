
import numpy as np


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

        kps = prepare_pif_kps(dic['keypoints'])
        conf = float(np.sort(np.array(kps[2]))[-3])  # The confidence is the 3rd highest value for the keypoints

        # Add 15% for y and 20% for x
        delta_h = (box[3] - box[1]) / 7
        delta_w = (box[2] - box[0]) / 3.5
        assert delta_h > -5 and delta_w > -5, "Bounding box <=0"
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
