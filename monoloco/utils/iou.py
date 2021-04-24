
import json

import numpy as np


def calculate_iou(box1, box2):

    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
    # box1 = [-3, 8.5, 3, 11.5]
    # box2 = [-3, 9.5, 3, 12.5]
    # box1 = [1086.84, 156.24, 1181.62, 319.12]
    # box2 = [1078.333357, 159.086347, 1193.771014, 322.239107]

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)  # Max keeps into account not overlapping box

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def get_iou_matrix(boxes, boxes_gt):
    """
    Get IoU matrix between predicted and ground truth boxes
    Dim: (boxes, boxes_gt)
    """
    iou_matrix = np.zeros((len(boxes), len(boxes_gt)))
    for idx, box in enumerate(boxes):
        for idx_gt, box_gt in enumerate(boxes_gt):
            iou_matrix[idx, idx_gt] = calculate_iou(box, box_gt)
    return iou_matrix


def get_iou_matches(boxes, boxes_gt, iou_min=0.3):
    """From 2 sets of boxes and a minimum threshold, compute the matching indices for IoU matches"""

    matches = []
    used = []
    if not boxes or not boxes_gt:
        return []
    confs = [box[4] for box in boxes]

    indices = list(np.argsort(confs))
    for idx in indices[::-1]:
        box = boxes[idx]
        ious = []
        for box_gt in boxes_gt:
            iou = calculate_iou(box, box_gt)
            ious.append(iou)
        idx_gt_max = int(np.argmax(ious))
        if (ious[idx_gt_max] >= iou_min) and (idx_gt_max not in used):
            matches.append((int(idx), idx_gt_max))
            used.append(idx_gt_max)
    return matches


def get_iou_matches_matrix(boxes, boxes_gt, thresh):
    """From 2 sets of boxes and a minimum threshold, compute the matching indices for IoU matchings"""

    iou_matrix = get_iou_matrix(boxes, boxes_gt)
    if not iou_matrix.size:
        return []

    matches = []
    iou_max = np.max(iou_matrix)
    while iou_max > thresh:
        # Extract the indeces of the max
        args_max = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        matches.append(args_max)
        iou_matrix[args_max[0], :] = 0
        iou_matrix[:, args_max[1]] = 0
        iou_max = np.max(iou_matrix)
    return matches


def reorder_matches(matches, boxes, mode='left_rigth'):
    """
    Reorder a list of (idx, idx_gt) matches based on position of the detections in the image
    ordered_boxes = (5, 6, 7, 0, 1, 4, 2, 4)
    matches = [(0, x), (2,x), (4,x), (3,x), (5,x)]
    Output --> [(5, x), (0, x), (3, x), (2, x), (5, x)]
    """

    assert mode == 'left_right'

    # Order the boxes based on the left-right position in the image and
    ordered_boxes = np.argsort([box[0] for box in boxes])  # indices of boxes ordered from left to right
    matches_left = [int(idx) for (idx, _) in matches]

    return [matches[matches_left.index(idx_boxes)] for idx_boxes in ordered_boxes if idx_boxes in matches_left]


def get_category(keypoints, path_byc):
    """Find the category for each of the keypoints"""

    dic_byc = open_annotations(path_byc)
    boxes_byc = dic_byc['boxes'] if dic_byc else []
    boxes_ped = make_lower_boxes(keypoints)

    matches = get_matches_bikes(boxes_ped, boxes_byc)
    list_byc = [match[0] for match in matches]
    categories = [1.0 if idx in list_byc else 0.0 for idx, _ in enumerate(boxes_ped)]
    return categories


def get_matches_bikes(boxes_ped, boxes_byc):
    matches = get_iou_matches_matrix(boxes_ped, boxes_byc, thresh=0.15)
    matches_b = []
    for idx, idx_byc in matches:
        box_ped = boxes_ped[idx]
        box_byc = boxes_byc[idx_byc]
        width_ped = box_ped[2] - box_ped[0]
        width_byc = box_byc[2] - box_byc[0]
        center_ped = (box_ped[2] + box_ped[0]) / 2
        center_byc = (box_byc[2] + box_byc[0]) / 2
        if abs(center_ped - center_byc) < min(width_ped, width_byc) / 4:
            matches_b.append((idx, idx_byc))
    return matches_b


def make_lower_boxes(keypoints):
    lower_boxes = []
    keypoints = np.array(keypoints)
    for kps in keypoints:
        lower_boxes.append([min(kps[0, 9:]), min(kps[1, 9:]), max(kps[0, 9:]), max(kps[1, 9:])])
    return lower_boxes


def open_annotations(path_ann):
    try:
        with open(path_ann, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        annotations = []
    return annotations
