
import numpy as np
import torch
import time
import logging


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


def calculate_iou(box1, box2):

    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
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


def get_idx_max(box, boxes_gt):

    """Compute and save errors between a single box and the gt box which match"""

    iou_max = 0
    idx_max = 0
    for idx_gt, box_gt in enumerate(boxes_gt):
        iou = calculate_iou(box, box_gt)
        if iou > iou_max:
            idx_max = idx_gt
            iou_max = iou

    return idx_max, iou_max


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


def get_iou_matches(boxes, boxes_gt, thresh):
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


def reparametrize_box3d(box):
    """Reparametrized 3D box in the XZ plane and add the height"""

    hh, ww, ll = box[0:3]
    x_c, y_c, z_c = box[3:6]

    x1 = x_c - ll/2
    z1 = z_c - ww/2
    x2 = x_c + ll/2
    z2 = z_c + ww / 2

    return [x1, z1, x2, z2, hh]


def laplace_sampling(outputs, n_samples):

    # np.random.seed(1)
    t0 = time.time()
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda

    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    xxs = torch.empty((0, mu.shape[0])).to(device)
    laplace = torch.distributions.Laplace(mu, bi)
    for ii in range(1):
        xx = laplace.sample((n_samples,))
        xxs = torch.cat((xxs, xx.view(n_samples, -1)), 0)

    return xxs


def epistemic_variance(total_outputs):
    """Compute epistemic variance"""

    # var_y = np.sum(total_outputs**2, axis=0) / total_outputs.shape[0] - (np.mean(total_outputs, axis=0))**2
    var_y = np.var(total_outputs, axis=0)
    lb = np.quantile(a=total_outputs, q=0.25, axis=0)
    up = np.quantile(a=total_outputs, q=0.75, axis=0)
    var_new = (up-lb)

    return var_y, var_new


def append_cluster(dic_jo, phase, xx, dd, kps):

    """Append the annotation based on its distance"""

    if dd <= 10:
        dic_jo[phase]['clst']['10']['kps'].append(kps)
        dic_jo[phase]['clst']['10']['X'].append(xx)
        dic_jo[phase]['clst']['10']['Y'].append([dd])

    elif dd <= 20:
        dic_jo[phase]['clst']['20']['kps'].append(kps)
        dic_jo[phase]['clst']['20']['X'].append(xx)
        dic_jo[phase]['clst']['20']['Y'].append([dd])

    elif dd <= 30:
        dic_jo[phase]['clst']['30']['kps'].append(kps)
        dic_jo[phase]['clst']['30']['X'].append(xx)
        dic_jo[phase]['clst']['30']['Y'].append([dd])

    else:
        dic_jo[phase]['clst']['>30']['kps'].append(kps)
        dic_jo[phase]['clst']['>30']['X'].append(xx)
        dic_jo[phase]['clst']['>30']['Y'].append([dd])


def get_task_error(dd):
    """Get target error not knowing the gender"""
    mm_gender = 0.0556
    return mm_gender * dd


