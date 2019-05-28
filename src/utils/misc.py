
import numpy as np
import torch
import time
import logging
# from shapely.geometry import box as Sbox

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


def reparametrize_box3d(box):
    """Reparametrized 3D box in the XZ plane and add the height"""

    hh, ww, ll = box[0:3]
    x_c, y_c, z_c = box[3:6]

    x1 = x_c - ll/2
    z1 = z_c - ww/2
    x2 = x_c + ll/2
    z2 = z_c + ww / 2

    return [x1, z1, x2, z2, hh]


# def calculate_iou3d(box3d_1, box3d_2):
#     """3D intersection over union. Boxes are parametrized as x1, z1, x2, z2, hh
#     We compute 2d iou in the birds plane and then add a factor for height differences (0-1)"""
#
#     poly1 = Sbox(box3d_1[0], box3d_1[1], box3d_1[2], box3d_1[3])
#     poly2 = Sbox(box3d_2[0], box3d_2[1], box3d_2[2], box3d_2[3])
#
#     inter_2d = poly1.intersection(poly2).area
#     union_2d = poly1.area + poly2.area - inter_2d
#
#     # height_factor = 1 - abs(box3d_1[4] - box3d_2[4]) / max(box3d_1[4], box3d_2[4])
#
#     #
#     iou_3d = inter_2d / union_2d  # * height_factor
#
#     return iou_3d


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
    t1 = time.time()

    xxs = torch.empty((0, mu.shape[0])).to(device)
    t2 = time.time()

    laplace = torch.distributions.Laplace(mu, bi)
    t3 = time.time()
    for ii in range(1):
        xx = laplace.sample((n_samples,))
        t4a = time.time()
        xxs = torch.cat((xxs, xx.view(n_samples, -1)), 0)
    t4 = time.time()

    # time_tot = t4 - t0
    # time_1 = t1 - t0
    # time_2 = t2 - t1
    # time_3 = t3 - t2
    # time_4a = t4a - t3
    # time_4 = t4 - t3
    # print("Time 1: {:.1f}%".format(time_1 / time_tot * 100))
    # print("Time 2: {:.1f}%".format(time_2 / time_tot * 100))
    # print("Time 3: {:.1f}%".format(time_3 / time_tot * 100))
    # print("Time 4a: {:.1f}%".format(time_4a / time_tot * 100))
    # print("Time 4: {:.1f}%".format(time_4 / time_tot * 100))
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

    # if dd <= 6:
    #     dic_jo[phase]['clst']['6']['kps'].append(kps)
    #     dic_jo[phase]['clst']['6']['X'].append(xx)
    #     dic_jo[phase]['clst']['6']['Y'].append([dd])  # Trick to make it (nn,1) instead of (nn, )

    if dd <= 10:
        dic_jo[phase]['clst']['10']['kps'].append(kps)
        dic_jo[phase]['clst']['10']['X'].append(xx)
        dic_jo[phase]['clst']['10']['Y'].append([dd])

    # elif dd <= 15:
    #     dic_jo[phase]['clst']['15']['kps'].append(kps)
    #     dic_jo[phase]['clst']['15']['X'].append(xx)
    #     dic_jo[phase]['clst']['15']['Y'].append([dd])

    elif dd <= 20:
        dic_jo[phase]['clst']['20']['kps'].append(kps)
        dic_jo[phase]['clst']['20']['X'].append(xx)
        dic_jo[phase]['clst']['20']['Y'].append([dd])

    # elif dd <= 25:
    #     dic_jo[phase]['clst']['25']['kps'].append(kps)
    #     dic_jo[phase]['clst']['25']['X'].append(xx)
    #     dic_jo[phase]['clst']['25']['Y'].append([dd])

    elif dd <= 30:
        dic_jo[phase]['clst']['30']['kps'].append(kps)
        dic_jo[phase]['clst']['30']['X'].append(xx)
        dic_jo[phase]['clst']['30']['Y'].append([dd])

    # elif dd <= 40:
    #     dic_jo[phase]['clst']['40']['kps'].append(kps)
    #     dic_jo[phase]['clst']['40']['X'].append(xx)
    #     dic_jo[phase]['clst']['40']['Y'].append([dd])
    #
    # elif dd <= 50:
    #     dic_jo[phase]['clst']['50']['kps'].append(kps)
    #     dic_jo[phase]['clst']['50']['X'].append(xx)
    #     dic_jo[phase]['clst']['50']['Y'].append([dd])

    else:
        dic_jo[phase]['clst']['>30']['kps'].append(kps)
        dic_jo[phase]['clst']['>30']['X'].append(xx)
        dic_jo[phase]['clst']['>30']['Y'].append([dd])


def distance_from_disparity(list_dds, list_kps):
    """Associate instances in left and right images and compute disparuty"""

    stereo_dds = None
    return stereo_dds


