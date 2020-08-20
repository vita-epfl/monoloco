
import json
import os

import numpy as np
import torch
import torchvision

from ..utils import get_keypoints, pixel_to_camera, to_cartesian, back_correct_angles

BF = 0.54 * 721
z_min = 4
z_max = 60
D_MIN = BF / z_max
D_MAX = BF / z_min


def preprocess_monstereo(keypoints, keypoints_r, kk):
    """
    Combine left and right keypoints in all-vs-all settings
    """
    clusters = []
    inputs_l = preprocess_monoloco(keypoints, kk)
    inputs_r = preprocess_monoloco(keypoints_r, kk)

    inputs = torch.empty((0, 68)).to(inputs_l.device)
    for idx, inp_l in enumerate(inputs_l.split(1)):
        clst = 0
        # inp_l = torch.cat((inp_l, cat[:, idx:idx+1]), dim=1)
        for idx_r, inp_r in enumerate(inputs_r.split(1)):
            # if D_MIN < avg_disparities[idx_r] < D_MAX:  # Check the range of disparities
            inp_r = inputs_r[idx_r, :]
            inp = torch.cat((inp_l, inp_l - inp_r), dim=1)  # (1,68)
            inputs = torch.cat((inputs, inp), dim=0)
            clst += 1
        clusters.append(clst)
    return inputs, clusters


def preprocess_monoloco(keypoints, kk, zero_center=False):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    if zero_center:
        kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    else:
        kps_norm = xy1_all
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    # kps_out = torch.cat((kps_out, keypoints[:, 2, :]), dim=1)
    return kps_out


def factory_for_gt(im_size, name=None, path_gt=None, verbose=True):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        if verbose:
            print('-' * 120 + "\nGround-truth file opened")
    except (FileNotFoundError, TypeError):
        if verbose:
            print('-' * 120 + "\nGround-truth file not found")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        if verbose:
            print("Matched ground-truth file!")
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2  # 1.7 for MOT
        # pixel_factor = 1
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration
        if verbose:
            print("Using a standard calibration matrix...")

    return kk, dic_gt


def laplace_sampling(outputs, n_samples):

    torch.manual_seed(1)
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

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi


def preprocess_mask(dir_ann, basename, mode='left'):

    dir_ann = os.path.join(os.path.split(dir_ann)[0], 'mask')
    if mode == 'left':
        path_ann = os.path.join(dir_ann, basename + '.json')
    elif mode == 'right':
        path_ann = os.path.join(dir_ann + '_right', basename + '.json')

    from ..utils import open_annotations
    dic = open_annotations(path_ann)
    if isinstance(dic, list):
        return [], []

    keypoints = []
    for kps in dic['keypoints']:
        kps = prepare_pif_kps(np.array(kps).reshape(51,).tolist())
        keypoints.append(kps)
    return dic['boxes'], keypoints


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
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

        if conf >= min_conf:
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


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)


def extract_outputs(outputs, tasks=()):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'x': outputs[:, 0:1],
               'y': outputs[:, 1:2],
               'd': outputs[:, 2:4],
               'h': outputs[:, 4:5],
               'w': outputs[:, 5:6],
               'l': outputs[:, 6:7],
               'ori': outputs[:, 7:9]}

    if outputs.shape[1] == 10:
        dic_out['aux'] = outputs[:, 9:10]

    # Multi-task training
    if len(tasks) >= 1:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    bi = unnormalize_bi(dic_out['d'])
    dic_out['bi'] = bi

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='x')
    y = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='y')
    d = dic_out['d'][:, 0:1]
    z = torch.sqrt(d**2 - x**2 - y**2)
    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])
    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    if outputs.shape[1] == 10:
        dic_out['aux'] = torch.sigmoid(dic_out['aux'])

    return dic_out


def extract_labels_aux(labels, tasks=None):

    dic_gt_out = {'aux': labels[:, 0:1]}

    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}
    return dic_gt_out


def extract_labels(labels, tasks=None):

    dic_gt_out = {'x': labels[:, 0:1], 'y': labels[:, 1:2], 'z': labels[:, 2:3], 'd': labels[:, 3:4],
                  'h': labels[:, 4:5], 'w': labels[:, 5:6], 'l': labels[:, 6:7],
                  'ori': labels[:, 7:9], 'aux': labels[:, 10:11]}

    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}
    return dic_gt_out


def cluster_outputs(outputs, clusters):
    """Cluster the outputs based on the number of right keypoints"""

    # Check for "no right keypoints" condition
    if clusters == 0:
        clusters = max(1, round(outputs.shape[0] / 2))

    assert outputs.shape[0] % clusters == 0, "Unexpected number of inputs"
    outputs = outputs.view(-1, clusters, outputs.shape[1])
    return outputs


def filter_outputs(outputs):
    """Extract a single output for each left keypoint"""

    # Max of auxiliary task
    val = outputs[:, :, -1]
    best_val, _ = val.max(dim=1, keepdim=True)
    mask = val >= best_val
    output = outputs[mask]  # broadcasting happens only if 3rd dim not present
    return output, mask


def extract_outputs_mono(outputs, tasks=None):
    """
    Extract the outputs for single di
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'xyz': outputs[:, 0:3], 'zb': outputs[:, 2:4],
               'h': outputs[:, 4:5], 'w': outputs[:, 5:6], 'l': outputs[:, 6:7], 'ori': outputs[:, 7:9]}

    # Multi-task training
    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    bi = unnormalize_bi(dic_out['zb'])

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    dd = torch.norm(dic_out['xyz'], p=2, dim=1).view(-1, 1)
    dic_out['xyzd'] = torch.cat((dic_out['xyz'], dd), dim=1)

    dic_out['d'], dic_out['bi'] = dd, bi

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])

    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry
    return dic_out
