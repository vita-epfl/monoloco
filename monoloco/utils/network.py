
import numpy as np
import torch
from ..utils.camera import get_keypoints, pixel_to_camera


def get_monoloco_inputs(keypoints, kk):

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
    # xy1_center[:, 1].fill_(0)  #TODO
    kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    return kps_out


def laplace_sampling(outputs, n_samples):

    # np.random.seed(1)
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


def epistemic_variance(total_outputs):
    """Compute epistemic variance"""

    # var_y = np.sum(total_outputs**2, axis=0) / total_outputs.shape[0] - (np.mean(total_outputs, axis=0))**2
    var_y = np.var(total_outputs, axis=0)
    lower_b = np.quantile(a=total_outputs, q=0.25, axis=0)
    upper_b = np.quantile(a=total_outputs, q=0.75, axis=0)
    var_new = (upper_b - lower_b)

    return var_y, var_new


def unnormalize_bi(outputs):
    """Unnormalize relative bi of a nunmpy array"""

    outputs[:, 1] = torch.exp(outputs[:, 1]) * outputs[:, 0]
    return outputs
