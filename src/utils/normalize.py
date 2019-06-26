
import torch
import numpy as np


def normalize_labels(labels, dic_norm):
    """Apply the log to remove right-skewness and then apply Gaussian Normalization"""

    labels_new = torch.div(torch.log(labels) - dic_norm['mean']['Y'], dic_norm['std']['Y'])
    return labels_new


def unnormalize_labels(labels, dic_norm):
    """Apply the log to remove right-skewness and then apply Gaussian Normalization"""

    return torch.exp(labels * dic_norm['std']['Y'] + dic_norm['mean']['Y'])


def unnormalize_outputs(outputs, dic_norm, dim, normalize):
    """Unnnormalize outputs, with an option to only pass from si to bi"""

    if dim == 1:
        assert normalize, "Inconsistency"
        return unnormalize_labels(outputs, dic_norm)

    else:
        mu = outputs[:, 0:1]
        si = outputs[:, 1:2]
        bi = torch.exp(si)

        if normalize:
            mu = unnormalize_labels(mu, dic_norm)
            bi = torch.exp(bi * dic_norm['std']['Y'])

        return torch.cat((mu, bi), 1)


def unnormalize_bi(outputs):
    """Unnormalize relative bi of a nunmpy array"""

    outputs[:, 1] = torch.exp(outputs[:, 1]) * outputs[:, 0]
    return outputs
