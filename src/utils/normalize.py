
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


# def normalize_arrays_jo(dic_jo):
#     """Normalize according to the mean and std of each keypoint in the training dataset
#     PS normalization of training also for test and val"""
#
#     # Normalize
#     phase = 'train'
#     kps_orig_tr = np.array(dic_jo[phase]['X'])
#     # dd_orig_tr = np.array(dic_jo[phase]['Y']).reshape(-1, 1)
#     kps_mean = np.mean(kps_orig_tr, axis=0)
#     plt.hist(kps_orig_tr, bins=100)
#     plt.show()
#     kps_std = np.std(kps_orig_tr, axis=0)
#     # dd_mean = np.mean(dd_orig_tr, axis=0)
#     # dd_std = np.std(dd_orig_tr, axis=0)
#
#     for phase in dic_jo:
#
#         # Compute the normalized arrays
#         kps_orig = np.array(dic_jo[phase]['X'])
#         dd_orig = np.array(dic_jo[phase]['Y']).reshape(-1, 1)
#         kps_norm = np.divide((kps_orig - kps_mean), kps_std)
#
#         # dd_norm = np.divide((dd_orig - dd_mean), dd_std)  # ! No normalization on the output
#
#         # Substitute the new values in the dictionary and save the mean and std
#         dic_jo[phase]['X'] = kps_norm.tolist()
#         dic_jo[phase]['mean']['X'] = kps_mean.tolist()
#         dic_jo[phase]['std']['X'] = kps_std.tolist()
#
#         dic_jo[phase]['Y'] = dd_orig.tolist()
#         # dic_jo[phase]['mean']['Y'] = float(dd_mean)
#         # dic_jo[phase]['std']['Y'] = float(dd_std)
#
#         # Normalize all the clusters
#         for clst in dic_jo[phase]['clst']:
#
#             # Extract
#             kps_orig = np.array(dic_jo[phase]['clst'][clst]['X'])
#             dd_orig = np.array(dic_jo[phase]['clst'][clst]['Y']).reshape(-1, 1)
#             # Normalize
#             kps_norm = np.divide((kps_orig - kps_mean), kps_std)
#
#             # dd_norm = np.divide((dd_orig - dd_mean), dd_std)  #! No normalization on the output
#
#             # Put back
#             dic_jo[phase]['clst'][clst]['X'] = kps_norm.tolist()
#             dic_jo[phase]['clst'][clst]['Y'] = dd_orig.tolist()
#
#     return dic_jo
#
#
# def check_cluster_dim(dic_jo):
#     """ Check that the sum of the clusters corresponds to all annotations"""
#
#     for phase in ['train', 'val', 'test']:
#         cnt_clst = 0
#         cnt_all = len(dic_jo[phase]['X'])
#         for clst in dic_jo[phase]['clst']:
#             cnt_clst += len(dic_jo[phase]['clst'][clst]['X'])
#         assert cnt_all == cnt_clst



