
import random


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


def get_task_error(dd, mode='std'):
    """Get target error not knowing the gender, modeled through a Gaussian Mixure model"""
    assert mode in ('std', 'mad')
    h_mean = 171.5  # average h of the human distribution
    if mode == 'std':
        delta_h = 9.07  # delta h for 63% confidence interval
    elif mode == 'mad':
        delta_h = 7.83  # delta_h of mean absolute deviation
    return dd * (1 - h_mean / (h_mean + delta_h))


def get_pixel_error(dd_gt, zz_gt):
    """calculate error in stereo distance due to +-1 pixel mismatch (function of depth)"""

    disp = 0.54 * 721 / zz_gt
    random.seed(1)
    sign = random.choice((-1, 1))
    delta_z = zz_gt - 0.54 * 721 / (disp + sign)
    return dd_gt + delta_z
