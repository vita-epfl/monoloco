
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
    """Get target error not knowing the gender"""
    assert mode in ('std', 'mad')
    if mode == 'std':
        mm_gender = 0.0557
    elif mode == 'mad':  # mean absolute deviation
        mm_gender = 0.0457
    return mm_gender * dd


def get_pixel_error(dd_gt, zz_gt):
    """calculate error in stereo distance due to +-1 pixel mismatch (function of depth)"""

    disp = 0.54 * 721 / zz_gt
    random.seed(1)
    sign = random.choice((-1, 1))
    delta_z = zz_gt - 0.54 * 721 / (disp + sign)
    return dd_gt + delta_z
