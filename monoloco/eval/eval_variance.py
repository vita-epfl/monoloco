# pylint: disable=too-many-statements

"""Joints Analysis: Supplementary material of MonStereo"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from ..utils import find_cluster, average
from ..visuals.figures import get_distances
from ..prep.transforms import COCO_KEYPOINTS


def joints_variance(joints, clusters, dic_ms):
    # CLUSTERS = ('3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23', '25', '27', '29', '31', '49')
    BF = 0.54 * 721
    phase = 'train'
    methods = ('pifpaf', 'mask')
    dic_fin = {}

    for method in methods:
        dic_var = defaultdict(lambda: defaultdict(list))
        dic_joints = defaultdict(list)
        dic_avg = defaultdict(lambda: defaultdict(float))
        path_joints = joints + '_' + method + '.json'

        with open(path_joints, 'r') as f:
            dic_jo = json.load(f)

        for idx, keypoint in enumerate(dic_jo[phase]['kps']):
            # if dic_jo[phase]['names'][idx] == '005856.txt' and dic_jo[phase]['Y'][idx][2] > 14:
            #     aa = 4
            assert len(keypoint) < 2
            kps = np.array(keypoint[0])[:, :17]
            kps_r = np.array(keypoint[0])[:, 17:]
            disps = kps[0] - kps_r[0]
            zz = dic_jo[phase]['Y'][idx][2]
            disps_3 = get_variance(kps, kps_r, zz)
            disps_8 = get_variance_conf(kps, kps_r, num=8)
            disps_4 = get_variance_conf(kps, kps_r, num=4)
            disp_gt = BF / zz
            clst = find_cluster(zz, clusters)   # 4 = '3'    35 = '31'     42 = 2 = 'excl'
            dic_var['std_d'][clst].append(disps.std())
            errors = np.minimum(30, np.abs(zz - BF / disps))
            dic_var['mean_dev'][clst].append(min(30, abs(zz - BF / np.median(disps))))
            dic_var['mean_3'][clst].append(min(30, abs(zz - BF / disps_3.mean())))
            dic_var['mean_8'][clst].append(min(30, abs(zz - BF / np.median(disps_8))))
            dic_var['mean_4'][clst].append(min(30, abs(zz - BF / np.median(disps_4))))
            arg_best = np.argmin(errors)
            conf = np.mean((kps[2][arg_best], kps_r[2][arg_best]))
            dic_var['mean_best'][clst].append(np.min(errors))
            dic_var['conf_best'][clst].append(conf)
            dic_var['conf'][clst].append(np.mean((np.mean(kps[2]), np.mean(kps_r[2]))))
            # dic_var['std_z'][clst].append(zzs.std())
            for ii, el in enumerate(disps):
                if abs(el-disp_gt) < 1:
                    dic_var['rep'][clst].append(1)
                    dic_joints[str(ii)].append(1)
                else:
                    dic_var['rep'][clst].append(0)
                    dic_joints[str(ii)].append(0)

        for key in dic_var:
            for clst in clusters[:-1]:  # 41 needs to be excluded (36 = '31')
                dic_avg[key][clst] = average(dic_var[key][clst])
        dic_fin[method] = dic_avg
        for key in dic_joints:
            dic_fin[method]['joints'][key] = average(dic_joints[key])
        dic_fin['monstereo'] = {clst: dic_ms[clst]['mean'] for clst in clusters[:-1]}
    variance_figures(dic_fin, clusters)


def get_variance(kps, kps_r, zz):

    thresh = 0.5 - zz / 100
    disps_2 = []
    disps = kps[0] - kps_r[0]
    arg_disp = np.argsort(disps)[::-1]

    for idx in arg_disp[1:]:
        if kps[2][idx] > thresh and kps_r[2][idx] > thresh:
            disps_2.append(disps[idx])
        if len(disps_2) >= 3:
            return np.array(disps_2)
    return disps


def get_variance_conf(kps, kps_r, num=8):

    disps_conf = []
    confs = (kps[2, :] + kps_r[2, :]) / 2
    disps = kps[0] - kps_r[0]
    arg_disp = np.argsort(confs)[::-1]

    for idx in arg_disp[:num]:
        disps_conf.append(disps[idx])
    return np.array(disps_conf)


def variance_figures(dic_fin, clusters):
    """Predicted confidence intervals and task error as a function of ground-truth distance"""

    dir_out = 'docs'
    x_min = 3
    x_max = 43
    y_min = 0
    y_max = 1

    plt.figure(0)
    plt.xlabel("Ground-truth distance [m]")
    plt.title("Repeatability by distance")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(linewidth=0.2)

    xxs = get_distances(clusters)
    yys_p = [el for _, el in dic_fin['pifpaf']['rep'].items()]
    yys_m = [el for _, el in dic_fin['mask']['rep'].items()]
    plt.plot(xxs, yys_p, marker='s', label="PifPaf")
    plt.plot(xxs, yys_m, marker='o', label="Mask R-CNN")
    plt.tight_layout()
    plt.legend()
    path_fig = os.path.join(dir_out, 'repeatability.png')
    plt.savefig(path_fig)
    print("Figure of repeatability saved in {}".format(path_fig))

    plt.figure(1)
    plt.xlabel("Ground-truth distance [m]")
    plt.ylabel("[m]")
    plt.title("Depth error")
    plt.grid(linewidth=0.2)
    y_min = 0
    y_max = 2.7
    plt.ylim(y_min, y_max)
    yys_p = [el for _, el in dic_fin['pifpaf']['mean_dev'].items()]
    # yys_m = [el for _, el in dic_fin['mask']['mean_dev'].items()]
    yys_p_3 = [el for _, el in dic_fin['pifpaf']['mean_3'].items()]
    yys_p_8 = [el for _, el in dic_fin['pifpaf']['mean_8'].items()]
    yys_p_4 = [el for _, el in dic_fin['pifpaf']['mean_4'].items()]
    # yys_m_3 = [el for _, el in dic_fin['mask']['mean_3'].items()]
    yys_ms = [el for _, el in dic_fin['monstereo'].items()]
    yys_p_best = [el for _, el in dic_fin['pifpaf']['mean_best'].items()]
    plt.plot(xxs, yys_p_4, marker='o', linestyle=':', label="PifPaf (highest 4)")
    plt.plot(xxs, yys_p, marker='+', label="PifPaf (median)")
    # plt.plot(xxs, yys_m, marker='o', label="Mask R-CNN (median")
    plt.plot(xxs, yys_p_3, marker='s', linestyle='--', label="PifPaf (closest 3)")
    plt.plot(xxs, yys_p_8, marker='*', linestyle=':', label="PifPaf (highest 8)")
    plt.plot(xxs, yys_ms, marker='^', label="MonStereo")
    plt.plot(xxs, yys_p_best, marker='o', label="PifPaf (best)")
    # plt.plot(xxs, yys_m_3, marker='o', color='r', label="Mask R-CNN (closest 3)")
    # plt.plot(xxs, yys_mon, marker='o', color='b', label="Our MonStereo")

    plt.legend()
    plt.tight_layout()
    path_fig = os.path.join(dir_out, 'mean_deviation.png')
    plt.savefig(path_fig)
    print("Figure of mean deviation saved in {}".format(path_fig))

    plt.figure(2)
    plt.xlabel("Ground-truth distance [m]")
    plt.ylabel("Pixels")
    plt.title("Standard deviation of joints disparity")
    yys_p = [el for _, el in dic_fin['pifpaf']['std_d'].items()]
    yys_m = [el for _, el in dic_fin['mask']['std_d'].items()]
    # yys_p_z = [el for _, el in dic_fin['pifpaf']['std_z'].items()]
    # yys_m_z = [el for _, el in dic_fin['mask']['std_z'].items()]
    plt.plot(xxs, yys_p, marker='s', label="PifPaf")
    plt.plot(xxs, yys_m, marker='o', label="Mask R-CNN")
    # plt.plot(xxs, yys_p_z, marker='s', color='b', label="PifPaf (meters)")
    # plt.plot(xxs, yys_m_z, marker='o', color='r', label="Mask R-CNN (meters)")

    plt.grid(linewidth=0.2)
    plt.legend()
    path_fig = os.path.join(dir_out, 'std_joints.png')
    plt.savefig(path_fig)
    print("Figure of standard deviation of joints by distance in {}".format(path_fig))

    plt.figure(3)
    # plt.style.use('ggplot')
    width = 0.35
    xxs = np.arange(len(COCO_KEYPOINTS))
    yys_p = [el for _, el in dic_fin['pifpaf']['joints'].items()]
    yys_m = [el for _, el in dic_fin['mask']['joints'].items()]
    plt.bar(xxs, yys_p, width, color='C0', label='Pifpaf')
    plt.bar(xxs + width, yys_m, width, color='C1', label='Mask R-CNN')
    plt.ylim(0, 1)

    plt.xlabel("Keypoints")
    plt.title("Repeatability by keypoint type")

    plt.xticks(xxs + width / 2, xxs)
    plt.legend(loc='best')
    path_fig = os.path.join(dir_out, 'repeatability_2.png')
    plt.savefig(path_fig)
    plt.close('all')
    print("Figure of standard deviation of joints by keypointd in {}".format(path_fig))

    plt.figure(4)
    plt.xlabel("Ground-truth distance [m]")
    plt.ylabel("Confidence")
    plt.grid(linewidth=0.2)
    xxs = get_distances(clusters)
    yys_p_conf = [el for _, el in dic_fin['pifpaf']['conf'].items()]
    yys_p_conf_best = [el for _, el in dic_fin['pifpaf']['conf_best'].items()]
    yys_m_conf = [el for _, el in dic_fin['mask']['conf'].items()]
    yys_m_conf_best = [el for _, el in dic_fin['mask']['conf_best'].items()]
    plt.plot(xxs, yys_p_conf_best, marker='s', color='lightblue', label="PifPaf (best)")
    plt.plot(xxs, yys_p_conf, marker='s', color='b', label="PifPaf (mean)")
    plt.plot(xxs, yys_m_conf_best, marker='^', color='darkorange', label="Mask (best)")
    plt.plot(xxs, yys_m_conf, marker='o', color='r', label="Mask R-CNN (mean)")
    plt.legend()
    plt.tight_layout()
    path_fig = os.path.join(dir_out, 'confidence.png')
    plt.savefig(path_fig)
    print("Figure of confidence saved in {}".format(path_fig))
