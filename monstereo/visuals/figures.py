# pylint: disable=R0915

import math
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..utils import get_task_error, get_pixel_error


def show_results(dic_stats, clusters, show=False, save=False, stereo=True):
    """
    Visualize error as function of the distance and compare it with target errors based on human height analyses
    """

    dir_out = 'docs'
    phase = 'test'
    x_min = 3
    x_max = 42
    y_min = 0
    # y_max = 2.2
    y_max = 3.5 if stereo else 5.2

    xx = np.linspace(x_min, x_max, 100)
    excl_clusters = ['all', 'easy', 'moderate', 'hard']
    clusters = [clst for clst in clusters if clst not in excl_clusters]
    plt.figure(0)
    styles = printing_styles(stereo)
    for idx_style, style in enumerate(styles.items()):
        plt.figure(idx_style)
        plt.grid(linewidth=0.2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Ground-truth distance [m]")
        plt.ylabel("Average localization error (ALE) [m]")
        for idx, method in enumerate(styles['methods']):
            errs = [dic_stats[phase][method][clst]['mean'] for clst in clusters[:-1]]  # last cluster only a bound
            cnts = [dic_stats[phase][method][clst]['cnt'] for clst in clusters[:-1]]  # last cluster only a bound
            assert errs, "method %s empty" % method
            xxs = get_distances(clusters)

            plt.plot(xxs, errs, marker=styles['mks'][idx], markersize=styles['mksizes'][idx],
                     linewidth=styles['lws'][idx],
                     label=styles['labels'][idx], linestyle=styles['lstyles'][idx], color=styles['colors'][idx])
            if method in ('monstereo', 'pseudo-lidar'):
                for i, x in enumerate(xxs):
                    plt.text(x, errs[i], str(cnts[i]), fontsize=10)
    if not stereo:
        plt.plot(xx, get_task_error(xx), '--', label="Task error", color='lightgreen', linewidth=2.5)
    # if stereo:
    #     yy_stereo = get_pixel_error(xx)
    #     plt.plot(xx, yy_stereo, linewidth=1.4, color='k', label='Pixel error')

    plt.legend(loc='upper left')
    if save:
        plt.tight_layout()
        mode = 'stereo' if stereo else 'mono'
        path_fig = os.path.join(dir_out, 'results_' + mode + '.png')
        plt.savefig(path_fig)
        print("Figure of results " + mode + " saved in {}".format(path_fig))
    if show:
        plt.show()
    plt.close('all')


def show_spread(dic_stats, clusters, show=False, save=False):
    """Predicted confidence intervals and task error as a function of ground-truth distance"""

    phase = 'test'
    dir_out = 'docs'
    excl_clusters = ['all', 'easy', 'moderate', 'hard']
    clusters = [clst for clst in clusters if clst not in excl_clusters]
    x_min = 3
    x_max = 42
    y_min = 0

    for method in ('monoloco_pp', 'monstereo'):
        plt.figure(2)
        xxs = get_distances(clusters)
        bbs = np.array([dic_stats[phase][method][key]['std_ale'] for key in clusters[:-1]])
        if method == 'monoloco_pp':
            y_max = 5
            color = 'deepskyblue'
            epis = np.array([dic_stats[phase][method][key]['std_epi'] for key in clusters[:-1]])
            plt.plot(xxs, epis, marker='o', color='coral', label="Combined uncertainty (\u03C3)")
        else:
            y_max = 3.5
            color = 'b'
            plt.plot(xx, get_pixel_error(xx), linewidth=1.4, color='k', label='Pixel error')
        plt.plot(xxs, bbs, marker='s', color=color, label="Aleatoric uncertainty (b)")
        xx = np.linspace(x_min, x_max, 100)
        plt.plot(xx, get_task_error(xx), '--', label="Task error (monocular bound)", color='lightgreen', linewidth=2.5)

        plt.xlabel("Ground-truth distance [m]")
        plt.ylabel("Uncertainty [m]")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(linewidth=0.2)
        plt.legend()
        if save:
            plt.tight_layout()
            path_fig = os.path.join(dir_out, 'spread_' + method + '.png')
            plt.savefig(path_fig)
            print("Figure of confidence intervals saved in {}".format(path_fig))
        if show:
            plt.show()
        plt.close('all')


def show_task_error(show, save):
    """Task error figure"""
    plt.figure(3)
    dir_out = 'docs'
    xx = np.linspace(0.1, 50, 100)
    mu_men = 178
    mu_women = 165
    mu_child_m = 164
    mu_child_w = 156
    mm_gmm, mm_male, mm_female = calculate_gmm()
    mm_young_male = mm_male + (mu_men - mu_child_m) / mu_men
    mm_young_female = mm_female + (mu_women - mu_child_w) / mu_women
    yy_male = target_error(xx, mm_male)
    yy_female = target_error(xx, mm_female)
    yy_young_male = target_error(xx, mm_young_male)
    yy_young_female = target_error(xx, mm_young_female)
    yy_gender = target_error(xx, mm_gmm)
    yy_stereo = get_pixel_error(xx)
    plt.grid(linewidth=0.3)
    plt.plot(xx, yy_young_male, linestyle='dotted', linewidth=2.1, color='b', label='Adult/young male')
    plt.plot(xx, yy_young_female, linestyle='dotted', linewidth=2.1, color='darkorange', label='Adult/young female')
    plt.plot(xx, yy_gender, '--', color='lightgreen', linewidth=2.8, label='Generic adult (task error)')
    plt.plot(xx, yy_female, '-.', linewidth=1.7, color='darkorange', label='Adult female')
    plt.plot(xx, yy_male, '-.', linewidth=1.7, color='b', label='Adult male')
    plt.plot(xx, yy_stereo, linewidth=1.7, color='k', label='Pixel error')
    plt.xlim(np.min(xx), np.max(xx))
    plt.xlabel("Ground-truth distance from the camera $d_{gt}$ [m]")
    plt.ylabel("Localization error $\hat{e}$  due to human height variation [m]")  # pylint: disable=W1401
    plt.legend(loc=(0.01, 0.55))  # Location from 0 to 1 from lower left
    if save:
        path_fig = os.path.join(dir_out, 'task_error.png')
        plt.savefig(path_fig)
        print("Figure of task error saved in {}".format(path_fig))
    if show:
        plt.show()
    plt.close('all')


def show_method(save):
    """ method figure"""
    dir_out = 'docs'
    std_1 = 0.75
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ell_3 = Ellipse((0, 2), width=std_1 * 2, height=0.3, angle=-90, color='b', fill=False, linewidth=2.5)
    ell_4 = Ellipse((0, 2), width=std_1 * 3, height=0.3, angle=-90, color='r', fill=False,
                    linestyle='dashed', linewidth=2.5)
    ax.add_patch(ell_4)
    ax.add_patch(ell_3)
    plt.plot(0, 2, marker='o', color='skyblue', markersize=9)
    plt.plot([0, 3], [0, 4], 'k--')
    plt.plot([0, -3], [0, 4], 'k--')
    plt.xlim(-3, 3)
    plt.ylim(0, 3.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    if save:
        path_fig = os.path.join(dir_out, 'output_method.png')
        plt.savefig(path_fig)
        print("Figure of method saved in {}".format(path_fig))


def show_box_plot(dic_errors, clusters, show=False, save=False):
    import pandas as pd
    dir_out = 'docs'
    excl_clusters = ['all', 'easy', 'moderate', 'hard']
    clusters = [int(clst) for clst in clusters if clst not in excl_clusters]
    methods = ('monstereo', 'pseudo-lidar', '3dop', 'monoloco')
    y_min = 0
    y_max = 25  # 18 for the other
    xxs = get_distances(clusters)
    labels = [str(xx) for xx in xxs]
    for idx, method in enumerate(methods):
        df = pd.DataFrame([dic_errors[method][str(clst)] for clst in clusters[:-1]]).T
        df.columns = labels

        plt.figure(idx)
        _ = df.boxplot()
        name = 'MonStereo' if method == 'monstereo' else method
        plt.title(name)
        plt.ylabel('Average localization error (ALE) [m]')
        plt.xlabel('Ground-truth distance [m]')
        plt.ylim(y_min, y_max)

        if save:
            path_fig = os.path.join(dir_out, 'box_plot_' + name + '.png')
            plt.tight_layout()
            plt.savefig(path_fig)
            print("Figure of box plot saved in {}".format(path_fig))
        if show:
            plt.show()
        plt.close('all')


def target_error(xx, mm):
    return mm * xx


def calculate_gmm():
    dist_gmm, dist_male, dist_female = height_distributions()
    # get_percentile(dist_gmm)
    mu_gmm = np.mean(dist_gmm)
    mm_gmm = np.mean(np.abs(1 - mu_gmm / dist_gmm))
    mm_male = np.mean(np.abs(1 - np.mean(dist_male) / dist_male))
    mm_female = np.mean(np.abs(1 - np.mean(dist_female) / dist_female))

    print("Mean of GMM distribution: {:.4f}".format(mu_gmm))
    print("coefficient for gmm: {:.4f}".format(mm_gmm))
    print("coefficient for men: {:.4f}".format(mm_male))
    print("coefficient for women: {:.4f}".format(mm_female))
    return mm_gmm, mm_male, mm_female


def get_confidence(xx, zz, std):
    theta = math.atan2(zz, xx)

    delta_x = std * math.cos(theta)
    delta_z = std * math.sin(theta)
    return (xx - delta_x, xx + delta_x), (zz - delta_z, zz + delta_z)


def get_distances(clusters):
    """Extract distances as intermediate values between 2 clusters"""
    distances = []
    for idx, _ in enumerate(clusters[:-1]):
        clst_0 = float(clusters[idx])
        clst_1 = float(clusters[idx + 1])
        distances.append((clst_1 - clst_0) / 2 + clst_0)
    return tuple(distances)


def get_confidence_points(confidences, distances, errors):
    confidence_points = []
    distance_points = []
    for idx, dd in enumerate(distances):
        conf_perc = confidences[idx]
        confidence_points.append(errors[idx] + conf_perc)
        confidence_points.append(errors[idx] - conf_perc)
        distance_points.append(dd)
        distance_points.append(dd)

    return distance_points, confidence_points


def height_distributions():
    mu_men = 178
    std_men = 7
    mu_women = 165
    std_women = 7
    dist_men = np.random.normal(mu_men, std_men, int(1e7))
    dist_women = np.random.normal(mu_women, std_women, int(1e7))

    dist_gmm = np.concatenate((dist_men, dist_women))
    return dist_gmm, dist_men, dist_women


def expandgrid(*itrs):
    mm = 0
    combinations = list(itertools.product(*itrs))

    for h_i, h_gt in combinations:
        mm += abs(float(1 - h_i / h_gt))

    mm /= len(combinations)

    return combinations


def get_percentile(dist_gmm):
    dd_gt = 1000
    mu_gmm = np.mean(dist_gmm)
    dist_d = dd_gt * mu_gmm / dist_gmm
    perc_d, _ = np.nanpercentile(dist_d, [18.5, 81.5])  # Laplace bi => 63%
    perc_d2, _ = np.nanpercentile(dist_d, [23, 77])
    mu_d = np.mean(dist_d)
    # mm_bi = (mu_d - perc_d) / mu_d
    # mm_test = (mu_d - perc_d2) / mu_d
    # mad_d = np.mean(np.abs(dist_d - mu_d))


def printing_styles(stereo):
    if stereo:
        style = {"labels": ['3DOP', 'PSF', 'MonoLoco', 'MonoPSR', 'Pseudo-Lidar', 'Our MonStereo'],
                 "methods": ['3dop', 'psf', 'monoloco', 'monopsr', 'pseudo-lidar', 'monstereo'],
                 "mks": ['s', 'p', 'o', 'v', '*', '^'],
                 "mksizes": [6, 6, 6, 6, 6, 6], "lws": [1.2, 1.2, 1.2, 1.2, 1.3, 1.5],
                 "colors": ['gold', 'skyblue', 'darkgreen', 'pink', 'darkorange', 'b'],
                 "lstyles": ['solid', 'solid', 'dashed', 'dashed', 'solid', 'solid']}
    else:
        style = {"labels": ['Mono3D', 'Geometric Baseline', 'MonoPSR', '3DOP (stereo)', 'MonoLoco', 'Monoloco++'],
                 "methods": ['m3d', 'geometric', 'monopsr', '3dop', 'monoloco', 'monoloco_pp'],
                 "mks": ['*', '^', 'p', '.', 's', 'o', 'o'],
                 "mksizes": [6, 6, 6, 6, 6, 6], "lws": [1.5, 1.5, 1.5, 1.5, 1.5, 2.2],
                 "colors": ['r', 'purple', 'olive', 'darkorange', 'b', 'darkblue'],
                 "lstyles": ['solid', 'solid', 'solid', 'dashdot', 'solid', 'solid', ]}

    return style
