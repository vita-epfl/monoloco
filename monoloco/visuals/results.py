# pylint: disable=R0915

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def print_results(dic_stats, show=False):

    """
    Visualize error as function of the distance on the test set and compare it with target errors based on human
    height analyses. We consider:
    Position error in meters due to a height variation of 7 cm (Standard deviation already knowing the sex)
    Position error not knowing the gender (13cm as average difference --> 7.5cm of error to add)
    """

    # ALE figure
    dir_out = 'docs'
    phase = 'test'
    x_min = 0
    x_max = 38
    xx = np.linspace(0, 60, 100)
    mm_gender = 0.0556
    excl_clusters = ['all', '50', '>50', 'easy', 'moderate', 'hard']
    clusters = tuple([clst for clst in dic_stats[phase]['our'] if clst not in excl_clusters])
    yy_gender = target_error(xx, mm_gender)
    yy_gps = np.linspace(5., 5., xx.shape[0])

    plt.figure(0)
    fig_name = 'results.png'
    plt.xlabel("Distance [meters]")
    plt.ylabel("Average localization error [m]")
    plt.xlim(x_min, x_max)
    labels = ['Mono3D', 'Geometric Baseline', 'MonoDepth', 'Our MonoLoco', '3DOP (stereo)']
    mks = ['*', '^', 'p', 's', 'o']
    mksizes = [6, 6, 6, 6, 6]
    lws = [1.5, 1.5, 1.5, 2.2, 1.6]
    colors = ['r', 'deepskyblue', 'grey', 'b', 'darkorange']
    lstyles = ['solid', 'solid', 'solid', 'solid', 'dashdot']

    plt.plot(xx, yy_gps, '-', label="GPS Error", color='y')
    for idx, method in enumerate(['m3d_merged', 'geom_merged', 'md_merged', 'our_merged', '3dop_merged']):
        errs = [dic_stats[phase][method][clst]['mean'] for clst in clusters]
        xxs = get_distances(clusters)

        plt.plot(xxs, errs, marker=mks[idx], markersize=mksizes[idx], linewidth=lws[idx], label=labels[idx],
                 linestyle=lstyles[idx], color=colors[idx])
    plt.plot(xx, yy_gender, '--', label="Task error", color='lightgreen', linewidth=2.5)
    plt.legend(loc='upper left')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(dir_out, fig_name))
    plt.close()

    # SPREAD b Figure
    plt.figure(1)
    fig, ax = plt.subplots(2, sharex=True)
    plt.xlabel("Distance [m]")
    plt.ylabel("Aleatoric uncertainty [m]")
    ar = 0.5  # Change aspect ratio of ellipses
    scale = 1.5  # Factor to scale ellipses
    rec_c = 0  # Center of the rectangle
    plots_line = True

    bbs = np.array([dic_stats[phase]['our'][key]['std_ale'] for key in clusters])
    xxs = get_distances(clusters)
    yys = target_error(np.array(xxs), mm_gender)
    ax[1].plot(xxs, bbs, marker='s', color='b', label="Spread b")
    ax[1].plot(xxs, yys, '--', color='lightgreen', label="Task error", linewidth=2.5)
    yys_up = [rec_c + ar/2 * scale * yy for yy in yys]
    bbs_up = [rec_c + ar/2 * scale * bb for bb in bbs]
    yys_down = [rec_c - ar/2 * scale * yy for yy in yys]
    bbs_down = [rec_c - ar/2 * scale * bb for bb in bbs]

    if plots_line:
        ax[0].plot(xxs, yys_up, '--', color='lightgreen', markersize=5, linewidth=1)
        ax[0].plot(xxs, yys_down, '--', color='lightgreen', markersize=5, linewidth=1)
        ax[0].plot(xxs, bbs_up, marker='s', color='b', markersize=5, linewidth=0.7)
        ax[0].plot(xxs, bbs_down, marker='s', color='b', markersize=5, linewidth=0.7)

    for idx, xx in enumerate(xxs):
        te = Ellipse((xx, rec_c), width=yys[idx]*ar*scale, height=scale, angle=90, color='lightgreen', fill=True)
        bi = Ellipse((xx, rec_c), width=bbs[idx]*ar*scale, height=scale, angle=90, color='b', linewidth=1.8,
                     fill=False)

        ax[0].add_patch(te)
        ax[0].add_patch(bi)

    fig.subplots_adjust(hspace=0.1)
    plt.setp([aa.get_yticklabels() for aa in fig.axes[:-1]], visible=False)
    plt.legend()
    if show:
        plt.show()
    plt.close()


def target_error(xx, mm):
    """Multiplication"""
    return mm * xx


def get_distances(clusters):
    """Extract distances as intermediate values between 2 clusters"""

    clusters_ext = list(clusters)
    clusters_ext.insert(0, str(0))
    distances = []
    for idx, _ in enumerate(clusters_ext[:-1]):
        clst_0 = float(clusters_ext[idx])
        clst_1 = float(clusters_ext[idx + 1])
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
