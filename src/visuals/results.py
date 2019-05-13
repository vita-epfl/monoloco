
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def print_results(dic_stats, show=False, save=False):

    """
    Visualize error as function of the distance on the test set and compare it with target errors based on human
    height analyses. We consider:
    Position error in meters due to a height variation of 7 cm (Standard deviation already knowing the sex)
    Position error not knowing the gender (13cm as average difference --> 7.5cm of error to add)
    """
    dir_out = 'docs'
    phase = 'test'
    x_min = 0
    x_max = 38
    xx = np.linspace(0, 60, 100)
    mm_std = 0.04
    mm_gender = 0.0556
    excl_clusters = ['all', '50', '>50', 'easy', 'moderate', 'hard']
    clusters = tuple([clst for clst in dic_stats[phase]['our']['mean'] if clst not in excl_clusters])
    yy_gender = target_error(xx, mm_gender)
    yy_gps = np.linspace(5., 5., xx.shape[0])

    # # Recall results
    # plt.figure(1)
    # plt.xlabel("Distance [meters]")
    # plt.ylabel("Detected instances")
    # plt.xlim(x_min, x_max)
    # for method in ['our', 'm3d', '3dop']:
    #     xxs = get_distances(clusters)
    #     dic_cnt = dic_stats[phase][method]['cnt']
    #     cnts = get_values(dic_cnt, clusters)
    #     plt.plot(xxs, cnts, marker='s', label=method + '_method')
    # plt.legend()
    # plt.show()
    #
    # Precision on same instances
    fig_name = 'results.png'
    plt.xlabel("Distance [meters]")
    plt.ylabel("Average localization error [m]")
    plt.xlim(x_min, x_max)
    labels = ['Mono3D', 'Geometric Baseline', 'MonoDepth', 'Our MonoLoco', '3DOP']
    mks = ['*', '^', 'p', 's', 'o']
    mksizes = [6, 6, 6, 6, 6]
    lws = [1.5, 1.5, 1.5, 2.2, 1.6]
    colors = ['r', 'deepskyblue', 'grey', 'b', 'darkorange']
    lstyles = ['solid', 'solid', 'solid', 'solid', 'dashdot']

    plt.plot(xx, yy_gps, '-', label="GPS Error", color='y')
    for idx, method in enumerate(['m3d_merged', 'geom_merged', 'md_merged', 'our_merged', '3dop_merged']):
        dic_errs = dic_stats[phase][method]['mean']
        errs = get_values(dic_errs, clusters)
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

    # FIGURE SPREAD
    fig_name = 'spread.png'
    # fig = plt.figure(3)
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(2, sharex=True)
    plt.xlabel("Distance [m]")
    plt.ylabel("Aleatoric uncertainty [m]")
    ar = 0.5  # Change aspect ratio of ellipses
    scale = 1.5  # Factor to scale ellipses
    rec_c = 0  # Center of the rectangle
    # rec_h = 2.8  # Height of the rectangle
    plots_line = True
    # ax[0].set_ylim([-3, 3])
    # ax[1].set_ylim([0, 3])
    # ax[1].set_ylabel("Aleatoric uncertainty [m]")
    # ax[0].set_ylabel("Confidence intervals")

    dic_ale = dic_stats[phase]['our']['std_ale']
    bbs = np.array(get_values(dic_ale, clusters))
    xxs = get_distances(clusters)
    yys = target_error(np.array(xxs), mm_gender)
    # ale_het = tuple(bbs - yys)
    # plt.plot(xxs, ale_het, marker='s', label=method)
    ax[1].plot(xxs, bbs, marker='s', color='b', label="Spread b")
    ax[1].plot(xxs, yys,  '--', color='lightgreen', label="Task error", linewidth=2.5)
    yys_up = [rec_c + ar/2 * scale * yy for yy in yys]
    bbs_up = [rec_c + ar/2 * scale * bb for bb in bbs]
    yys_down = [rec_c - ar/2 * scale * yy for yy in yys]
    bbs_down = [rec_c - ar/2 * scale * bb for bb in bbs]

    if plots_line:
        ax[0].plot(xxs, yys_up, '--', color='lightgreen', markersize=5, linewidth=1)
        ax[0].plot(xxs, yys_down, '--', color='lightgreen', markersize=5, linewidth=1)
        ax[0].plot(xxs, bbs_up, marker='s', color='b', markersize=5, linewidth=0.7)
        ax[0].plot(xxs, bbs_down, marker='s', color='b', markersize=5, linewidth=0.7)

    # rectangle = Rectangle((2, rec_c - (rec_h/2)), width=34, height=rec_h, color='black', fill=False)

    for idx, xx in enumerate(xxs):
        te = Ellipse((xx, rec_c), width=yys[idx]*ar*scale, height=scale, angle=90, color='lightgreen', fill=True)
        bi = Ellipse((xx, rec_c), width=bbs[idx]*ar*scale, height=scale, angle=90, color='b',linewidth=1.8,
                     fill=False)

        # ax[0].add_patch(rectangle)
        ax[0].add_patch(te)
        ax[0].add_patch(bi)

    fig.subplots_adjust(hspace=0.1)
    plt.setp([aa.get_yticklabels() for aa in fig.axes[:-1]], visible=False)
    plt.legend()
    if show:
        plt.show()
    plt.close()


def target_error(xx, mm):
    return mm * xx


def get_distances(clusters):

    clusters_ext = list(clusters)
    clusters_ext.insert(0, str(0))
    distances = []
    for idx, _ in enumerate(clusters_ext[:-1]):
        clst_0 = float(clusters_ext[idx])
        clst_1 = float(clusters_ext[idx + 1])
        distances.append((clst_1 - clst_0) / 2 + clst_0)
    return tuple(distances)


def get_values(dic_err, clusters):

    errs = []
    for key in clusters:
        errs.append(dic_err[key])
    return errs


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



# def histogram(self):
#     """
#     Visualize histograms to compare error performances of net and baseline
#     """
#     # for mode in ['mean', 'std']:
#     for mode in ['mean']:
#
#         # for phase in ['train', 'val', 'test']:
#         for phase in ['test']:
#
#             err_geom = []
#             err_net = []
#             counts = []
#             clusters = ('10', '20', '30', '40', '50', '>50', 'all')
#             for clst in clusters:
#
#                 err_geom.append(dic_geom[phase]['error'][clst][mode])
#                 counts.append(dic_geom[phase]['error'][clst]['count'])
#                 err_net.append(dic_net[phase][clst][mode])
#
#             nn = len(clusters)
#
#             # TODO Shortcut to print 2 models in a single histogram
#             err_net_l1 = [1.1, 1.32, 2.19, 3.29, 4.38, 6.58, 2.21]
#
#             ind = np.arange(nn)  # the x locations for the groups
#             width = 0.30  # the width of the bars
#
#             fig, ax = plt.subplots()
#             rects1 = ax.bar(ind, err_geom, width, color='b')
#             rects2 = ax.bar(ind + width, err_net_l1, width, color='r')
#             rects3 = ax.bar(ind + 2 * width, err_net, width, color='g')
#
#             # add some text for labels, title and axes ticks
#             ax.set_ylabel('Distance error [m]')
#             ax.set_title(mode + ' of errors in ' + phase)
#             ax.set_xticks(ind + width / 2)
#             ax.set_xticklabels(clusters)
#
#             ax.legend((rects1[0], rects2[0], rects3[0]), ('Geometrical', 'L1 Loss', 'Laplacian Loss'))
#
#             # Attach a text label above each bar displaying number of annotations
#             for idx, rect in enumerate(rects1):
#                 height = rect.get_height()
#                 count = counts[idx]
#                 ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
#                         '%d' % int(count), ha='center', va='bottom')
#
#         plt.show()

    # def error(self):
#     """
#     Visualize error as function of the distance on the test set and compare it with target errors based on human
#     height analyses. We consider:
#     Position error in meters due to a height variation of 7 cm (Standard deviation already knowing the sex)
#     Position error not knowing the gender (13cm as average difference --> 7.5cm of error to add)
#     """
#     phase = 'test'
#     xx = np.linspace(0, 60, 100)
#     mm_std = 0.04
#     mm_gender = 0.0556
#     clusters = tuple([clst for clst in dic_net[phase] if clst not in ['all', '>50']])
#     # errors_geom = [dic_geom[phase]['error'][clst][mode] for clst in clusters]
#     errors_net = [dic_net[phase][clst]['mean'] for clst in clusters]
#     confidences = [dic_net[phase][clst]['std'][0] for clst in clusters]
#     distances = get_distances(clusters)
#     dds, zzs = get_confidence_points(confidences, distances, errors_net)
#
#     # Set the plot
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     plt.xlim(0, 50)
#     plt.xlabel("Distance [meters]")
#     plt.ylabel("Error [meters]")
#     plt.title("Error on Z Position Estimate due to height variation")
#
#     # Plot the target errors
#     yy_std = target_error(xx, mm_std)
#     yy_gender = target_error(xx, mm_gender)
#     # yy_child = target_error(xx, mm_child)
#     # yy_test = target_error(xx, mm_test)
#     yy_gps = np.linspace(5., 5., xx.shape[0])
#     plt.plot(xx, yy_std, '--', label="Knowing the gender", color='g')
#     plt.plot(xx, yy_gender, '--', label="NOT knowing the gender", color='b')
#     plt.plot(xx, yy_gps, '-', label="GPS Error", color='y')
#
#     for idx in range(0, len(dds), 2):
#
#         plt.plot(dds[idx: idx+2], zzs[idx: idx+2], color='k')
#
#     # Plot the geometric and net errors as dots
#     _ = ax.plot(distances, errors_net, 'ro', marker='s', label="Network Error (test)", markersize='8')
#     # _ = ax.plot(distances, errors_geom, 'ko', marker='o', label="Baseline Error", markersize='5')
#
#     ax.legend(loc='upper left')
#     plt.show()