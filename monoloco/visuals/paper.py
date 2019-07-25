

import math
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def paper():
    """Print paper figures"""

    method = False
    task_error = True

    # Pull figure
    if method:
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
        plt.savefig(os.path.join('docs', 'output_method.png'))

    # Task error figure
    if task_error:
        plt.figure(2)
        xx = np.linspace(0, 40, 100)
        mu_men = 178
        mu_women = 165
        mu_child_m = 164
        mu_child_w = 156
        mm_gmm, mm_male, mm_female = gmm()
        mm_young_male = mm_male + (mu_men - mu_child_m) / mu_men
        mm_young_female = mm_female + (mu_women - mu_child_w) / mu_women
        yy_male = target_error(xx, mm_male)
        yy_female = target_error(xx, mm_female)
        yy_young_male = target_error(xx, mm_young_male)
        yy_young_female = target_error(xx, mm_young_female)
        yy_gender = target_error(xx, mm_gmm)
        yy_gps = np.linspace(5., 5., xx.shape[0])
        plt.grid(linewidth=0.3)
        plt.plot(xx, yy_gps, color='y', label='GPS')
        plt.plot(xx, yy_young_male, linestyle='dotted', linewidth=2.1, color='b', label='Adult/young male')
        plt.plot(xx, yy_young_female, linestyle='dotted', linewidth=2.1, color='darkorange', label='Adult/young female')
        plt.plot(xx, yy_gender, '--', color='lightgreen', linewidth=2.8, label='Generic adult (task error)')
        plt.plot(xx, yy_female, '-.', linewidth=1.7, color='darkorange', label='Adult female')
        plt.plot(xx, yy_male, '-.', linewidth=1.7, color='b', label='Adult male')
        plt.xlim(np.min(xx), np.max(xx))
        plt.xlabel("Ground-truth distance from the camera $d_{gt}$ [m]")
        plt.ylabel("Localization error $\hat{e}$  due to human height variation [m]")
        plt.legend(loc=(0.01, 0.55))  # Location from 0 to 1 from lower left
        plt.savefig(os.path.join('docs', 'task_error.png'))


def target_error(xx, mm):
    return mm * xx


def gmm():
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


def plot_dist(dist_gmm, dist_men, dist_women):
    try:
        import seaborn as sns
        sns.distplot(dist_men, hist=False, rug=False, label="Men")
        sns.distplot(dist_women, hist=False, rug=False, label="Women")
        sns.distplot(dist_gmm, hist=False, rug=False, label="GMM")
        plt.xlabel("X [cm]")
        plt.ylabel("Height distributions of men and women")
        plt.legend()
        plt.show()
        plt.close()
    except ImportError:
        print("Import Seaborn first")


def get_percentile(dist_gmm):
    dd_gt = 1000
    mu_gmm = np.mean(dist_gmm)
    dist_d = dd_gt * mu_gmm / dist_gmm
    perc_d, _ = np.nanpercentile(dist_d, [18.5, 81.5]) # Laplace bi => 63%
    perc_d2, _ = np.nanpercentile(dist_d, [23, 77])
    mu_d = np.mean(dist_d)
    # mm_bi = (mu_d - perc_d) / mu_d
    # mm_test = (mu_d - perc_d2) / mu_d
    # mad_d = np.mean(np.abs(dist_d - mu_d))
