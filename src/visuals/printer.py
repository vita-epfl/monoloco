
import math
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse, Circle

from collections import OrderedDict
from utils.camera import pixel_to_camera
from utils.misc import get_task_error


class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    RADIUS_KPS = 6
    FONTSIZE_BV = 16
    FONTSIZE = 18
    TEXTCOLOR = 'darkorange'
    COLOR_KPS = 'yellow'

    def __init__(self, image, output_path, dic_ann, kk, output_types, show=False,
                 draw_kps=False, text=True, legend=True, epistemic=False, z_max=30, fig_width=10):

        self.im = image
        self.kk = kk
        self.output_types = output_types
        self.show = show
        self.draw_kps = draw_kps
        self.text = text
        self.epistemic = epistemic
        self.legend = legend
        self.z_max = z_max  # To include ellipses in the image
        self.fig_width = fig_width


        # Define the output dir
        self.path_out = output_path

        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_ale_epi = dic_ann['stds_epi']
        self.xx_gt = [xx[0] for xx in dic_ann['xyz_real']]
        self.zz_gt = [xx[2] if xx[2] < self.z_max - self.stds_ale_epi[idx] else 0
                      for idx, xx in enumerate(dic_ann['xyz_real'])]
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_ale_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]
        self.dds_real = dic_ann['dds_real']

        self.uv_centers = dic_ann['uv_centers']
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.uv_kps = dic_ann['uv_kps']

        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.ww = self.im.size[0]
        self.hh = self.im.size[1]
        self.radius = 14 / 1600 * self.ww

    def print(self):
        """
        Show frontal, birds-eye-view or combined visualization
        With or without ground truth
        Either front and/or bird visualization or combined one
        """
        # Parameters

        # Resize image for aesthetic proportions in combined visualization
        if 'combined' in self.output_types:
            y_scale = self.ww / (self.hh * 1.8)  # Defined proportion
            self.im = self.im.resize((self.ww, round(self.hh * y_scale)))
            self.ww = self.im.size[0]
            self.hh = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width
            fig_height = self.fig_width * self.hh / self.ww

            # Distinguish between KITTI images and general images
            if y_scale > 1.7:
                fig_ar_1 = 1.7
            else:
                fig_ar_1 = 1.3
            width_ratio = 1.9
            ext = '.combined.png'

            fig, (ax1, ax0) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [1, width_ratio]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--combined arguments is not supported with other visualizations"

        # Initialize front
        elif 'front' in self.output_types:
            y_scale = 1
            width = self.fig_width
            height = self.fig_width * self.hh / self.ww

            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)

        # Create front
        if any(xx in self.output_types for xx in ['front', 'combined']):

            ax0.set_axis_off()
            ax0.set_xlim(0, self.ww)
            ax0.set_ylim(self.hh, 0)
            ax0.imshow(self.im)
            z_min = 0
            bar_ticks = self.z_max // 5 + 1

            cmap = cm.get_cmap('jet')
            num = 0
            for idx, uv in enumerate(self.uv_shoulders):

                if self.draw_kps:
                    ax0 = self.show_kps(ax0, self.uv_kps[idx], y_scale, self.RADIUS_KPS, self.COLOR_KPS)

                elif min(self.zz_pred[idx], self.zz_gt[idx]) > 0:
                    color = cmap((self.zz_pred[idx] % self.z_max) / self.z_max)
                    circle = Circle((uv[0], uv[1] * y_scale), radius=self.radius, color=color, fill=True)
                    ax0.add_patch(circle)

                    if self.text:
                        ax0.text(uv[0]+self.radius, uv[1] * y_scale - self.radius, str(num),
                                 fontsize=self.FONTSIZE, color=self.TEXTCOLOR, weight='bold')
                        num += 1

            ax0.get_xaxis().set_visible(False)
            ax0.get_yaxis().set_visible(False)

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='3%', pad=0.05)

            norm = matplotlib.colors.Normalize(vmin=z_min, vmax=self.z_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ticks=np.linspace(z_min, self.z_max, bar_ticks),
                         boundaries=np.arange(z_min - 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')
        # Save Front
        if 'front' in self.output_types:
            if self.show:
                plt.show()
            else:
                plt.savefig(self.path_out + '.front.png', bbox_inches='tight')

        # Initialize Bird
        if 'bird' in self.output_types:
            plt.close()
            plt.figure(1)
            ext = '.bird.png'
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)

        # Create bird or combine it with front)
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            uv_max = [0., float(self.hh)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk

            for idx, _ in enumerate(self.xx_gt):
                if self.zz_gt[idx] > 0:
                    target = get_task_error(self.dds_real[idx])

                    angle = get_angle(self.xx_gt[idx], self.zz_gt[idx])
                    ellipse_real = Ellipse((self.xx_gt[idx], self.zz_gt[idx]), width=target * 2, height=1,
                                           angle=angle, color='lightgreen', fill=True, label="Task error")
                    ax1.add_patch(ellipse_real)
                    ax1.plot(self.xx_gt[idx], self.zz_gt[idx], 'kx', label="Ground truth", markersize=3)

            # Print prediction and the real ground truth. Color of prediction depends if ground truth exists
            num = 0
            for idx, _ in enumerate(self.xx_pred):
                if self.zz_gt[idx] > 0:  # only the merging ones and inside the interval

                    angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
                    ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
                                          height=1, angle=angle, color='b', fill=False, label="Aleatoric Uncertainty",
                                          linewidth=1.3)
                    ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale_epi[idx] * 2,
                                          height=1, angle=angle, color='r', fill=False, label="Uncertainty",
                                          linewidth=1, linestyle='--')

                    ax1.add_patch(ellipse_ale)
                    if self.epistemic:
                        ax1.add_patch(ellipse_var)

                    ax1.plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

                    # Plot the number
                    if not self.draw_kps:
                        (_, x_pos), (_, z_pos) = get_confidence(self.xx_pred[idx], self.zz_pred[idx], self.stds_ale_epi[idx])

                        if self.text:
                            ax1.text(x_pos, z_pos, str(num), fontsize=self.FONTSIZE_BV, color='darkorange')
                            num += 1

            # To avoid repetitions in the legend
            if self.legend:
                handles, labels = ax1.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())

            # Adding field of view
            ax1.plot([0, x_max], [0, self.z_max], 'k--')
            ax1.plot([0, -x_max], [0, self.z_max], 'k--')
            ax1.set_ylim(0, self.z_max+1)
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Z [m]")

            # TO remove axis numbers
            # plt.setp([ax1.get_yticklabels() for aa in fig.axes[:-1]], visible=False)
            # plt.setp([ax1.get_xticklabels() for aa in fig.axes[:-1]], visible=False)

            if self.show:
                plt.show()
            else:
                plt.savefig(self.path_out + ext, bbox_inches='tight')

            if self.draw_kps:
                im = cv2.imread(self.path_out + ext)
                im = self.increase_brightness(im, value=30)
                im_new = im[0 : self.hh, 0:round(self.ww / 1.7)]
                cv2.imwrite(self.path_out, im_new)

        plt.close('all')

    def show_kps(self, ax0, uv_kp_single, y_scale, radius, color):

        for idx, uu in enumerate(uv_kp_single[0]):
            vv = uv_kp_single[1][idx]
            circle = Circle((uu, vv * y_scale), radius=radius, color=color, fill=True)
            ax0.add_patch(circle)

        return ax0

    @staticmethod
    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


def get_confidence(xx, zz, std):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)

    delta_x = std * math.cos(theta)
    delta_z = std * math.sin(theta)

    return (xx - delta_x, xx + delta_x), (zz - delta_z, zz + delta_z)


def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle
