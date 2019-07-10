
import math
from collections import OrderedDict
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    def __init__(self, image, output_path, kk, output_types, text=True, legend=True, epistemic=False,
                 z_max=30, fig_width=10, box=True):

        self.im = image
        self.kk = kk
        self.output_types = output_types
        self.text = text
        self.epistemic = epistemic
        self.legend = legend
        self.z_max = z_max  # To include ellipses in the image
        self.y_scale = 1
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.fig_width = fig_width
        self.box = box

        # Define the output dir
        self.path_out = output_path
        self.cmap = cm.get_cmap('jet')
        self.extensions = []
        self.mpl_im0 = None

    def _process_results(self, dic_ann):
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
        self.boxes = dic_ann['boxes']
        self.boxes_gt = dic_ann['boxes_gt']

        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.radius = 11 / 1600 * self.width

    def factory_axes(self):
        """Create axes for figures: front bird combined"""
        axes = []
        figures = []

        #  Initialize combined figure, resizing it for aesthetic proportions
        if 'combined' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 1.8)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width
            fig_height = self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            if self.y_scale > 1.7:
                fig_ar_1 = 1.7
            else:
                fig_ar_1 = 1.3
            width_ratio = 1.9
            self.extensions.append('.combined.png')

            fig, (ax1, ax0) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [1, width_ratio]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)
            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--combined arguments is not supported with other visualizations"

        # Initialize front figure
        elif 'front' in self.output_types:
            width = self.fig_width
            height = self.fig_width * self.height / self.width
            self.extensions.append(".front.png")
            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)

            figures.append(fig0)

        # Create front figure axis
        if any(xx in self.output_types for xx in ['front', 'combined']):

            ax0.set_axis_off()
            ax0.set_xlim(0, self.width)
            ax0.set_ylim(self.height, 0)
            self.mpl_im0 = ax0.imshow(self.im)
            z_min = 0
            bar_ticks = self.z_max // 5 + 1
            ax0.get_xaxis().set_visible(False)
            ax0.get_yaxis().set_visible(False)

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='3%', pad=0.05)

            norm = matplotlib.colors.Normalize(vmin=z_min, vmax=self.z_max)
            scalar_mappable = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array([])
            plt.colorbar(scalar_mappable, ticks=np.linspace(z_min, self.z_max, bar_ticks),
                         boundaries=np.arange(z_min - 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')

            axes.append(ax0)
        if not axes:
            axes.append(None)

        if 'bird' in self.output_types:
            self.extensions.append(".bird.png")
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)
            figures.append(fig1)
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk

            # Adding field of view
            ax1.plot([0, x_max], [0, self.z_max], 'k--')
            ax1.plot([0, -x_max], [0, self.z_max], 'k--')
            ax1.set_ylim(0, self.z_max+1)
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Z [m]")
            axes.append(ax1)
        return figures, axes

    def draw(self, figures, axes, dic_out, image, save=False, show=False):

        self._process_results(dic_out)
        num = 0
        if any(xx in self.output_types for xx in ['front', 'combined']):
            self.mpl_im0.set_data(image)
            for idx, uv in enumerate(self.uv_shoulders):

                if min(self.zz_pred[idx], self.zz_gt[idx]) > 0:
                    color = self.cmap((self.zz_pred[idx] % self.z_max) / self.z_max)
                    circle = Circle((uv[0], uv[1] * self.y_scale), radius=self.radius, color=color, fill=True)
                    axes[0].add_patch(circle)

                    if self.box:
                        ww_box = self.boxes[idx][2] - self.boxes[idx][0]
                        hh_box = (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale
                        ww_box_gt = self.boxes_gt[idx][2] - self.boxes_gt[idx][0]
                        hh_box_gt = (self.boxes_gt[idx][3] - self.boxes_gt[idx][1]) * self.y_scale

                        rectangle = Rectangle((self.boxes[idx][0], self.boxes[idx][1] * self.y_scale),
                                             width=ww_box, height=hh_box, fill=False, color=color, linewidth=3)
                        rectangle_gt = Rectangle((self.boxes_gt[idx][0], self.boxes_gt[idx][1] * self.y_scale),
                                                 width=ww_box_gt, height=hh_box_gt, fill=False, color='g', linewidth=2)

                        axes[0].add_patch(rectangle_gt)
                        axes[0].add_patch(rectangle)


                    if self.text:
                        axes[0].text(uv[0]+self.radius, uv[1] * self.y_scale - self.radius, str(num),
                                     fontsize=self.FONTSIZE, color=self.TEXTCOLOR, weight='bold')
                        num += 1
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            for idx, _ in enumerate(self.xx_gt):
                if self.zz_gt[idx] > 0:
                    target = get_task_error(self.dds_real[idx])

                    angle = get_angle(self.xx_gt[idx], self.zz_gt[idx])
                    ellipse_real = Ellipse((self.xx_gt[idx], self.zz_gt[idx]), width=target * 2, height=1,
                                           angle=angle, color='lightgreen', fill=True, label="Task error")
                    axes[1].add_patch(ellipse_real)
                    if abs(self.zz_gt[idx] - self.zz_pred[idx]) > 0.001:
                        axes[1].plot(self.xx_gt[idx], self.zz_gt[idx], 'kx', label="Ground truth", markersize=3)

            # Print prediction and the real ground truth.
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

                    axes[1].add_patch(ellipse_ale)
                    if self.epistemic:
                        axes[1].add_patch(ellipse_var)

                    axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

                    # Setup the legend to avoid repetitions
                    if self.legend:
                        handles, labels = axes[1].get_legend_handles_labels()
                        by_label = OrderedDict(zip(labels, handles))
                        axes[1].legend(by_label.values(), by_label.keys())

                    # Plot the number
                    (_, x_pos), (_, z_pos) = get_confidence(self.xx_pred[idx], self.zz_pred[idx],
                                                            self.stds_ale_epi[idx])

                    if self.text:
                        axes[1].text(x_pos, z_pos, str(num), fontsize=self.FONTSIZE_BV, color='darkorange')
                        num += 1

        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            if save:
                fig.savefig(self.path_out + self.extensions[idx], bbox_inches='tight')
            if show:
                fig.show()


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
