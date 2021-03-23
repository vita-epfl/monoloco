"""
Class for drawing frontal, bird-eye-view and multi figures
"""
# pylint: disable=attribute-defined-outside-init
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..utils import pixel_to_camera


def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle


def image_attributes(dpi, output_types):
    c = 0.7 if 'front' in output_types else 1.0
    return dict(dpi=dpi,
                fontsize_d=round(14 * c),
                fontsize_bv=round(24 * c),
                fontsize_num=round(22 * c),
                fontsize_ax=round(16 * c),
                linewidth=round(8 * c),
                markersize=round(13 * c),
                y_box_margin=round(24 * math.sqrt(c)),
                stereo=dict(color='deepskyblue',
                            numcolor='darkorange',
                            linewidth=1 * c),
                mono=dict(color='red',
                          numcolor='firebrick',
                          linewidth=2 * c)
    )


class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    FIG_WIDTH = 15
    extensions = []
    y_scale = 1
    nones = lambda n: [None for _ in range(n)]
    mpl_im0, stds_ale, stds_epi, xx_gt, zz_gt, xx_pred, zz_pred, dd_real, uv_centers, uv_shoulders, uv_kps, boxes, \
        boxes_gt, uv_camera, radius, auxs = nones(16)

    def __init__(self, image, output_path, kk, args):

        self.im = image
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.output_path = output_path
        self.kk = kk
        self.output_types = args.output_types
        self.z_max = args.z_max  # set max distance to show instances
        self.show = args.show
        self.show_all = args.show_all
        self.save = not args.no_save

        # define image attributes
        self.attr = image_attributes(args.dpi, args.output_types)

    def _process_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_epi = dic_ann['stds_epi']
        self.gt = dic_ann['gt']  # regulate ground-truth matching
        self.xx_gt = [xx[0] for xx in dic_ann['xyz_real']]
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]

        # Set maximum distance
        self.dd_pred = dic_ann['dds_pred']
        self.dd_real = dic_ann['dds_real']
        self.z_max = int(min(self.z_max, 4 + max(max(self.dd_pred), max(self.dd_real, default=0))))

        # Do not print instances outside z_max
        self.zz_gt = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                      for idx, xx in enumerate(dic_ann['xyz_real'])]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]

        self.uv_heads = dic_ann['uv_heads']
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.boxes = dic_ann['boxes']
        self.boxes_gt = dic_ann['boxes_gt']
        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.auxs = dic_ann['aux']
        if len(self.auxs) == 0:
            self.modes = ['mono'] * len(self.dd_pred)
        else:
            self.modes = []
            for aux in self.auxs:
                if aux <= 0.3:
                    self.modes.append('mono')
                else:
                    self.modes.append('stereo')

    def factory_axes(self, dic_out):
        """Create axes for figures: front bird multi"""
        axes = []
        figures = []

        # Process the annotation dictionary of monoloco
        self._process_results(dic_out)

        #  Initialize multi figure, resizing it for aesthetic proportion
        if 'multi' in self.output_types:
            assert 'bird' not in self.output_types and 'front' not in self.output_types, \
                "multi figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.FIG_WIDTH + 0.6 * self.FIG_WIDTH
            fig_height = self.FIG_WIDTH * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.multi.png')

            fig, (ax0, ax1) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [width_ratio, 1]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)
            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--multi arguments is not supported with other visualizations"

        # Initialize front figure
        elif 'front' in self.output_types:
            width = self.FIG_WIDTH
            height = self.FIG_WIDTH * self.height / self.width
            self.extensions.append(".front.png")
            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)
            figures.append(fig0)

        # Create front figure axis
        if any(xx in self.output_types for xx in ['front', 'multi']):
            ax0 = self._set_axes(ax0, axis=0)
            axes.append(ax0)
        if not axes:
            axes.append(None)

        # Initialize bird-eye-view figure
        if 'bird' in self.output_types:
            self.extensions.append(".bird.png")
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)
            figures.append(fig1)
        if any(xx in self.output_types for xx in ['bird', 'multi']):
            ax1 = self._set_axes(ax1, axis=1)  # Adding field of view
            axes.append(ax1)
        return figures, axes

    def draw(self, figures, axes, image):

        # whether to include instances that don't match the ground-truth
        iterator = range(len(self.zz_pred)) if self.show_all else range(len(self.zz_gt))
        if not iterator:
            print("-" * 110 + '\n' + '! No instances detected' '\n' + '-' * 110)

        # Draw the front figure
        number = dict(flag=False, num=97)
        if any(xx in self.output_types for xx in ['front', 'multi']):
            number['flag'] = True  # add numbers
            self.mpl_im0.set_data(image)
        for idx in iterator:
            if any(xx in self.output_types for xx in ['front', 'multi']) and self.zz_pred[idx] > 0:
                self._draw_front(axes[0],
                                 self.dd_pred[idx],
                                 idx,
                                 number)
                number['num'] += 1

        # Draw the bird figure
        number['num'] = 97
        for idx in iterator:
            if any(xx in self.output_types for xx in ['bird', 'multi']) and self.zz_pred[idx] > 0:

                # Draw ground truth and uncertainty
                self._draw_uncertainty(axes, idx)

                # Draw bird eye view text
                if number['flag']:
                    self._draw_text_bird(axes, idx, number['num'])
                    number['num'] += 1
        self._draw_legend(axes)

        # Draw, save or/and show the figures
        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            if self.save:
                fig.savefig(self.output_path + self.extensions[idx], bbox_inches='tight', dpi=self.attr['dpi'])
            if self.show:
                fig.show()
            plt.close(fig)

    def _draw_front(self, ax, z, idx, number):

        # Bbox
        w = min(self.width-2, self.boxes[idx][2] - self.boxes[idx][0])
        h = min(self.height-2, (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale)
        x0 = self.boxes[idx][0]
        y0 = self.boxes[idx][1] * self.y_scale
        y1 = y0 + h
        rectangle = Rectangle((x0, y0),
                              width=w,
                              height=h,
                              fill=False,
                              color=self.attr[self.modes[idx]]['color'],
                              linewidth=self.attr[self.modes[idx]]['linewidth'])
        ax.add_patch(rectangle)
        z_str = str(z).split(sep='.')
        text = z_str[0] + '.' + z_str[1][0]
        bbox_config = {'facecolor': self.attr[self.modes[idx]]['color'], 'alpha': 0.4, 'linewidth': 0}

        x_t = x0 - 1.5
        y_t = y1 + self.attr['y_box_margin']
        if y_t < (self.height-10):
            ax.annotate(
                text,
                (x_t, y_t),
                fontsize=self.attr['fontsize_d'],
                weight='bold',
                xytext=(5.0, 5.0),
                textcoords='offset points',
                color='white',
                bbox=bbox_config,
            )
            if number['flag']:
                ax.text(x0 - 17,
                        y1 + 14,
                        chr(number['num']),
                        fontsize=self.attr['fontsize_num'],
                        color=self.attr[self.modes[idx]]['numcolor'],
                        weight='bold')

    def _draw_text_bird(self, axes, idx, num):
        """Plot the number in the bird eye view map"""

        std = self.stds_epi[idx] if self.stds_epi[idx] > 0 else self.stds_ale[idx]
        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])

        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)

        axes[1].text(self.xx_pred[idx] + delta_x + 0.2, self.zz_pred[idx] + delta_z + 0/2, chr(num),
                     fontsize=self.attr['fontsize_bv'],
                     color=self.attr[self.modes[idx]]['numcolor'])

    def _draw_uncertainty(self, axes, idx):

        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])
        dic_std = {'ale': self.stds_ale[idx], 'epi': self.stds_epi[idx]}
        dic_x, dic_y = {}, {}

        # Aleatoric and epistemic
        for key, std in dic_std.items():
            delta_x = std * math.cos(theta)
            delta_z = std * math.sin(theta)
            dic_x[key] = (self.xx_pred[idx] - delta_x, self.xx_pred[idx] + delta_x)
            dic_y[key] = (self.zz_pred[idx] - delta_z, self.zz_pred[idx] + delta_z)

        # MonoLoco
        if not self.auxs:
            axes[1].plot(dic_x['epi'],
                         dic_y['epi'],
                         color='coral',
                         linewidth=round(self.attr['linewidth']/2),
                         label="Epistemic Uncertainty")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.attr['linewidth'],
                         label="Aleatoric Uncertainty")

            axes[1].plot(self.xx_pred[idx],
                         self.zz_pred[idx],
                         color='cornflowerblue',
                         label="Prediction",
                         markersize=self.attr['markersize'],
                         marker='o')

            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=8,
                             marker='x')

        # MonStereo(stereo case)
        elif self.auxs[idx] > 0.5:
            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='r',
                         linewidth=self.attr['linewidth'],
                         label="Prediction (mono)")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.attr['linewidth'],
                         label="Prediction (stereo+mono)")

            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=self.attr['markersize'],
                             marker='x')

        # MonStereo (monocular case)
        else:
            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.attr['linewidth'],
                         label="Prediction (stereo+mono)")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='r',
                         linewidth=self.attr['linewidth'],
                         label="Prediction (mono)")
            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=self.attr['markersize'],
                             marker='x')

    def _draw_legend(self, axes):
        # Bird eye view legend
        if any(xx in self.output_types for xx in ['bird', 'multi']):
            handles, labels = axes[1].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            axes[1].legend(by_label.values(), by_label.keys(), loc='best', prop={'size': 15})

    def _set_axes(self, ax, axis):
        assert axis in (0, 1)

        if axis == 0:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        else:
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk
            corr = round(float(x_max / 3))
            ax.plot([0, x_max], [0, self.z_max], 'k--')
            ax.plot([0, -x_max], [0, self.z_max], 'k--')
            ax.set_xlim(-x_max + corr, x_max - corr)
            ax.set_ylim(0, self.z_max + 1)
            ax.set_xlabel("X [m]")
            plt.xticks(fontsize=self.attr['fontsize_ax'])
            plt.yticks(fontsize=self.attr['fontsize_ax'])
        return ax
