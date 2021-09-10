
import math

from matplotlib.patches import Circle, FancyArrow


class DrawOrientation:

    att = dict(
        front={
            'length': 5,
            'fill': False,
            'alpha': 0.6,
            'zorder_circle': 0.5,
            'zorder_arrow': 5,
            'linewidth': 1.5,
            'edgecolor': 'k',
        },
        bird={
            'length': 1.3,
            'linewidth': 2.3,
            'head_width': 0.3,
            'fill': True,
            'alpha': 1,
            'zorder_circle': 2,
            'zorder_arrow': 1,
            'radius': 0.2,
        })

    def __init__(self, angles, colors, mode, shoulders=None, y_scale=1):

        assert mode in ('front', 'bird')
        self.mode = mode
        self.angles = angles
        self.y_scale = y_scale
        self.uv_shoulders = shoulders
        self.colors = colors
        self.length = self.att[mode]['length']
        self.fill = self.att[mode]['fill']
        self.alpha = self.att[mode]['alpha']
        self.zorder_circle = self.att[mode]['zorder_circle']
        self.zorder_arrow = self.att[mode]['zorder_arrow']
        self.linewidth = self.att[mode]['linewidth']

    def draw(self, ax, idx, center):
        """
        Draw orientation for both the frontal and bird eye view figures
        Depending whether the image is front or bird mode, the center is in uv coordinates or xz ones
        """

        theta = self.angles[idx]
        color = self.colors[idx]
        if self.mode == 'front':
            assert self.uv_shoulders is not None, "required uv shoulders for front figure"
            center[1] *= self.y_scale
            radius = (abs(center[1] - self.uv_shoulders[idx][1] * self.y_scale)) / 1.4
            head_width = max(10, radius / 1.5)
            x_arr = center[0] + (self.length + radius) * math.cos(theta)
            z_arr = self.length + center[1] + (self.length + radius) * math.sin(theta)
            delta_x = math.cos(theta)
            delta_z = math.sin(theta)
            edgecolor = self.att['front']['edgecolor']
        else:
            radius = self.att['bird']['radius']
            head_width = self.att['bird']['head_width']
            x_arr = center[0]
            z_arr = center[1]
            delta_x = self.length * math.cos(theta)
            delta_z = self.length * math.sin(-theta)
            self.length += 0.007 * center[1]  # increase arrow length
            edgecolor = color

        circle = Circle(center, radius=radius, color=color, fill=self.fill, alpha=self.alpha, zorder=self.zorder_circle)
        arrow = FancyArrow(x_arr, z_arr, delta_x, delta_z, head_width=head_width, edgecolor=edgecolor,
                           facecolor=color, linewidth=self.linewidth, zorder=self.zorder_arrow, label='Orientation')
        ax.add_patch(circle)
        ax.add_patch(arrow)
        if self.mode == 'bird':
            ax.legend(handles=[arrow])
