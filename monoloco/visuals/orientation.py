
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

    def __init__(self, angles, colors, uv_shoulders, y_scale=1):

        self.angles = angles
        self.y_scale = y_scale
        self.uv_shoulders = uv_shoulders
        self.colors = colors

    def draw(self, ax, idx, center, mode):
        """
        Draw orientation for both the frontal and bird eye view figures
        Depending whether the image is front or bird mode, the center is in uv coordinates or xz ones
        """
        assert mode in ('front', 'bird')
        length = self.att[mode]['length']
        fill = self.att[mode]['fill']
        alpha = self.att[mode]['alpha']
        zorder_circle = self.att[mode]['zorder_circle']
        zorder_arrow = self.att[mode]['zorder_arrow']
        linewidth = self.att[mode]['linewidth']
        color = self.colors[mode][idx]
        edgecolor = self.att[mode]['edgecolor'] if mode == 'front' else color
        theta = self.angles[idx]

        if mode == 'front':
            center[1] *= self.y_scale
            radius = (center[1] - self.uv_shoulders[idx][1] * self.y_scale) / 1.1
            head_width = max(10, radius / 1.5)
            x_arr = center[0] + (length + radius) * math.cos(theta)
            z_arr = length + center[1] + (length + radius) * math.sin(-theta)
            delta_x = math.cos(theta)
            delta_z = math.sin(theta)
        else:
            radius = self.att[mode]['radius']
            head_width = self.att[mode]['head_width']
            x_arr = center[0]
            z_arr = center[1]
            delta_x = length * math.cos(theta)
            delta_z = length * math.sin(-theta)
            length += 0.007 * center[1]  # increase arrow length

        circle = Circle(center, radius=radius, color=color, fill=fill, alpha=alpha, zorder=zorder_circle)
        arrow = FancyArrow(x_arr, z_arr, delta_x, delta_z, head_width=head_width, edgecolor=edgecolor,
                           facecolor=color, linewidth=linewidth, zorder=zorder_arrow, label='Orientation')
        ax.add_patch(circle)
        ax.add_patch(arrow)
        if mode == 'bird':
            ax.legend(handles=[arrow])
