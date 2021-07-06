
# pylint: disable=too-many-statements

import math
import copy
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt

from .network.process import laplace_sampling
from .visuals.pifpaf_show import KeypointPainter, image_canvas, get_pifpaf_outputs
from .visuals.printer import draw_orientation, social_distance_colors


def social_interactions(idx, centers, angles, dds, stds=None, social_distance=False,
                        n_samples=100, threshold_prob=0.25, threshold_dist=2, radii=(0.3, 0.5)):
    """
    return flag of alert if social distancing is violated
    """

    # A) Check whether people are close together
    xx = centers[idx][0]
    zz = centers[idx][1]
    distances = [math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2)
                 for i, _ in enumerate(centers)]
    sorted_idxs = np.argsort(distances)
    indices = [idx_t for idx_t in sorted_idxs[1:]
               if distances[idx_t] <= threshold_dist]

    # B) Check whether people are looking inwards and whether there are no intrusions
    # Deterministic
    if n_samples < 2:
        for idx_t in indices:
            if check_f_formations(idx, idx_t, centers, angles,
                                  radii=radii,  # Binary value
                                  social_distance=social_distance):
                return True

    # Probabilistic
    else:
        # Samples distance
        dds = torch.tensor(dds).view(-1, 1)
        stds = torch.tensor(stds).view(-1, 1)
        # stds_te = get_task_error(dds)  # similar results to MonoLoco but lower true positive
        laplace_d = torch.cat((dds, stds), dim=1)
        samples_d = laplace_sampling(laplace_d, n_samples=n_samples)

        # Iterate over close people
        for idx_t in indices:
            f_forms = []
            for s_d in range(n_samples):
                new_centers = copy.deepcopy(centers)
                for el in (idx, idx_t):
                    delta_d = dds[el] - float(samples_d[s_d, el])
                    theta = math.atan2(new_centers[el][1], new_centers[el][0])
                    delta_x = delta_d * math.cos(theta)
                    delta_z = delta_d * math.sin(theta)
                    new_centers[el][0] += delta_x
                    new_centers[el][1] += delta_z
                f_forms.append(check_f_formations(idx, idx_t, new_centers, angles,
                                                  radii=radii,
                                                  social_distance=social_distance))
            if (sum(f_forms) / n_samples) >= threshold_prob:
                return True
    return False


def is_raising_hand(kp):
    """
    Returns flag of alert if someone raises their hand
    """
    x=0
    y=1

    nose = 0
    l_ear = 3
    l_shoulder = 5
    l_elbow = 7
    l_hand = 9
    r_ear = 4
    r_shoulder = 6
    r_elbow = 8
    r_hand = 10

    head_width = kp[x][l_ear]- kp[x][r_ear]
    head_top = (kp[y][nose] - head_width)

    l_forearm = [kp[x][l_hand] - kp[x][l_elbow], kp[y][l_hand] - kp[y][l_elbow]]
    l_arm = [kp[x][l_shoulder] - kp[x][l_elbow], kp[y][l_shoulder] - kp[y][l_elbow]]

    r_forearm = [kp[x][r_hand] - kp[x][r_elbow], kp[y][r_hand] - kp[y][r_elbow]]
    r_arm = [kp[x][r_shoulder] - kp[x][r_elbow], kp[y][r_shoulder] - kp[y][r_elbow]]

    l_angle = (90/np.pi) * np.arccos(np.dot(l_forearm/np.linalg.norm(l_forearm), l_arm/np.linalg.norm(l_arm)))
    r_angle = (90/np.pi) * np.arccos(np.dot(r_forearm/np.linalg.norm(r_forearm), r_arm/np.linalg.norm(r_arm)))

    is_l_up = kp[y][l_hand] < kp[y][l_shoulder]
    is_r_up = kp[y][r_hand] < kp[y][r_shoulder]

    l_too_close = kp[x][l_hand] <= kp[x][l_shoulder] and kp[y][l_hand]>=head_top
    r_too_close = kp[x][r_hand] >= kp[x][r_shoulder] and kp[y][r_hand]>=head_top

    is_left_risen = is_l_up and l_angle >= 30 and not l_too_close
    is_right_risen = is_r_up and r_angle >= 30 and not r_too_close

    if is_left_risen and is_right_risen:
        return 'both'

    if is_left_risen:
        return 'left'

    if is_right_risen:
        return 'right'

    return None


def check_f_formations(idx, idx_t, centers, angles, radii, social_distance=False):
    """
    Check F-formations for people close together (this function do not expect far away people):
    1) Empty space of a certain radius (no other people or themselves inside)
    2) People looking inward
    """

    # Extract centers and angles
    other_centers = np.array(
        [cent for l, cent in enumerate(centers) if l not in (idx, idx_t)])
    theta0 = angles[idx]
    theta1 = angles[idx_t]

    # Find the center of o-space as average of two candidates (based on their orientation)
    for radius in radii:
        x_0 = np.array([float(centers[idx][0]), float(centers[idx][1])])
        x_1 = np.array([float(centers[idx_t][0]), float(centers[idx_t][1])])

        mu_0 = np.array([
            float(centers[idx][0]) + radius * math.cos(theta0),
            float(centers[idx][1]) - radius * math.sin(theta0)])
        mu_1 = np.array([
            float(centers[idx_t][0]) + radius * math.cos(theta1),
            float(centers[idx_t][1]) - radius * math.sin(theta1)])
        o_c = (mu_0 + mu_1) / 2

        # 1) Verify they are looking inwards.
        # The distance between mus and the center should be less wrt the original position and the center
        d_new = np.linalg.norm(
            mu_0 - mu_1) / 2 if social_distance else np.linalg.norm(mu_0 - mu_1)
        d_0 = np.linalg.norm(x_0 - o_c)
        d_1 = np.linalg.norm(x_1 - o_c)

        # 2) Verify no intrusion for third parties
        if other_centers.size:
            other_distances = np.linalg.norm(
                other_centers - o_c.reshape(1, -1), axis=1)
        else:
            # Condition verified if no other people
            other_distances = 100 * np.ones((1, 1))

        # Binary Classification
        # if np.min(other_distances) > radius:  # Ablation without orientation
        if d_new <= min(d_0, d_1) and np.min(other_distances) > radius:
            return True
    return False


def show_activities(args, image_t, output_path, annotations, dic_out):
    """Output frontal image with poses or combined with bird eye view"""

    assert 'front' in args.output_types or 'bird' in args.output_types, "outputs allowed: front and/or bird"

    colors = ['deepskyblue' for _ in dic_out['uv_heads']]
    if 'social_distance' in args.activities:
        colors = social_distance_colors(colors, dic_out)

    angles = dic_out['angles']
    stds = dic_out['stds_ale']
    xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

    # Draw keypoints and orientation
    if 'front' in args.output_types:
        keypoint_sets, _ = get_pifpaf_outputs(annotations)
        uv_centers = dic_out['uv_heads']
        sizes = [abs(dic_out['uv_heads'][idx][1] - uv_s[1]) / 1.5 for idx, uv_s in
                 enumerate(dic_out['uv_shoulders'])]
        keypoint_painter = KeypointPainter(show_box=False)

        with image_canvas(image_t,
                          output_path + '.front.png',
                          show=args.show,
                          fig_width=10,
                          dpi_factor=1.0) as ax:
            keypoint_painter.keypoints(
                ax, keypoint_sets, activities=args.activities, dic_out=dic_out,
                size=image_t.size, colors=colors)
            draw_orientation(ax, uv_centers, sizes,
                             angles, colors, mode='front')

    if 'bird' in args.output_types:
        z_max = min(args.z_max, 4 + max([el[1] for el in xz_centers]))
        with bird_canvas(output_path, z_max) as ax1:
            draw_orientation(ax1, xz_centers, [], angles, colors, mode='bird')
            draw_uncertainty(ax1, xz_centers, stds)


@contextmanager
def bird_canvas(output_path, z_max):
    fig, ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    output_path = output_path + '.bird.png'
    x_max = z_max / 1.5
    ax.plot([0, x_max], [0, z_max], 'k--')
    ax.plot([0, -x_max], [0, z_max], 'k--')
    ax.set_ylim(0, z_max + 1)
    yield ax
    fig.savefig(output_path)
    plt.close(fig)
    print('Bird-eye-view image saved')


def draw_uncertainty(ax, centers, stds):
    for idx, std in enumerate(stds):
        std = stds[idx]
        theta = math.atan2(centers[idx][1], centers[idx][0])
        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)
        x = (centers[idx][0] - delta_x, centers[idx][0] + delta_x)
        z = (centers[idx][1] - delta_z, centers[idx][1] + delta_z)
        ax.plot(x, z, color='g', linewidth=2.5)
