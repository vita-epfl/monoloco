
import json
import os
from collections import defaultdict

from visuals.printer import Printer
from openpifpaf import show


def factory_for_gt(im_size, name=None, path_gt=None):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        print('-' * 120 + "\nMonoloco: Ground-truth file opened\n")
    except FileNotFoundError:
        print('-' * 120 + "\nMonoloco: ground-truth file not found\n")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        print("Monoloco: matched ground-truth file!\n" + '-' * 120)
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration

        print("Ground-truth annotations for the image not found\n" 
              "Using a standard calibration matrix...\n" + '-' * 120)

    return kk, dic_gt


def factory_outputs(args, images_outputs, output_path, pifpaf_outputs, monoloco_outputs=None, kk=None):
    """Output json files or images according to the choice"""

    # Save json file
    if 'pifpaf' in args.networks:
        keypoint_sets, scores, pifpaf_out = pifpaf_outputs[:]

        # Visualizer
        keypoint_painter = show.KeypointPainter(show_box=True)
        skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                                markersize=1, linewidth=4)

        if 'json' in args.output_types and keypoint_sets.size > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        if 'keypoints' in args.output_types:
            with show.image_canvas(images_outputs[0],
                                   output_path + '.keypoints.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets)

        if 'skeleton' in args.output_types:
            with show.image_canvas(images_outputs[0],
                                   output_path + '.skeleton.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    if 'monoloco' in args.networks:

        dic_out = monoloco_post_process(monoloco_outputs)

        if any((xx in args.output_types for xx in ['front', 'bird', 'combined'])):

            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            printer = Printer(images_outputs[1], output_path, dic_out, kk, output_types=args.output_types,
                              show=args.show, z_max=args.z_max, epistemic=epistemic)
            printer.print()

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(monoloco_outputs, ff)


def monoloco_post_process(monoloco_outputs):

    """Post process monoloco to output final dictionary with all information for visualizations"""
    # Create output files
    dic_out = defaultdict(list)
    if dic_gt:
        boxes_gt, dds_gt = dic_gt['boxes'], dic_gt['dds']

    for idx, box in enumerate(uv_boxes):
        dd_pred = float(outputs[idx][0])
        ale = float(outputs[idx][1])
        var_y = float(varss[idx])

        # Find the corresponding ground truth if available
        if dic_gt:
            idx_max, iou_max = get_idx_max(box, boxes_gt)
            if iou_max > self.IOU_MIN:
                dd_real = dds_gt[idx_max]
                boxes_gt.pop(idx_max)
                dds_gt.pop(idx_max)
            # In case of no matching
            else:
                dd_real = 0
        # In case of no ground truth
        else:
            dd_real = dd_pred

        uv_center = uv_centers[idx]
        xyz_real = get_depth(uv_center, kk, dd_real)
        xyz_pred = get_depth(uv_center, kk, dd_pred)
        dic_out['boxes'].append(box)
        dic_out['dds_real'].append(dd_real)
        dic_out['dds_pred'].append(dd_pred)
        dic_out['stds_ale'].append(ale)
        dic_out['stds_epi'].append(var_y)
        dic_out['xyz_real'].append(xyz_real)
        dic_out['xyz_pred'].append(xyz_pred)
        dic_out['xy_kps'].append(xy_kps[idx])
        dic_out['uv_kps'].append(uv_kps[idx])
        dic_out['uv_centers'].append(uv_center)
        dic_out['uv_shoulders'].append(uv_shoulders[idx])

    return dic_out



