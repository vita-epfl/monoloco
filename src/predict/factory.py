
import json
import os
from collections import defaultdict
from openpifpaf import show
from visuals.printer import Printer
from utils.misc import get_iou_matches
from utils.camera import get_depth, get_keypoints_torch


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


def monoloco_post_process(monoloco_outputs, iou_min=0.25):
    """Post process monoloco to output final dictionary with all information for visualizations"""

    dic_out = defaultdict(list)
    outputs, varss, boxes, keypoints, kk, dic_gt = monoloco_outputs[:]
    if dic_gt:
        boxes_gt, dds_gt = dic_gt['boxes'], dic_gt['dds']
        matches = get_iou_matches(boxes, boxes_gt, thresh=iou_min)
    else:
        matches = [(idx, idx_gt) for idx, idx_gt in range(len(boxes))]  # Replicate boxes

    uv_shoulders = get_keypoints_torch(keypoints, mode='shoulder')
    uv_centers = get_keypoints_torch(keypoints, mode='center')

    # Match with ground truth if available
    for idx, idx_gt in matches:
        dd_pred = float(outputs[idx][0])
        ale = float(outputs[idx][1])
        var_y = float(varss[idx])
        dd_real = dds_gt[idx_gt] if dic_gt else dd_pred

        kps = keypoints[idx]
        box = boxes[idx]
        uu_s, vv_s = uv_shoulders.tolist()[idx][0:2]
        uu_c, vv_c = uv_centers.tolist()[idx][0:2]
        uv_shoulder = [round(uu_s), round(vv_s)]
        uv_center = [round(uu_c), round(vv_c)]
        xyz_real = get_depth(uv_center, kk, dd_real)
        xyz_pred = get_depth(uv_center, kk, dd_pred)
        dic_out['boxes'].append(box)
        dic_out['dds_real'].append(dd_real)
        dic_out['dds_pred'].append(dd_pred)
        dic_out['stds_ale'].append(ale)
        dic_out['stds_epi'].append(var_y)
        dic_out['xyz_real'].append(xyz_real)
        dic_out['xyz_pred'].append(xyz_pred)
        dic_out['uv_kps'].append(kps)
        dic_out['uv_centers'].append(uv_center)
        dic_out['uv_shoulders'].append(uv_shoulder)

    return dic_out



