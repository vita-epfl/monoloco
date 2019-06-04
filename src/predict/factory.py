
import json
import os
from visuals.printer import Printer
from openpifpaf import show

from PIL import Image


def factory_for_gt(image, name, path_gt):
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
        x_factor = image.size[0] / 1600
        y_factor = image.size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2
        if image.size[0] / image.size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration

        print("Ground-truth annotations for the image not found\n" 
              "Using a standard calibration matrix...\n" + '-' * 120)

    return kk, dic_gt


def factory_outputs(image, output_path, pifpaf_outputs, monoloco_outputs, kk, args):
    """Output json files or images according to the choice"""

    # Save json file
    if 'pifpaf' in args.networks:

        keypoint_sets, pifpaf_out, scores = pifpaf_outputs[:]

        # Visualizer
        keypoint_painter = show.KeypointPainter(show_box=True)
        skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                                markersize=1, linewidth=4)

        if 'json' in args.output_types and keypoint_sets.size > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        if 'keypoints' in args.output_types:
            with show.image_canvas(image,
                                   output_path + '.keypoints.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets)

        if 'skeleton' in args.output_types:
            with show.image_canvas(image,
                                   output_path + '.skeleton.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    if 'monoloco' in args.networks:
        if any((xx in args.output_types for xx in ['front', 'bird', 'combined'])):

            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            printer = Printer(image, output_path, monoloco_outputs, kk, output_types=args.output_types,
                              show=args.show, z_max=args.z_max, epistemic=epistemic)
            printer.print()

        if 'json' in args.output_types:
            with open(os.path.join(args.output_path + '.monoloco.json'), 'w') as ff:
                json.dump(monoloco_outputs, ff)


