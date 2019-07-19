
import json
import os
from openpifpaf import show
from ..visuals.printer import Printer


def factory_for_gt(im_size, name=None, path_gt=None):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        print('-' * 120 + "\nGround-truth file opened")
    except (FileNotFoundError, TypeError):
        print('-' * 120 + "\nGround-truth file not found")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        print("Matched ground-truth file!")
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2   # TODO remove and check it
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration

        print("Using a standard calibration matrix...")

    return kk, dic_gt


def factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=None, kk=None):
    """Output json files or images according to the choice"""

    # Save json file
    if 'pifpaf' in args.networks:
        keypoint_sets, scores, pifpaf_out = pifpaf_outputs[:]

        # Visualizer
        keypoint_painter = show.KeypointPainter(show_box=False)
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
        if any((xx in args.output_types for xx in ['front', 'bird', 'combined'])):
            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            if dic_out['boxes']:  # Only print in case of detections
                printer = Printer(images_outputs[1], output_path, kk, output_types=args.output_types
                                  , z_max=args.z_max, epistemic=epistemic)
                figures, axes = printer.factory_axes()
                printer.draw(figures, axes, dic_out, images_outputs[1], draw_box=args.draw_box,
                             save=True, show=args.show)

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)
