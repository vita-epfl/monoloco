
# pylint: disable=too-many-statements, too-many-branches, undefined-loop-variable

import os
import json
from collections import defaultdict


import torch
from PIL import Image

from .visuals.printer import Printer
from .visuals.pifpaf_show import KeypointPainter, image_canvas
from .network import PifPaf, ImageList, Loco
from .network.process import factory_for_gt, preprocess_pifpaf


def predict(args):

    cnt = 0

    # Load Models
    pifpaf = PifPaf(args)
    assert args.mode in ('mono', 'stereo', 'pifpaf')

    if 'mono' in args.mode:
        monoloco = Loco(model=args.model, net='monoloco_pp',
                        device=args.device, n_dropout=args.n_dropout, p_dropout=args.dropout)

    if 'stereo' in args.mode:
        monstereo = Loco(model=args.model, net='monstereo',
                         device=args.device, n_dropout=args.n_dropout, p_dropout=args.dropout)

    # data
    data = ImageList(args.images, scale=args.scale)
    if args.mode == 'stereo':
        assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"
        bs = 2
    else:
        bs = 1
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=bs, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    for idx, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = pifpaf.fields(processed_images)

        # unbatch stereo pair
        for ii, (image_path, image, processed_image_cpu, fields) in enumerate(zip(
                image_paths, images, processed_images_cpu, fields_batch)):

            if args.output_directory is None:
                output_path = image_paths[0]
            else:
                file_name = os.path.basename(image_paths[0])
                output_path = os.path.join(args.output_directory, file_name)
            print('image', idx, image_path, output_path)
            keypoint_sets, scores, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

            if ii == 0:
                pifpaf_outputs = [keypoint_sets, scores, pifpaf_out]  # keypoints_sets and scores for pifpaf printing
                images_outputs = [image]  # List of 1 or 2 elements with pifpaf tensor and monoloco original image
                pifpaf_outs = {'left': pifpaf_out}
                image_path_l = image_path
            else:
                pifpaf_outs['right'] = pifpaf_out

        if args.mode in ('stereo', 'mono'):
            # Extract calibration matrix and ground truth file if present
            with open(image_path_l, 'rb') as f:
                pil_image = Image.open(f).convert('RGB')
                images_outputs.append(pil_image)

            im_name = os.path.basename(image_path_l)
            im_size = (float(image.size()[1] / args.scale), float(image.size()[0] / args.scale))  # Original
            kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)

            # Preprocess pifpaf outputs and run monoloco
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)

            if args.mode == 'mono':
                print("Prediction with MonoLoco++")
                dic_out = monoloco.forward(keypoints, kk)
                dic_out = monoloco.post_process(dic_out, boxes, keypoints, kk, dic_gt)

            else:
                print("Prediction with MonStereo")
                boxes_r, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                dic_out = monstereo.forward(keypoints, kk, keypoints_r=keypoints_r)
                dic_out = monstereo.post_process(dic_out, boxes, keypoints, kk, dic_gt)

        else:
            dic_out = defaultdict(list)
            kk = None

        factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=dic_out, kk=kk)
        print('Image {}\n'.format(cnt) + '-' * 120)
        cnt += 1


def factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=None, kk=None):
    """Output json files or images according to the choice"""

    # Save json file
    if args.mode == 'pifpaf':
        keypoint_sets, scores, pifpaf_out = pifpaf_outputs[:]

        # Visualizer
        keypoint_painter = KeypointPainter(show_box=False)
        skeleton_painter = KeypointPainter(show_box=False, color_connections=True, markersize=1, linewidth=4)

        if 'json' in args.output_types and keypoint_sets.size > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        if 'keypoints' in args.output_types:
            with image_canvas(images_outputs[0],
                              output_path + '.keypoints.png',
                              show=args.show,
                              fig_width=args.figure_width,
                              dpi_factor=args.dpi_factor) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets)

        if 'skeleton' in args.output_types:
            with image_canvas(images_outputs[0],
                              output_path + '.skeleton.png',
                              show=args.show,
                              fig_width=args.figure_width,
                              dpi_factor=args.dpi_factor) as ax:
                skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    else:
        if any((xx in args.output_types for xx in ['front', 'bird', 'combined'])):
            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            if dic_out['boxes']:  # Only print in case of detections
                printer = Printer(images_outputs[1], output_path, kk, output_types=args.output_types
                                  , z_max=args.z_max, epistemic=epistemic)
                figures, axes = printer.factory_axes()
                printer.draw(figures, axes, dic_out, images_outputs[1], show_all=args.show_all, draw_box=args.draw_box,
                             save=True, show=args.show)

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)
