
# pylint: disable=too-many-statements, too-many-branches, undefined-loop-variable

import os
import glob
import json
import logging
from collections import defaultdict


import torch
import PIL
from PIL import Image
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf.predict import processor_factory, preprocess_factory

from .visuals.printer import Printer
from .visuals.pifpaf_show import KeypointPainter, image_canvas
from .network import PifPaf, ImageList, Loco
from .network.process import factory_for_gt, preprocess_pifpaf

LOG = logging.getLogger(__name__)


def factory_from_args(args):

    # Merge the model_pifpaf argument
    if not args.checkpoint:
        args.checkpoint = 'shufflenetv2k30'  # Default model
    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.disable_cuda = False
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    # Add num_workers
    args.loader_workers = 8

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    # TODO
    args.long_edge = None
    if args.net == 'monstereo':
        args.batch_size = 2
    else:
        args.batch_size = 1

    return args


def predict(args):

    cnt = 0
    args = factory_from_args(args)

    # Load Models
    assert args.net in ('monoloco_pp', 'monstereo', 'pifpaf')

    if args.net in ('monoloco_pp', 'monstereo'):
        net = Loco(model=args.model, net=args.net, device=args.device, n_dropout=args.n_dropout, p_dropout=args.dropout)

    # data

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList(args.images, preprocess=preprocess)
    if args.net == 'monstereo':
        assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    annotation_painter = openpifpaf.show.AnnotationPainter()

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            pred = preprocess.annotations_inverse(pred, meta)

            if args.output_directory is None:
                splits = os.path.split(meta['file_name'])
                output_path = os.path.join(splits[0], 'out_' + splits[1])
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, 'out_' + file_name)
            print('image', batch_i, meta['file_name'], output_path)
            pifpaf_out = [ann.json_data() for ann in pred]

            if batch_i == 0:
                pifpaf_outputs = pred  # to only print left image for stereo
                pifpaf_outs = {'left': pifpaf_out}
                with open(meta_batch[0]['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
            else:
                pifpaf_outs['right'] = pifpaf_out

        # Load the original image
        if args.net in ('monoloco_pp', 'monstereo'):

            im_name = os.path.basename(meta['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])  # Original
            kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)

            # Preprocess pifpaf outputs and run monoloco
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)

            if args.net == 'monoloco_pp':
                print("Prediction with MonoLoco++")
                dic_out = net.forward(keypoints, kk)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)

            else:
                print("Prediction with MonStereo")
                boxes_r, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                dic_out = net.forward(keypoints, kk, keypoints_r=keypoints_r)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)

        else:
            dic_out = defaultdict(list)
            kk = None

        # TODO Clean
        factory_outputs(args, annotation_painter, cpu_image, output_path, pifpaf_outputs, pifpaf_out,
                        dic_out=dic_out, kk=kk)
        print('Image {}\n'.format(cnt) + '-' * 120)
        cnt += 1


def factory_outputs(args, annotation_painter, cpu_image, output_path, pred, pifpaf_out, dic_out=None, kk=None):
    """Output json files or images according to the choice"""

    # Save json file
    if args.net == 'pifpaf':
        with openpifpaf.show.image_canvas(cpu_image, output_path) as ax:
            annotation_painter.annotations(ax, pred)

        # Visualizer
        keypoint_painter = KeypointPainter(show_box=False)
        skeleton_painter = KeypointPainter(show_box=False, color_connections=True, markersize=1, linewidth=4)

        if 'json' in args.output_types and len(pred) > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        # if 'keypoints' in args.output_types:
        #     with image_canvas(images_outputs[0],
        #                       output_path + '.keypoints.png',
        #                       show=args.show,
        #                       fig_width=args.figure_width,
        #                       dpi_factor=args.dpi_factor) as ax:
        #         keypoint_painter.keypoints(ax, keypoint_sets)
        #
        # if 'skeleton' in args.output_types:
        #     with image_canvas(images_outputs[0],
        #                       output_path + '.skeleton.png',
        #                       show=args.show,
        #                       fig_width=args.figure_width,
        #                       dpi_factor=args.dpi_factor) as ax:
        #         skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    else:
        if any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
            print(output_path)
            if dic_out['boxes']:  # Only print in case of detections
                printer = Printer(cpu_image, output_path, kk, args)
                figures, axes = printer.factory_axes()
                printer.draw(figures, axes, dic_out, cpu_image)

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)
