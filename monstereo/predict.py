# pylint: disable=too-many-statements, too-many-branches, undefined-loop-variable

"""
Adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/predict.py
"""

import os
import glob
import json
import logging
from collections import defaultdict


import torch
import PIL
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show

from .visuals.printer import Printer
from .network import Loco
from .network.process import factory_for_gt, preprocess_pifpaf
from .activity import show_social

LOG = logging.getLogger(__name__)

OPENPIFPAF_PATH = 'data/models/shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl'  # Default model


def factory_from_args(args):

    # Data
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # Model
    if not args.checkpoint:
        if os.path.exists(OPENPIFPAF_PATH):
            args.checkpoint = OPENPIFPAF_PATH
        else:
            LOG.info("Checkpoint for OpenPifPaf not specified and default model not found in 'data/models'. "
                  "Using a ShuffleNet backbone")
            args.checkpoint = 'shufflenetv2k30'

    # Devices
    args.device = torch.device('cpu')
    args.disable_cuda = False
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    args.loader_workers = 8

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    if args.net == 'monstereo':
        args.batch_size = 2
    else:
        args.batch_size = 1

    # Make default pifpaf argument
    args.force_complete_pose = True
    LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

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

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch (only for MonStereo)
        for idx, (pred, meta) in enumerate(zip(pred_batch, meta_batch)):
            print('batch %d: %s', batch_i, meta['file_name'])
            pred = preprocess.annotations_inverse(pred, meta)

            if args.output_directory is None:
                splits = os.path.split(meta['file_name'])
                output_path = os.path.join(splits[0], 'out_' + splits[1])
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, 'out_' + file_name)
            print('image', batch_i, meta['file_name'], output_path)

            if idx == 0:
                with open(meta_batch[0]['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
                pifpaf_outs = {
                    'pred': pred,
                    'left': [ann.json_data() for ann in pred],
                    'image': cpu_image}
            else:
                pifpaf_outs['right'] = [ann.json_data() for ann in pred]

        # 3D Predictions
        if args.net in ('monoloco_pp', 'monstereo'):

            im_name = os.path.basename(meta['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])  # Original
            kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)

            # Preprocess pifpaf outputs and run monoloco
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)

            if args.net == 'monoloco_pp':
                LOG.info("Prediction with MonoLoco++")
                dic_out = net.forward(keypoints, kk)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)
                if args.social_distance:
                    dic_out = net.social_distance(dic_out, args)

            else:
                LOG.info("Prediction with MonStereo")
                boxes_r, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                dic_out = net.forward(keypoints, kk, keypoints_r=keypoints_r)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)

        else:
            dic_out = defaultdict(list)
            kk = None

        # Outputs
        factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=kk)
        LOG.info('Image {}\n'.format(cnt) + '-' * 120)
        cnt += 1


def factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=None):
    """Output json files or images according to the choice"""

    # Verify conflicting options
    if any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
        assert args.net != 'pifpaf', "please use pifpaf original arguments"
        if args.social_distance:
            assert args.net == 'monoloco_pp', "Social distancing only works with MonoLoco++ network"

    if args.net == 'pifpaf':
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(pifpaf_outs['image'], output_path) as ax:
            annotation_painter.annotations(ax, pifpaf_outs['pred'])

    elif any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
        LOG.info(output_path)
        if args.social_distance:
            show_social(args, pifpaf_outs['image'], output_path, pifpaf_outs['left'], dic_out)
        else:
            printer = Printer(pifpaf_outs['image'], output_path, kk, args)
            figures, axes = printer.factory_axes(dic_out)
            printer.draw(figures, axes, pifpaf_outs['image'])

    elif 'json' in args.output_types:
        with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
            json.dump(dic_out, ff)

    else:
        LOG.info("No output saved, please select one among front, bird, multi, or pifpaf options")
