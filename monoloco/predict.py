# pylint: disable=too-many-statements, too-many-branches, undefined-loop-variable

"""
Adapted from https://github.com/openpifpaf/openpifpaf/blob/main/openpifpaf/predict.py,
which is: 'Copyright 2019-2021 by Sven Kreiss and contributors. All rights reserved.'
and licensed under GNU AGPLv3
"""

import os
import glob
import json
import copy
import logging
from collections import defaultdict


import torch
import PIL
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show, logger
try:
    import gdown
    DOWNLOAD = copy.copy(gdown.download)
except ImportError:
    DOWNLOAD = None
from .visuals.printer import Printer
from .network import Loco
from .network.process import factory_for_gt, preprocess_pifpaf
from .activity import show_social

LOG = logging.getLogger(__name__)

OPENPIFPAF_MODEL = 'https://drive.google.com/uc?id=1b408ockhh29OLAED8Tysd2yGZOo0N_SQ'
MONOLOCO_MODEL_KI = 'https://drive.google.com/uc?id=1krkB8J9JhgQp4xppmDu-YBRUxZvOs96r'
MONOLOCO_MODEL_NU = 'https://drive.google.com/uc?id=1BKZWJ1rmkg5AF9rmBEfxF1r8s8APwcyC'
MONSTEREO_MODEL = 'https://drive.google.com/uc?id=1xztN07dmp2e_nHI6Lcn103SAzt-Ntg49'


def get_torch_checkpoints_dir():
    if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
        # new in pytorch 1.6.0
        base_dir = torch.hub.get_dir()
    elif os.getenv('TORCH_HOME'):
        base_dir = os.getenv('TORCH_HOME')
    elif os.getenv('XDG_CACHE_HOME'):
        base_dir = os.path.join(os.getenv('XDG_CACHE_HOME'), 'torch')
    else:
        base_dir = os.path.expanduser(os.path.join('~', '.cache', 'torch'))
    return os.path.join(base_dir, 'checkpoints')


def download_checkpoints(args):
    torch_dir = get_torch_checkpoints_dir()
    if args.checkpoint is None:
        pifpaf_model = os.path.join(torch_dir, 'shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl')
    else:
        pifpaf_model = args.checkpoint
    dic_models = {'keypoints': pifpaf_model}
    if not os.path.exists(pifpaf_model):
        assert DOWNLOAD is not None, "pip install gdown to download pifpaf model, or pass it as --checkpoint"
        LOG.info('Downloading OpenPifPaf model in %s', torch_dir)
        DOWNLOAD(OPENPIFPAF_MODEL, pifpaf_model, quiet=False)

    if args.mode == 'keypoints':
        return dic_models
    if args.model is not None:
        assert os.path.exists(args.model), "Model path not found"
        dic_models[args.mode] = args.model
        return dic_models
    if args.mode == 'stereo':
        assert not args.social_distance, "Social distance not supported in stereo modality"
        path = MONSTEREO_MODEL
        name = 'monstereo-201202-1212.pkl'
    elif args.social_distance:
        path = MONOLOCO_MODEL_NU
        name = 'monoloco_pp-201207-1350.pkl'
    else:
        path = MONOLOCO_MODEL_KI
        name = 'monoloco_pp-201203-1424.pkl'

    model = os.path.join(torch_dir, name)
    dic_models[args.mode] = model
    if not os.path.exists(model):
        assert DOWNLOAD is not None, "pip install gdown to download monoloco model, or pass it as --model"
        LOG.info('Downloading model in %s', torch_dir)
        DOWNLOAD(path, model, quiet=False)
    return dic_models


def factory_from_args(args):

    # Data
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    if args.path_gt is None:
        args.show_all = True

    # Models
    dic_models = download_checkpoints(args)
    args.checkpoint = dic_models['keypoints']

    logger.configure(args, LOG)  # logger first

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    if args.mode == 'stereo':
        args.batch_size = 2
        args.images = sorted(args.images)
    else:
        args.batch_size = 1

    # Patch for stereo images with batch_size = 2
    if args.batch_size == 2 and not args.long_edge:
        args.long_edge = 1238
        LOG.info("Long-edge set to %i", args.long_edge)

    # Make default pifpaf argument
    args.force_complete_pose = True
    LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args, dic_models


def predict(args):

    cnt = 0
    assert args.mode in ('keypoints', 'mono', 'stereo')
    args, dic_models = factory_from_args(args)

    # Load Models
    if args.mode in ('mono', 'stereo'):
        net = Loco(
            model=dic_models[args.mode],
            mode=args.mode,
            device=args.device,
            n_dropout=args.n_dropout,
            p_dropout=args.dropout)

    # data
    processor, pifpaf_model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList(args.images, preprocess=preprocess)
    if args.mode == 'stereo':
        assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(pifpaf_model, image_tensors_batch, device=args.device)

        # unbatch (only for MonStereo)
        for idx, (pred, meta) in enumerate(zip(pred_batch, meta_batch)):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            pred = [ann.inverse_transform(meta) for ann in pred]

            # Load image and collect pifpaf results
            if idx == 0:
                with open(meta_batch[0]['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
                pifpaf_outs = {
                    'pred': pred,
                    'left': [ann.json_data() for ann in pred],
                    'image': cpu_image}

                # Set output image name
                if args.output_directory is None:
                    splits = os.path.split(meta['file_name'])
                    output_path = os.path.join(splits[0], 'out_' + splits[1])
                else:
                    file_name = os.path.basename(meta['file_name'])
                    output_path = os.path.join(args.output_directory, 'out_' + file_name)

                im_name = os.path.basename(meta['file_name'])
                print(f'{batch_i} image {im_name} saved as {output_path}')

            # Only for MonStereo
            else:
                pifpaf_outs['right'] = [ann.json_data() for ann in pred]

        # 3D Predictions
        if args.mode != 'keypoints':
            im_size = (cpu_image.size[0], cpu_image.size[1])  # Original
            kk, dic_gt = factory_for_gt(im_size, focal_length=args.focal, name=im_name, path_gt=args.path_gt)

            # Preprocess pifpaf outputs and run monoloco
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)

            if args.mode == 'mono':
                LOG.info("Prediction with MonoLoco++")
                dic_out = net.forward(keypoints, kk)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)
                if args.social_distance:
                    dic_out = net.social_distance(dic_out, args)

            else:
                LOG.info("Prediction with MonStereo")
                _, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                dic_out = net.forward(keypoints, kk, keypoints_r=keypoints_r)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)

        else:
            dic_out = defaultdict(list)
            kk = None

        # Outputs
        factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=kk)
        print(f'Image {cnt}\n' + '-' * 120)
        cnt += 1


def factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=None):
    """Output json files or images according to the choice"""

    # Verify conflicting options
    if any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
        assert args.mode != 'keypoints', "for keypoints please use pifpaf original arguments"
    else:
        assert 'json' in args.output_types or args.mode == 'keypoints', \
            "No output saved, please select one among front, bird, multi, json, or pifpaf arguments"
    if args.social_distance:
        assert args.mode == 'mono', "Social distancing only works with monocular network"

    if args.mode == 'keypoints':
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(pifpaf_outs['image'], output_path) as ax:
            annotation_painter.annotations(ax, pifpaf_outs['pred'])
        return

    if any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
        LOG.info(output_path)
        if args.social_distance:
            show_social(args, pifpaf_outs['image'], output_path, pifpaf_outs['left'], dic_out)
        else:
            printer = Printer(pifpaf_outs['image'], output_path, kk, args)
            figures, axes = printer.factory_axes(dic_out)
            printer.draw(figures, axes, pifpaf_outs['image'])

    if 'json' in args.output_types:
        with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
            json.dump(dic_out, ff)
