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
import time
from collections import defaultdict

import numpy as np
import torch
import PIL
import openpifpaf
from openpifpaf import datasets
from openpifpaf import decoder, network, visualizer, show, logger, Predictor
from openpifpaf.predict import out_name

try:
    import gdown
    DOWNLOAD = copy.copy(gdown.download)
except ImportError:
    DOWNLOAD = None
from .visuals.printer import Printer
from .network import Loco, factory_for_gt, load_calibration, preprocess_pifpaf
from .activity import show_activities

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
    os.makedirs(torch_dir, exist_ok=True)
    if args.checkpoint is None:
        os.makedirs(torch_dir, exist_ok=True)
        pifpaf_model = os.path.join(torch_dir, 'shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl')
    else:
        pifpaf_model = args.checkpoint
    dic_models = {'keypoints': pifpaf_model}
    if not os.path.exists(pifpaf_model):
        assert DOWNLOAD is not None, \
            "pip install gdown to download a pifpaf model, or pass the model path as --checkpoint"
        LOG.info('Downloading OpenPifPaf model in %s', torch_dir)
        DOWNLOAD(OPENPIFPAF_MODEL, pifpaf_model, quiet=False)

    if args.mode == 'keypoints':
        return dic_models
    if args.model is not None:
        assert os.path.exists(args.model), "Model path not found"
        dic_models[args.mode] = args.model
        return dic_models
    if args.mode == 'stereo':
        assert 'social_distance' not in args.activities, "Social distance not supported in stereo modality"
        path = MONSTEREO_MODEL
        name = 'monstereo-201202-1212.pkl'
    elif args.calibration == 'kitti' or args.path_gt is not None:
        path = MONOLOCO_MODEL_KI
        name = 'monoloco_pp-201203-1424.pkl'
    else:
        path = MONOLOCO_MODEL_NU
        name = 'monoloco_pp-201207-1350.pkl'
    model = os.path.join(torch_dir, name)
    dic_models[args.mode] = model

    if not os.path.exists(model):
        os.makedirs(torch_dir, exist_ok=True)
        assert DOWNLOAD is not None, \
            "pip install gdown to download a monoloco model, or pass the model path as --model"
        LOG.info('Downloading model in %s', torch_dir)
        DOWNLOAD(path, model, quiet=False)
    print(f"Using model: {name}")
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
    if not args.output_types and args.mode != 'keypoints':
        args.output_types = ['multi']
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

    if args.mode != 'keypoints':
        assert any((xx in args.output_types for xx in ['front', 'bird', 'multi', 'json'])), \
            "No output type specified, please select one among front, bird, multi, json, or choose mode=keypoints"

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
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

    # for openpifpaf predictions
    predictor = Predictor(checkpoint=args.checkpoint)

    # data
    data = datasets.ImageList(args.images, preprocess=predictor.preprocess)
    if args.mode == 'stereo':
        assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"

    pifpaf_outs = {}
    start = time.time()
    timing = []
    for idx, (pred, _, meta) in enumerate(predictor.images(args.images, batch_size=args.batch_size)):

        if idx % args.batch_size != 0:  # Only for MonStereo
            pifpaf_outs['right'] = [ann.json_data() for ann in pred]
        else:
            if args.json_output is not None:
                json_out_name = out_name(args.json_output, meta['file_name'], '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in pred], f)

            pifpaf_outs['pred'] = pred
            pifpaf_outs['left'] = [ann.json_data() for ann in pred]
            pifpaf_outs['file_name'] = meta['file_name']
            pifpaf_outs['width_height'] = meta['width_height']

            # Set output image name
            if args.output_directory is None:
                splits = os.path.split(meta['file_name'])
                output_path = os.path.join(splits[0], 'out_' + splits[1])
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(
                    args.output_directory, 'out_' + file_name)

            im_name = os.path.basename(meta['file_name'])
            print(f'{idx} image {im_name} saved as {output_path}')

        if (args.mode == 'mono') or (args.mode == 'stereo' and idx % args.batch_size != 0):
            # 3D Predictions
            if args.mode == 'keypoints':
                dic_out = defaultdict(list)
                kk = None
            else:
                im_size = (float(pifpaf_outs['width_height'][0]), float(pifpaf_outs['width_height'][1]))

                if args.path_gt is not None:
                    dic_gt, kk = factory_for_gt(args.path_gt, im_name)
                else:
                    kk = load_calibration(args.calibration, im_size, focal_length=args.focal_length)
                    dic_gt = None
                # Preprocess pifpaf outputs and run monoloco
                boxes, keypoints = preprocess_pifpaf(
                    pifpaf_outs['left'], im_size, enlarge_boxes=False)

                if args.mode == 'mono':
                    LOG.info("Prediction with MonoLoco++")
                    dic_out = net.forward(keypoints, kk)
                    fwd_time = (time.time()-start)*1000
                    timing.append(fwd_time)  # Skip Reordering and saving images
                    print(f"Forward time: {fwd_time:.0f} ms")
                    dic_out = net.post_process(
                        dic_out, boxes, keypoints, kk, dic_gt)
                    if 'social_distance' in args.activities:
                        dic_out = net.social_distance(dic_out, args)
                    if 'raise_hand' in args.activities:
                        dic_out = net.raising_hand(dic_out, keypoints)

                else:
                    LOG.info("Prediction with MonStereo")
                    _, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                    dic_out = net.forward(keypoints, kk, keypoints_r=keypoints_r)
                    fwd_time = (time.time()-start)*1000
                    timing.append(fwd_time)
                    dic_out = net.post_process(
                        dic_out, boxes, keypoints, kk, dic_gt)

            # Output
            factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=kk)
            print(f'Image {cnt}\n' + '-' * 120)
            cnt += 1
            start = time.time()
    timing = np.array(timing)
    avg_time = int(np.mean(timing))
    std_time = int(np.std(timing))
    print(f'Processed {idx * args.batch_size} images with an average time of {avg_time} ms and a std of {std_time} ms')


def factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=None):
    """
    Output json files or images according to the choice
    """
    if 'json' in args.output_types:
        with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
            json.dump(dic_out, ff)
        if len(args.output_types) == 1:
            return

    with open(pifpaf_outs['file_name'], 'rb') as f:
        cpu_image = PIL.Image.open(f).convert('RGB')
    if args.mode == 'keypoints':
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(cpu_image, output_path) as ax:
            annotation_painter.annotations(ax, pifpaf_outs['pred'])
        return

    if any((xx in args.output_types for xx in ['front', 'bird', 'multi'])):
        LOG.info(output_path)
        if args.activities:
            show_activities(
                args, cpu_image, output_path, pifpaf_outs['left'], dic_out)
        else:
            printer = Printer(cpu_image, output_path, kk, args)
            figures, axes = printer.factory_axes(dic_out)
            printer.draw(figures, axes, cpu_image, dic_out)
