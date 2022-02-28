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
    elif args.calibration == 'kitti':
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
    # else:
    #     args.batch_size = 1

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

    # for openpifpaf predictions
    predictor = Predictor(checkpoint=args.checkpoint)

    'glob("/path/to/directory/*/", recursive = True)'
    base_folder = '/data/lorenzo-data/JRDB_activity/images/'
    im_folders = ['image_0', 'image_2', 'image_4', 'image_6', 'image_8', 'image_stitched']
    out_folder = '/data/lorenzo-data/pifpaf-data/JRDB-activity/'

    for im_folder in im_folders:
        folders = glob.glob(os.path.join(base_folder, im_folder) + '/*')
        for folder in folders:
            args.images = glob.glob(folder + '/*.jpg')

            for idx, (pred, _, meta) in enumerate(predictor.images(args.images, batch_size=args.batch_size)):
                folder_name = os.path.basename(folder)
                im_name = os.path.basename(meta['file_name'])
                json_out_name = os.path.join(out_folder, folder_name + '_' + im_folder + '_' + im_name + '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in pred], f)
                print(f'Saved {json_out_name}')
