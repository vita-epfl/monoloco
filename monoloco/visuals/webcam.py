# pylint: disable=W0212
"""
Webcam demo application

Implementation adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/webcam.py

"""

import time
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from openpifpaf import decoder, network, visualizer, show
import openpifpaf.datasets as datasets
from openpifpaf.predict import processor_factory, preprocess_factory

from ..visuals import Printer
from ..network import Loco
from ..network.process import preprocess_pifpaf, factory_for_gt

OPENPIFPAF_PATH = 'data/models/shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl'


def factory_from_args(args):

    # Model
    if not args.checkpoint:
        if os.path.exists(OPENPIFPAF_PATH):
            args.checkpoint = OPENPIFPAF_PATH
        else:
            args.checkpoint = 'shufflenetv2k30'

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    if args.net == 'monstereo':
        args.batch_size = 2
    else:
        args.batch_size = 1

    # Make default pifpaf argument
    args.force_complete_pose = True

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def webcam(args):

    args = factory_from_args(args)
    # Load Models
    net = Loco(model=args.model, net=args.net, device=args.device,
               n_dropout=args.n_dropout, p_dropout=args.dropout)

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # Start recording
    cam = cv2.VideoCapture(0)
    visualizer_monstereo = None

    while True:
        start = time.time()
        ret, frame = cam.read()
        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape
        print('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        data = datasets.PilImageList(
            make_list(pil_image), preprocess=preprocess)

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False,
            pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

        for (image_tensors_batch, _, meta_batch) in data_loader:
            pred_batch = processor.batch(
                model, image_tensors_batch, device=args.device)

            for idx, (pred, meta) in enumerate(zip(pred_batch, meta_batch)):
                pred = [ann.inverse_transform(meta) for ann in pred]

                if idx == 0:
                    pifpaf_outs = {
                        'pred': pred,
                        'left': [ann.json_data() for ann in pred],
                        'image': image}
                else:
                    pifpaf_outs['right'] = [ann.json_data() for ann in pred]

        if not ret:
            break
        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        intrinsic_size = [xx * 1.3 for xx in pil_image.size]
        kk, dic_gt = factory_for_gt(intrinsic_size,
                                    focal_length=args.focal,
                                    path_gt=args.path_gt)  # better intrinsics for mac camera
        boxes, keypoints = preprocess_pifpaf(
            pifpaf_outs['left'], (width, height))

        dic_out = net.forward(keypoints, kk)
        dic_out = net.post_process(dic_out, boxes, keypoints, kk, dic_gt)

        if args.activities:
            if 'social_distance' in args.activities:
                dic_out = net.social_distance(dic_out, args)
            if 'raise_hand' in args.activities:
                dic_out = net.raising_hand(dic_out, keypoints)
        if visualizer_monstereo is None:  # it is, at the beginning
            visualizer_monstereo = VisualizerMonstereo(kk,
                                                       args)(pil_image)  # create it with the first image
            visualizer_monstereo.send(None)

        print(dic_out)
        visualizer_monstereo.send((pil_image, dic_out, pifpaf_outs))

        end = time.time()
        print("run-time: {:.2f} ms".format((end-start)*1000))

    cam.release()

    cv2.destroyAllWindows()


class VisualizerMonstereo:
    def __init__(self, kk, args):
        self.kk = kk
        self.args = args

    def __call__(self, first_image, fig_width=1.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width *
                                 first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, output_path="",
                          kk=self.kk, args=self.args)

        figures, axes = printer.factory_axes(None)

        for fig in figures:
            fig.show()

        while True:
            image, dic_out, pifpaf_outs = yield

            # Clears previous annotations between frames
            axes[0].patches = []
            axes[0].lines = []
            axes[0].texts = []
            if len(axes) > 1:
                axes[1].patches = []
                axes[1].lines = [axes[1].lines[0], axes[1].lines[1]]
                axes[1].texts = []

            if dic_out:
                printer._process_results(dic_out)
                printer.draw(figures, axes, image, dic_out, pifpaf_outs['left'])
                mypause(0.01)


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


def make_list(*args):
    return list(args)
