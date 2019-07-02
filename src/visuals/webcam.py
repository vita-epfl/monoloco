
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from openpifpaf.network import nets
from openpifpaf import decoder
from openpifpaf import transforms
from visuals.printer import Printer
from utils.pifpaf import preprocess_pif
from predict.monoloco import MonoLoco
from predict.factory import factory_for_gt


def webcam(args):

    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')

    # load models
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)
    monoloco = MonoLoco(model_path=args.model, device=args.device)

    # Start recording
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        visualizer_monoloco = None
        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape
        print('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = transforms.image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        fields = processor.fields(torch.unsqueeze(processed_image, 0))[0]
        processor.set_cpu_image(image, processed_image_cpu)
        keypoint_sets, scores = processor.keypoint_sets(fields)

        pifpaf_out = [
            {'keypoints': np.around(kps, 1).reshape(-1).tolist(),
             'bbox': [np.min(kps[kps[:, 2] > 0, 0]), np.min(kps[kps[:, 2] > 0, 1]),
                      np.max(kps[kps[:, 2] > 0, 0]), np.max(kps[kps[:, 2] > 0, 1])]}
            for kps in keypoint_sets
        ]

        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        pil_image = Image.fromarray(image)

        kk, dict_gt = factory_for_gt(pil_image.size)
        if visualizer_monoloco is None:
                visualizer_monoloco = VisualizerMonoloco(kk, args)(pil_image)
                visualizer_monoloco.send(None)

        if pifpaf_out:
            boxes, keypoints = preprocess_pif(pifpaf_out, (width, height))
            outputs, varss = monoloco.forward(keypoints,  kk)
            dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dict_gt)
            visualizer_monoloco.send((pil_image, dic_out))

    cam.release()

    cv2.destroyAllWindows()


class VisualizerMonoloco:
    def __init__(self, kk, args, epistemic=False):
        self.kk = kk
        self.args = args
        self.z_max = args.z_max
        self.epistemic = epistemic
        self.output_types = args.output_types

    def __call__(self, first_image, fig_width=4.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width * first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, output_path="", kk=self.kk, output_types=self.output_types,
                          z_max=self.z_max, epistemic=self.epistemic)
        figures, axes = printer.factory_axes()

        for fig in figures:
            fig.show()

        while True:
            image, dict_ann = yield
            draw_start = time.time()
            while (len(axes)>0 and axes[0].patches):
                del axes[0].patches[0]
                del axes[0].texts[0]
                if len(axes) == 2:
                    del axes[1].patches[0]
                    if len(axes[1].lines) > 2:
                        del axes[1].lines[2]
                        del axes[1].texts[0]
            printer.draw(figures, axes, dict_ann, image)
            print('draw', time.time() - draw_start)
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
