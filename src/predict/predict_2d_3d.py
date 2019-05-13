
import glob
import json
import os

import numpy as np
import torch

from openpifpaf.network import nets
from openpifpaf import decoder, show
from openpifpaf import transforms
from predict.predict_monoloco import PredictMonoLoco
from utils.pifpaf import preprocess_pif

import torchvision
import torch

from PIL import Image, ImageFile


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, scale, image_transform=None):
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.image_transform
        self.scale = scale

        # data = datasets.ImageList(args.images, preprocess=transforms.RescaleRelative(2
        # .0)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.scale > 1.01 or self.scale < 0.99:
            image = torchvision.transforms.functional.resize(image,
                                                             (round(self.scale * image.size[1]),
                                                              round(self.scale * image.size[0])),
                                                             interpolation=Image.BICUBIC)
        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


def elaborate_cli(args):

    # Merge the model_pifpaf argument
    if not args.checkpoint:
        args.checkpoint = args.model_pifpaf
    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    # Add num_workers
    args.loader_workers = 8

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    return args


def predict(args):

    elaborate_cli(args)

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)

    # data
    data = ImageList(args.images, scale=args.scale)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    # Visualizer
    keypoint_painter = show.KeypointPainter(show_box=True)
    skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=4)

    keypoints_whole = []
    for idx, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = processor.fields(processed_images)

        # unbatch
        for image_path, image, processed_image_cpu, fields in zip(
                image_paths,
                images,
                processed_images_cpu,
                fields_batch):

            if args.output_directory is None:
                output_path = image_path
            else:
                file_name = os.path.basename(image_path)
                output_path = os.path.join(args.output_directory, file_name)
            print('image', idx, image_path, output_path)

            processor.set_cpu_image(image, processed_image_cpu)
            keypoint_sets, scores = processor.keypoint_sets(fields)

            # Correct to not change the confidence
            scale_np = np.array([args.scale, args.scale, 1] * 17).reshape(17, 3)

            pifpaf_out = [
                {'keypoints': np.around(kps / scale_np, 1).reshape(-1).tolist(),
                 'bbox': [np.min(kps[:, 0]) / args.scale, np.min(kps[:, 1]) / args.scale,
                          np.max(kps[:, 0]) / args.scale, np.max(kps[:, 1]) / args.scale]}
                for kps in keypoint_sets
            ]

            # Save json file
            if 'pifpaf' in args.networks:

                if 'json' in args.output_types and keypoint_sets.size > 0:
                    with open(output_path + '.pifpaf.json', 'w') as f:
                        json.dump(pifpaf_out, f)

                if keypoint_sets.size > 0:
                    keypoints_whole.append(np.around((keypoint_sets / scale_np), 1)
                                           .reshape(keypoint_sets.shape[0], -1).tolist())

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
                im_size = (float(image.size()[1] / args.scale),
                           float(image.size()[0] / args.scale))  # Width, Height (original)
                boxes, keypoints = preprocess_pif(pifpaf_out, im_size)
                predict_monoloco = PredictMonoLoco(boxes, keypoints, image_path, output_path, args)
                predict_monoloco.run()

    return keypoints_whole
