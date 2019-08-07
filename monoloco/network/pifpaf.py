
import glob

import numpy as np
import torchvision
import torch
from PIL import Image, ImageFile
from openpifpaf.network import nets
from openpifpaf import decoder

from .process import image_transform


class ImageList(torch.utils.data.Dataset):
    """It defines transformations to apply to images and outputs of the dataloader"""
    def __init__(self, image_paths, scale):
        self.image_paths = image_paths
        self.scale = scale

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
        # PIL images are not iterables
        original_image = torchvision.transforms.functional.to_tensor(image)  # 0-255 --> 0-1
        image = image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


def factory_from_args(args):

    # Merge the model_pifpaf argument
    if not args.checkpoint:
        args.checkpoint = 'resnet152'  # Default model Resnet 152
    # glob
    if not args.webcam:
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


class PifPaf:
    def __init__(self, args):
        """Instanciate the mdodel"""
        factory_from_args(args)
        model_pifpaf, _ = nets.factory_from_args(args)
        model_pifpaf = model_pifpaf.to(args.device)
        self.processor = decoder.factory_from_args(args, model_pifpaf)
        self.keypoints_whole = []

        # Scale the keypoints to the original image size for printing (if not webcam)
        if not args.webcam:
            self.scale_np = np.array([args.scale, args.scale, 1] * 17).reshape(17, 3)
        else:
            self.scale_np = np.array([1, 1, 1] * 17).reshape(17, 3)

    def fields(self, processed_images):
        """Encoder for pif and paf fields"""
        fields_batch = self.processor.fields(processed_images)
        return fields_batch

    def forward(self, image, processed_image_cpu, fields):
        """Decoder, from pif and paf fields to keypoints"""
        self.processor.set_cpu_image(image, processed_image_cpu)
        keypoint_sets, scores = self.processor.keypoint_sets(fields)

        if keypoint_sets.size > 0:
            self.keypoints_whole.append(np.around((keypoint_sets / self.scale_np), 1)
                                        .reshape(keypoint_sets.shape[0], -1).tolist())

        pifpaf_out = [
            {'keypoints': np.around(kps / self.scale_np, 1).reshape(-1).tolist(),
             'bbox': [np.min(kps[:, 0]) / self.scale_np[0, 0], np.min(kps[:, 1]) / self.scale_np[0, 0],
                      np.max(kps[:, 0]) / self.scale_np[0, 0], np.max(kps[:, 1]) / self.scale_np[0, 0]]}
            for kps in keypoint_sets
        ]
        return keypoint_sets, scores, pifpaf_out
