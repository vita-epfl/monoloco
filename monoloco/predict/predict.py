
import os
from PIL import Image

import torch

from ..predict.pifpaf import PifPaf, ImageList
from ..predict.network import MonoLoco
from ..predict.factory import factory_for_gt, factory_outputs
from ..utils.pifpaf import preprocess_pif


def predict(args):

    cnt = 0

    # load pifpaf and monoloco models
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model_path=args.model, device=args.device, n_dropout=args.n_dropout, p_dropout=args.dropout)

    # data
    data = ImageList(args.images, scale=args.scale)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    for idx, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = pifpaf.fields(processed_images)

        # unbatch
        for image_path, image, processed_image_cpu, fields in zip(
                image_paths, images, processed_images_cpu, fields_batch):

            if args.output_directory is None:
                output_path = image_path
            else:
                file_name = os.path.basename(image_path)
                output_path = os.path.join(args.output_directory, file_name)
            print('image', idx, image_path, output_path)

            keypoint_sets, scores, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)
            pifpaf_outputs = [keypoint_sets, scores, pifpaf_out]  # keypoints_sets and scores for pifpaf printing
            images_outputs = [image]  # List of 1 or 2 elements with pifpaf tensor (resized) and monoloco original image

            if 'monoloco' in args.networks:
                im_size = (float(image.size()[1] / args.scale),
                           float(image.size()[0] / args.scale))  # Width, Height (original)

                # Extract calibration matrix and ground truth file if present
                with open(image_path, 'rb') as f:
                    pil_image = Image.open(f).convert('RGB')
                    images_outputs.append(pil_image)

                im_name = os.path.basename(image_path)

                kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)

                # Preprocess pifpaf outputs and run monoloco
                boxes, keypoints = preprocess_pif(pifpaf_out, im_size)
                outputs, varss = monoloco.forward(keypoints, kk)
                dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dic_gt)

            else:
                dic_out = None
                kk = None

            factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=dic_out, kk=kk)
            print('Image {}\n'.format(cnt) + '-' * 120)
            cnt += 1
