
import glob
import logging
import os
import cv2
import sys


def resize(input_glob, output_dir, factor=2):
    """
    Resize images using multiplicative factor
    """
    list_im = glob.glob(input_glob)


    for idx, path_in in enumerate(list_im):

        basename, _ = os.path.splitext(os.path.basename(path_in))
        im = cv2.imread(path_in)
        assert im is not None, "Image not found"

        # Paddle the image if requested and resized the dataset to a fixed dataset
        h_im = im.shape[0]
        w_im = im.shape[1]
        w_new = round(factor * w_im)
        h_new = round(factor * h_im)

        print("resizing image {} to: {} x {}".format(basename, w_new, h_new))
        im_new = cv2.resize(im, (w_new, h_new))

        # Save the image
        name_im = basename + '.png'
        path_out = os.path.join(output_dir, name_im)
        cv2.imwrite(path_out, im_new)
        sys.stdout.write('\r' + 'Saving image number: {}'.format(idx) + '\t')


