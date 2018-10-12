import numpy as np

from imageio import imread, imsave

from jicbioimage.segment import SegmentedImage


def imimsave(filename, im):

    converted = im.astype(np.uint8)

    rescaled = 255 * (converted - converted.min()) / (converted.max() - converted.min())

    imsave(filename, rescaled)


def rescale_as_float(im):

    return (im.astype(np.float32) - im.min()) / (im.max() - im.min())


def save_segmented_image_as_rgb(filename, segmented_image):

    segmentation_as_rgb = segmented_image.unique_color_image

    imsave(filename, segmentation_as_rgb)


def load_segmentation_from_rgb_image(filename):

    rgb_image = imread(filename)

    ydim, xdim, _ = rgb_image.shape

    segmentation = np.zeros((ydim, xdim), dtype=np.uint32)

    segmentation += rgb_image[:, :, 2]
    segmentation += rgb_image[:, :, 1].astype(np.uint32) * 256
    segmentation += rgb_image[:, :, 0].astype(np.uint32) * 256 * 256

    return segmentation.view(SegmentedImage)
