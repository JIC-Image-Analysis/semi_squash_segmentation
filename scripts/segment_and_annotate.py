import os

import click

import numpy as np
import pandas as pd

from imageio import imread, imsave

from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

from jicbioimage.segment import SegmentedImage

from utils import (
    imimsave,
    rescale_as_float,
    save_segmented_image_as_rgb,
    load_segmentation_from_rgb_image
)


def try_voronoi(csv_fpath, output_fpath):

    results = pd.read_csv(csv_fpath)

    centroids = list(zip(results.X, results.Y))

    rows = 1936
    cols = 1460

    voronoi_image = np.zeros((cols, rows), dtype=np.uint8)

    c_array = np.array(centroids)

    def closest_label(p):
        delta = c_array - p[None,:]
        dsq = np.sum(np.multiply(delta, delta), axis=1)
        return np.argmin(dsq)

    for r in np.arange(0, rows, 1):
        for c in np.arange(0, cols, 1):
            p = np.array((r, c))
            label = closest_label(p)
            voronoi_image[c, r] = label

    vsegmented = voronoi_image.view(SegmentedImage)

    save_segmented_image_as_rgb(output_fpath, vsegmented)


def convert_nuclei_centroids_to_voronoi(csv_fpath, output_fpath):

    try_voronoi(csv_fpath, output_fpath)


def generate_mask_image(image_fpath, output_fpath):

    im = imread(image_fpath)

    scaled = rescale_as_float(im)

    imsave('auto.png', scaled)

    adapt = equalize_adapthist(scaled)

    imsave('adapt.png', adapt)

    smoothed = gaussian(adapt, sigma=5)

    mask = smoothed > 0.2

    imimsave(output_fpath, mask)


def apply_mask_to_image(mask, image):

    masked_image = np.zeros((image.shape), image.dtype)

    masked_image[np.where(mask)] = image[np.where(mask)]

    return masked_image


def make_composite_image(c0, c1):

    # c1 = rescale_as_float(c1)

    rdim, cdim = c0.shape

    composite = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    composite[:,:,0] = c0
    # composite[:,:,2] = c1

    # composite[np.where(c1 > 0),2] = c1

    for r, c in zip(*np.where(c1 > 0)):
        composite[r, c, 2] = c1[r, c]
        composite[r, c, 1] = c1[r, c]

    # composite[np.where(c1 > 0.3)] = [0, 255, 255]

    return composite


def generate_annotated_segmentation_image(label):

    voronoi_image_fpath = "{}-voronoi.png".format(label)
    voronoi = load_segmentation_from_rgb_image(voronoi_image_fpath)

    mask_image_fpath = "{}-mask.png".format(label)
    mask = rescale_as_float(imread(mask_image_fpath))

    masked_image = apply_mask_to_image(mask, voronoi)
    imimsave('pretty.png', masked_image.view(SegmentedImage).pretty_color_image)

    boundaries = find_boundaries(masked_image)

    imimsave('boundaries.png', boundaries)

    # TODO - FIXME
    label_image_fpath = "{}-image_to_label.png".format(label)
    label_image = imread(label_image_fpath)
    comp = make_composite_image(imread('adapt.png'), label_image)

    comp[np.where(boundaries)] = [255, 255, 0]

    segmentation_image_fpath = "{}-segmented.png".format(label)
    imsave(segmentation_image_fpath, comp)


@click.command()
@click.argument('data_fpath')
def main(data_fpath):

    label = os.path.basename(data_fpath)

    nuclei_positions = "{}-results.csv".format(label)
    voronoi_image_fpath = "{}-voronoi.png".format(label)

    if not os.path.exists(voronoi_image_fpath):
        convert_nuclei_centroids_to_voronoi(nuclei_positions, voronoi_image_fpath)

    autof_image_fpath = os.path.join(data_fpath, 'converted_S0_C1_Z20.png')
    mask_image_fpath = "{}-mask.png".format(label)

    if not os.path.exists(mask_image_fpath):
        generate_mask_image(autof_image_fpath, mask_image_fpath)

    segmentation_image_fpath = "{}-segmented.png".format(label)

    if not os.path.exists(segmentation_image_fpath):
        generate_annotated_segmentation_image(label)


if __name__ == '__main__':
    main()
