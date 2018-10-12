from pathlib import Path

import click

import numpy as np
import pandas as pd

from imageio import imread, imsave

from skimage.morphology import reconstruction
from skimage.filters import gaussian, median
from skimage.exposure import equalize_adapthist
from skimage.measure import label
from skimage.segmentation import find_boundaries

from jicbioimage.segment import SegmentedImage
from jicbioimage.core.util.color import pretty_color_from_identifier

from scipy.spatial import Voronoi


class ImageList(object):

    @classmethod
    def from_fpath_list(cls, fpath_list):

        new_class = cls()

        new_class.images = map(imread, fpath_list)

        return new_class


# def imsave_f(im):

#     imsave()

def load_all_channel(image_dir, c):

    glob_str = "*C{}*.png".format(c)

    fpath_list = Path(image_dir).glob(glob_str)

    il = ImageList.from_fpath_list(fpath_list)

    # new_il = []
    # for im in il.images:


def rescale_as_float(im):

    return (im.astype(np.float32) - im.min()) / (im.max() - im.min())


def impose_cutoff(im, cutoff):

    transformed = im.copy()
    coords = np.where(im < cutoff)
    im[coords] = 0

    return im


def hdome(im, h):

    seed = np.copy(im) - h

    dilated = reconstruction(seed, im, method='dilation')

    return im - dilated


def background_subtraction(im):

    seed = np.copy(im)
    seed[1:-1, 1:-1] = im.min()

    dilated = reconstruction(seed, im, method='dilation')

    return im - dilated


def psweep(im, func, output_fpath):

    output_fpath = Path(output_fpath)

    output_fpath.mkdir(exist_ok=True, parents=True)

    def get_output_fpath(p):
        return output_fpath / "hdome_{:.1f}.png".format(p)

    for p in np.arange(0.1, 0.9, 0.1):
        im_out = func(im, p)
        # full_output_fpath = get_output_fpath(p)
        imsave(full_output_fpath, im_out)


def thresh_conv(im, thresh):

    thresholded = im > thresh

    return 255 * thresholded.astype(np.uint8)


def spike_single(fpath):

    im = imread(fpath)

    scaled = rescale_as_float(im)
    blurred = gaussian(scaled, sigma=5)

    # psweep(blurred, 'scratch/psweep')

    nobg = hdome(blurred, 0.6)

    # psweep(nobg, thresh_conv, 'scratch/tsweep')

    # imsave('scratch/thresh.png', 255 * (nobg > 0.5).astype(np.uint8))

    # imsave('scratch/scaled.png', scaled)

    # imsave('scratch/hdome.png', hdome(scaled, 0.5))

    # imsave('scratch/nobg.png', background_subtraction(scaled))
    # h = 0.4
    # seed = np.copy(scaled) - h
    # # seed[1:-1,1:-1] = scaled.min()
    # dilated = reconstruction(seed, scaled, method='dilation')

    # imsave('scratch/nobg.png', scaled - dilated)

    threshed = (nobg > 0.1).astype(np.float32)

    imsave('scratch/threshed.png', threshed)

    labelled = label(threshed)

    imsave('scratch/label.png', labelled)

    return labelled


def rescale8(im):

    return (255 * (im - im.min()) / (im.max() - im.min())).astype(np.uint8)


def spike_bg(fpath, labels):

    im = imread(fpath)

    scaled = rescale_as_float(im)

    imsave('scratch/auto.png', scaled)

    adapt = equalize_adapthist(scaled)

    imsave('scratch/adapt.png', adapt)

    smoothed = gaussian(adapt, sigma=10)

    imsave('scratch/smoothed.png', smoothed)

    mask = (smoothed > 0.4).astype(np.float32)

    imsave('scratch/mask.png', mask)

    from skimage.morphology import watershed

    # shedded = watershed(-smoothed, labels, connectivity=255 * mask.astype(np.uint8), mask=mask)
    smoothed = rescale8(smoothed)
    labels = rescale8(labels)
    mask = rescale8(mask)
    shedded = watershed(-smoothed, labels, mask=mask, connectivity=1)

    imsave('smooooooooth.png', smoothed)
    imsave('scratch/shedded.png', shedded)


def generate_image_for_labelling(fpath):
    """Generate an image for manual seed marking"""

    im = imread(fpath)
    scaled = rescale_as_float(im)

    eq = equalize_adapthist(scaled)

    imsave('newmarkmeq.png', eq)


def results_csv_to_label_image(csv_fpath):

    results = pd.read_csv(csv_fpath)

    centroids = list(zip(results.X, results.Y))

    rows = 1460
    cols = 1936

    label_image = np.zeros((rows, cols), dtype=np.uint8)

    sz = 20
    for l, centroid in enumerate(centroids):
        c, r = centroid
        label_image[r-sz:r+sz, c-sz:c+sz] = l

    imsave('label_image.png', label_image)

    return label_image


def try_both(nuclei_image, auto_image):

    im = imread(nuclei_image)

    scaled = rescale_as_float(im)
    blurred = gaussian(scaled, sigma=30)

    imsave('newblurred.png', blurred)

    eq = equalize_adapthist(blurred)

    imsave('neweq.png', eq)


def save_segmented_image_as_rgb(filename, segmented_image):

    segmentation_as_rgb = segmented_image.unique_color_image

    imsave(filename, segmentation_as_rgb)


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
            l = closest_label(p)
            voronoi_image[c, r] = l

    vsegmented = voronoi_image.view(SegmentedImage)

    save_segmented_image_as_rgb(output_fpath, vsegmented)


def convert_nuclei_centroids_to_voronoi(csv_fpath, output_fpath):

    try_voronoi(csv_fpath, output_fpath)


def voronoi_qhull(csv_fpath):

    results = pd.read_csv(csv_fpath)
    centroids = list(zip(results.X, results.Y))
    c_array = np.array(centroids)

    vor = Voronoi(c_array)

    print(dir(vor))

    print(vor.vertices)


def load_segmentation_from_rgb_image(filename):

    rgb_image = imread(filename)

    ydim, xdim, _ = rgb_image.shape

    segmentation = np.zeros((ydim, xdim), dtype=np.uint32)

    segmentation += rgb_image[:, :, 2]
    segmentation += rgb_image[:, :, 1].astype(np.uint32) * 256
    segmentation += rgb_image[:, :, 0].astype(np.uint32) * 256 * 256

    return segmentation.view(SegmentedImage)


def apply_mask_to_image(mask, image):

    masked_image = np.zeros((image.shape), image.dtype)

    masked_image[np.where(mask)] = image[np.where(mask)]

    return masked_image


def make_mask(filename):

    adapt = imread(filename)

    adapt_f = rescale_as_float(adapt)

    smoothed = gaussian(adapt_f, sigma=5)

    return smoothed > 0.2


def imimsave(filename, im):

    converted = im.astype(np.uint8)

    rescaled = 255 * (converted - converted.min()) / (converted.max() - converted.min())

    imsave(filename, rescaled)


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


def load_voronoi_and_experiment(filename):

    voronoi = load_segmentation_from_rgb_image(filename)

    mask = make_mask('scratch/adapt.png')

    masked_image = apply_mask_to_image(mask, voronoi)
    imsave('pretty.png', masked_image.view(SegmentedImage).pretty_color_image)

    boundaries = find_boundaries(masked_image)

    imimsave('boundaries.png', boundaries)

    comp = make_composite_image(imread('scratch/adapt.png'), imread('markmeq.png'))

    comp[np.where(boundaries)] = [255, 255, 0]

    imsave('comp.png', comp)


@click.command()
@click.argument('image_dir')
def main(image_dir):

    # generate_image_for_labelling('data_intermediate/ColFRI-semisq_PP2A_FLC_02/converted_S0_C0_Z20.png')
    # convert_nuclei_centroids_to_voronoi('C1Results.csv', 'C1Voronoi.png')

    # load_voronoi_and_experiment('vrgb.png')
    # try_voronoi('Results.csv')
    # voronoi_qhull('Results.csv')
    # try_both('data_intermediate/converted_S0C0Z20.png', 'data_intermediate/converted_S0C1Z20.png')
    # load_all_channel(image_dir, 0)

    # labels = spike_single('data_intermediate/converted_S0C0Z20.png')

    # spike_bg('data_intermediate/converted_S0C1Z20.png', labels)


    # label_image = results_csv_to_label_image('Results.csv')
    # spike_bg('data_intermediate/converted_S0C1Z20.png', label_image)


if __name__ == '__main__':
    main()
