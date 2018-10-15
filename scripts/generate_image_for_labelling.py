import os

import click

import numpy as np

from imageio import imread, imsave

from skimage.exposure import equalize_adapthist


def rescale_as_float(im):

    return (im.astype(np.float32) - im.min()) / (im.max() - im.min())


def generate_image_for_labelling(input_fpath, output_fpath):
    """Generate an image for manual seed marking"""

    im = imread(input_fpath)
    scaled = rescale_as_float(im)

    eq = equalize_adapthist(scaled)

    imsave(output_fpath, eq)


@click.command()
@click.argument('nuclear_stain_fpath')
def main(nuclear_stain_fpath):

    dirname = os.path.dirname(nuclear_stain_fpath)
    label = os.path.basename(dirname)
    output_fpath = "{}-image_to_label.png".format(label)
    generate_image_for_labelling(nuclear_stain_fpath, output_fpath)


if __name__ == '__main__':
    main()
