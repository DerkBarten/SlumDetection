import math
import fiona
import rasterio
import argparse
import numpy as np
import pandas as pd

from rasterio import mask
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


def overlay_groundtruth(groundtruth, image, block_size):
    groundtruth = zoom(groundtruth, block_size, order=0)
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(groundtruth, alpha=0.5)
    plt.savefig('image.png', format='png', dpi=1000)
    plt.show()


def create_groundtruth(mask, block_size=20, threshold=0.1):
    """
    This functions creates a block based groundtruth map from the mask produced
    by the create_mask function. When a block of pixels has a certain amount of
    pixels that of the informal class, the pixel block will be assigned as
    informal.


    """
    minum_pixels = pow(block_size, 2) * threshold
    height = mask.shape[0]
    width = mask.shape[1]
    groundtruth = np.zeros((int(math.ceil(float(height) / block_size)),
                            int(math.ceil(float(width) / block_size))))

    for i in range(0, groundtruth.shape[0]):
        for j in range(groundtruth.shape[1]):
            chunck = mask[i * block_size:i * block_size + block_size,
                          j * block_size:j * block_size + block_size]
            if np.count_nonzero(chunck) > minum_pixels:
                groundtruth[i, j] = 1

    return groundtruth


def create_mask(shapefile, imagefile, maskname=None):
    with fiona.open(shapefile, "r") as sf:
        geoms = [feature["geometry"] for feature in sf]

    with rasterio.open(imagefile) as src:
        out_image, out_transform = mask.mask(src, geoms, crop=False,
                                             invert=False)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    if maskname:
        with rasterio.open(maskname + ".tif", "w", **out_meta) as dest:
            dest.write(out_image)

    THRESHOLD = 1000000
    out_image[out_image > THRESHOLD] = 0
    return out_image[1]


def create_dataset(feature, groundtruth):
    height, width = groundtruth.shape
    print(feature.shape)
    print(groundtruth.shape)
    dataset = {'feature': [], 'formality': []}

    for i in range(height):
        for j in range(width):
            if (groundtruth[i, j] == 0):
                dataset['formality'].append('Formal')
            else:
                dataset['formality'].append('Informal')
            
            dataset['feature'].append(feature[i, j])

    return pd.DataFrame.from_dict(dataset)


def create_dict(feature, groundtruth):
    height, width = groundtruth.shape
    dataset = {'Formal': [], 'Informal': []}

    for i in range(height):
        for j in range(width):
            if (groundtruth[i, j] != 0):
                dataset['Informal'].append(feature[i, j])
            else:
                dataset['Formal'].append(feature[i, j])
            
    return dataset


def calculate_padding(image_shape, block, scale):
    """
    Spfeas removes a few block for padding on the edge of the image when
    using scales larger than the block size. This function calculates the
    number of blocks that were removed for padding. This is based on the
    scale, image size and block size.

    Args:
        image_shape:    A tuple containing the shape of the image; integers.
        block:          The block size; integer.
        scale:          The scale; integer.

    Returns:
        The padding of the image in width and length; tuple of floats.

    """
    padding_x = math.ceil(image_shape[0] / float(block)) -\
        math.ceil(float(image_shape[0] - (scale - block)) / block)
    padding_y = math.ceil(image_shape[1] / float(block)) -\
        math.ceil(float(image_shape[1] - (scale - block)) / block)

    return (padding_x, padding_y)


def reshape_image(groundtruth, image_shape, block, scale):
    """
    This function resizes the groundtruth to the dimensions of the feature
    vector created by spfeas. The groundtruth needs to account for the padding
    introduced by the creation of the feature vector by spfeas.

    Args:
        groundtruth:    A zero filled nxm numpy matrix with ones on the
                        location of informal areas.
        image_shape:     A tuple containing the shape of the image, integers.
        block:          The block size; integer.
        scale:          The scale; integer.

    Returns:
        The groundtruth without padding in the same shape as the feature
        vector; nxm numpy matrix.

    """
    padding = calculate_padding(image_shape, block, scale)

    x_start = int(math.ceil(padding[0] / 2.0))
    x_end = int(math.floor(padding[0] / 2.0))
    y_start = int(math.ceil(padding[1] / 2.0))
    y_end = int(math.floor(padding[1] / 2.0))

    if x_end <= 0 and y_end <= 0:
        return groundtruth[x_start:, y_start:]
    if x_end <= 0:
        return groundtruth[x_start:, y_start:-y_end]
    if y_end <= 0:
        return groundtruth[x_start:-x_end, y_start:]
    return groundtruth[x_start:-x_end, y_start:-y_end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask from shape file")
    parser.add_argument("shapefile",  nargs="?",
                        help="The shapefile to use")
    parser.add_argument("imagefile",  nargs="?",
                        help="The imagefile to use")
    parser.add_argument("maskname", nargs="?",
                        help="Select name of the file where to save the mask.")

    args = parser.parse_args()

    mask = create_mask(args.shapefile, args.imagefile, args.maskname)
    groundtruth = create_groundtruth(mask)

    plt.imshow(groundtruth, cmap='gray')
    plt.title('Binary mask')    
    plt.show()
