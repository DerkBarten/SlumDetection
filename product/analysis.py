import re
import os
import sys
import glob
import math
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from groundtruth import create_dict
from util import read_geotiff, concat_tiff
from groundtruth import create_mask, create_groundtruth
from groundtruth import reshape_image, create_dataset


def parse_filename(filename):
    """
    Extracts the scale and block size from the name of a feature files that
    use the same convention of naming used in spfeas.

    Args:
        filename:   Name of the feature file; string.

    Returns:
        block:  The block size; integer.
        scale:  The scale; integer.

    """
    scale = int(re.search('(?<=SC)[0-9]*', filename).group(0))
    block = int(re.search('(?<=BK)[0-9]*', filename).group(0))
    return scale, block


def get_feature_path(image_name, block=20, scale=50, bands=[1, 2, 3],
                     feature_name='hog'):
    """
    Finds the path to the feature file based on the parameters used in its
    creation.

    Args:
        image_name:     The name of the image the feature file was created
                        from; string.
        block:          The block size used in the creation of the feature;
                        integer.
        scale:          The scale used in the creation of the feature, integer.
        bands:          The color bands used in the creation of the feature;
                        array of integers.
        feature_name:   The name of the feature, this is either 'hog' or
                        'lsr'; string.
    Returns:
        The path to the feature file if it exists, else it returns None.

    """

    folder = "features/features/{}__BD{}-{}-{}_BK{}_SC{}_TR{}/".format(
                os.path.basename(os.path.splitext(image_name)[0]), bands[0],
                bands[1], bands[2], block, scale, feature_name)
    filename = "*01.tif"

    path = os.path.join(folder, filename)
    path = glob.glob(path)

    if path and os.path.exists(path[0]):
        return path[0]
    return None


def create_feature(image_path, block, scale, bands, feature_name):
    """
    This function creates feature files by calling the spfeas program
    externally.

    Args:
        image_path:     The path to the image to extract the feature from.
        block:          The block size that will be used in the creation of the
                        feature; integer.
        scale:          The scale that will be used in the creation of the
                        feature; integer.
        bands:          The color bands will be used in the creation of the
                        feature; array of integers.
        feature_name:   The name of the feature to be created, this is either
                        'hog' or 'lsr'; string.

    """
    cmd = 'spfeas -i {} -o features --sect-size 100000 --block {} \
           --band-positions {} {} {} --scales {} --triggers {}'

    if block > scale:
        print("Block cannot be larger than scale")
        return
    spfeas = cmd.format(image_path, block, bands[0], bands[1], bands[2],
                        scale, feature_name)
    os.system(spfeas)


def analyze_feature(image_path, block, scale, bands, feature_name):
    """
    This function performs and analysis of a single feature using a boxplot,
    a heatmap, and kernel density estimation.

    Args:
        image_path:     The path to the image to extract the feature from;
                        string
        block:          The block size used in the creation of the feature;
                        integer.
        scale:          The scale used in the creation of the feature, integer.
        bands:          The color bands will be used in the creation of the
                        feature; array of integers.
        feature_name:   The name of the feature to be created, this is either
                        'hog' or 'lsr'; string.

    """
    shapefile = 'data/slums_approved.shp'
    image = np.array(read_geotiff(image_path))
    mask = create_mask(shapefile, image_path)
    feature_path = get_feature_path(image_path, block, scale, bands,
                                    feature_name)

    if feature_path is None:
        print("Error: cannot find feature file")
        return

    features_ = np.array(read_geotiff(feature_path))
    groundtruth = create_groundtruth(mask, block_size=block,
                                     threshold=0.5)

    image_shape = image[0].shape
    groundtruth = reshape_image(groundtruth, image_shape, block, scale)

    for i, feature_ in enumerate(features_):
        foldername = "analysis/{}_{}_BK{}_SC{}"
        featurename = "{}_{}_{}_BK{}_SC{}_F{}.png"
        base = os.path.basename(os.path.splitext(image_path)[0])

        featurefolder = foldername.format(base, feature_name, block, scale)

        if not os.path.exists(featurefolder):
            os.mkdir(featurefolder)

        dataset = create_dataset(feature_, groundtruth)
        name = featurename.format(base, 'boxplot', feature_name,
                                  block, scale, i)
        boxplot(dataset, featurefolder, name)

        dataset = create_dict(feature_, groundtruth)
        name = featurename.format(base, 'kde', feature_name,
                                  block, scale, i)
        kde(dataset, featurefolder, name)

        name = featurename.format(base, 'spatial', feature_name,
                                  block, scale, i)
        spatial_distribution(feature_, featurefolder, name)


def analysis(image_path, blocks, scales, bands, feature_names):
    """
    Overarching function for analysis. For every specified feature, block size,
    and scale, it first calculate the features and performs analysis on the
    features afterwards.

    image_path:     The path to the image to extract the feature from;
                    string
    blocks:         An array filled with the block sizes that will be used in
                    the creation of the features; integers.
    scale:          An array filled with the scales that will be used in the
                    creation of the feature, integers.
    bands:          The color bands will be used in the creation of the
                    feature; array of integers.
    feature_name:   An array filled with the name of the feature to be created,
                    this is either 'hog' or 'lsr'; strings.

    """
    for feature_name in feature_names:
        for block in blocks:
            for scale in scales:
                print("Processing {},\tfeature: {}\tblock: {}\tscale: {}\t".
                      format(os.path.basename(image_path), feature_name, block,
                             scale))
                # create_feature(image_path, block, scale, bands, feature_name)
                analyze_feature(image_path, block, scale, bands, feature_name)


def boxplot(dataset, folder, name):
    seaborn.boxplot(x="formality", y="feature", data=dataset)
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()


def kde(dataset, folder, name):
    seaborn.kdeplot(dataset['formal'], label="formal")
    seaborn.kdeplot(dataset['informal'], label="informal")
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()


def spatial_distribution(feature, folder, name):
    plt.imshow(feature, cmap="gray")
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()

if __name__ == "__main__":
    analysis('data/section_1.tif', [20], [150], [1, 2, 3], ['lsr'])
    # analysis('data/section_2.tif', [20, 40, 60], [50, 100, 150], [1, 2, 3],
    #          ['lsr', 'hog'])
    # analysis('data/section_3.tif', [20, 40, 60], [50, 100, 150], [1, 2, 3],
    #          ['lsr', 'hog'])
