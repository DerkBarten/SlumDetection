import re
import os
import sys
import glob
import math
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from groundtruth import create_dict
from util import read_geotiff
from groundtruth import create_mask, create_groundtruth
from groundtruth import reshape_image, create_dataset, overlay_groundtruth

from rid import RoadIntersectionDensity

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


def get_folder_name(image_name, block=20, scales=[50], bands=[1, 2, 3],
                     feature_names=['hog']):
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
    
    scale_string = get_scale_string(scales)
    feature_string = get_feature_string(sorted(feature_names), 'TR')

    folder = "features/features/{}__BD{}-{}-{}_BK{}_{}_{}/".format(
                os.path.basename(os.path.splitext(image_name)[0]), bands[0],
                bands[1], bands[2], block, scale_string, feature_string)
    
    return folder
    

def get_feature_from_folder(folder):
    path = os.path.join(folder, '*')
    path = glob.glob(path)

    if path and os.path.exists(path[0]):
        return path[0]
    return None


def create_feature(image_path, block, scales, bands, feature_names):
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
    spfeas_features = sorted(set(feature_names).intersection(['hog', 'lsr']))
    rid_feature = sorted(set(feature_names).intersection(['rid']))
    
    if spfeas_features:
        cmd = 'spfeas -i {} -o features --sect-size 100000 --block {} \
            --band-positions {} {} {} --scales {} --triggers {}'

        if block > max(scales):
            print("Block cannot be larger than scale")
            return
        scale_string = " ".join(map(str, scales))
        feature_string = " ".join(map(str, spfeas_features))
        spfeas = cmd.format(image_path, block, bands[0], bands[1], bands[2],
                            scale_string, feature_string)
        print("Running spfeas command: {}".format(spfeas))
        os.system(spfeas)
    if rid_feature:
        print("Creating Road Intersection Density feature")
        folder = get_folder_name(image_path, block, scales, bands, ['rid'])
        if not os.path.exists(folder):
            os.mkdir(folder)

        path = os.path.join(folder, 'feature.rid')
        if os.path.exists(path):
            print("Feature already calculated, skipping ...")
            return

        rid = RoadIntersectionDensity(image_path, block_size=block,
                                      scale=max(scales))
        RoadIntersectionDensity.save(rid, path)


def get_scale_string(scales):
    scale_string = 'SC'
    for i, scale in enumerate(scales):
        if i == len(scales) - 1:
            scale_string += str(scale)
        else:
            scale_string += '{}-'.format(scale)
    return scale_string


def get_feature_string(names, prefix=''):
    feature_string = prefix
    for i, name in enumerate(names):
        if i == len(names) - 1:
            feature_string += str(name)
        else:
            feature_string += '{}-'.format(name)
    return feature_string


def analyze_feature(image_path, block, scales, bands, feature_names):
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
    
    groundtruth = create_groundtruth(mask, block_size=block,
                                     threshold=0.6)

    # print(image.shape)
    # image = np.dstack((image[0], image[1], image[2]))
    # print(image.shape)
    # overlay_groundtruth(groundtruth, image, block)
    # exit()
    features_ = get_features(image_path, block, scales, bands, feature_names)
    image_shape = image[0].shape
    groundtruth = reshape_image(groundtruth, image_shape, block, max(scales))

    scale_string = get_scale_string(scales)
    feature_string = get_feature_string(feature_names)

    base = os.path.basename(os.path.splitext(image_path)[0])
    foldername = "analysis/{}_{}_BK{}_{}"

    for i, feature_ in enumerate(features_):
        featurename = "{}_{}_{}_BK{}_{}_F{}.png"
        featurefolder = foldername.format(base, feature_string, block,
                                          scale_string)

        if not os.path.exists(featurefolder):
            os.mkdir(featurefolder)

        dataset = create_dataset(feature_, groundtruth)
        name = featurename.format(base, 'boxplot', feature_string,
                                  block, scale_string, i)
        boxplot(dataset, featurefolder, name)

        dataset = create_dict(feature_, groundtruth)
        name = featurename.format(base, 'kde', feature_string,
                                  block, scale_string, i)
        kde(dataset, featurefolder, name)

        name = featurename.format(base, 'spatial', feature_string,
                                  block, scale_string, i)
        spatial_distribution(feature_, featurefolder, name)


def get_divergence(features):
    for feature in features:
        foldername = "analysis/{}_{}_BK{}_{}"
        featurename = "{}_{}_{}_BK{}_{}_F{}.txt"
        
        featurefolder = foldername.format(base, feature_string, block,
                                          scale_string)

        if not os.path.exists(featurefolder):
            os.mkdir(featurefolder)
    name = featurename.format(base, 'entropy', feature_string,
                              block, scale_string, 0)
    entropy(dataset)


def get_features(image_path, block, scales, bands, feature_names):
    if not set(feature_names).intersection(['hog', 'lsr', 'rid']):
        print("Error: cannot find specified feature, invalid feature name")
        exit()

    spfeas_features = sorted(set(feature_names).intersection(['hog', 'lsr']))
    rid_feature = sorted(set(feature_names).intersection(['rid']))

    if spfeas_features:
        folder = get_folder_name(image_path, block, scales, bands,
                                 spfeas_features)
        feature_path = get_feature_from_folder(folder)
        if feature_path is None:
            print("Error: cannot find feature: {}".format(folder))
            exit()

        spfeas_features = np.array(read_geotiff(feature_path))

    if rid_feature:
        folder = get_folder_name(image_path, block, [max(scales)], bands, ['rid'])
        feature_path = get_feature_from_folder(folder)

        if feature_path is None:
            print("Error: cannot find feature: {}".format(folder))
            exit()
        
        rid = RoadIntersectionDensity.load(feature_path)
        rid_feature = rid.get_feature()

    if len(spfeas_features) > 0 and len(rid_feature) > 0:
        return np.concatenate((spfeas_features, rid_feature), axis=0)
    if len(spfeas_features) > 0:
        return spfeas_features
    if len(rid_feature) > 0:
        return rid_feature


def analysis(image_path, blocks, scales_list, bands, feature_names_list):
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
    for feature_names in feature_names_list:
        for block in blocks:
            for scales in scales_list:
                print("Processing {},\tfeature_list: {}\tblock: {}\tscale_list: {}\t".
                      format(os.path.basename(image_path), feature_names, block,
                             scales_list))
                # create_feature(image_path, block, scales, bands, feature_names)
                analyze_feature(image_path, block, scales, bands, feature_names)


def boxplot(dataset, folder, name):
    seaborn.boxplot(x="formality", y="feature", data=dataset)
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()


def kde(dataset, folder, name):
    seaborn.kdeplot(dataset['formal'], label="Formal")
    seaborn.kdeplot(dataset['informal'], label="Informal")
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()


def spatial_distribution(feature, folder, name):
    plt.imshow(feature, cmap="gray")
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()

if __name__ == "__main__":
    analysis('data/section_5.tif', [20], [[50, 100, 150]], [1, 2, 3], [['hog'], ['lsr'], ['rid']])
    analysis('data/section_4.tif', [20], [[50, 100, 150]], [1, 2, 3], [['hog'], ['lsr'], ['rid']])
    analysis('data/section_3.tif', [20], [[50, 100, 150]], [1, 2, 3], [['hog'], ['lsr'], ['rid']])
    analysis('data/section_2.tif', [20], [[50, 100, 150]], [1, 2, 3], [['hog'], ['lsr'], ['rid']])
    analysis('data/section_1.tif', [20], [[50, 100, 150]], [1, 2, 3], [['hog'], ['lsr'], ['rid']])
    