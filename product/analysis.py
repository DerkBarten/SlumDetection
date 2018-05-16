import numpy as np
import seaborn
from util import read_geotiff, concat_tiff
import matplotlib.pyplot as plt
import argparse
from groundtruth import create_mask, create_groundtruth, create_dataset, create_dict
import logging
import logging.config
import math
import matplotlib.ticker as mtick
import re
import os



def value_distribution(feature):
    seaborn.kdeplot(np.ravel(feature))
    plt.show()


def feature_distribution(dataset, axis):
    seaborn.kdeplot(dataset['formal'], ax=axis, label="formal")
    seaborn.kdeplot(dataset['informal'], ax=axis, label="informal")


def spatial_distribution(feature):
    plt.imshow(feature, cmap="hot")
    plt.show()


def analyze(feature):
    value_distribution(feature)
    spatial_distribution(feature)


def calculate_padding(image, block, scale):
    padding_x = math.ceil(image.shape[0] / float(block)) -\
        math.ceil(float(image.shape[0] - (scale - block)) / block)
    padding_y = math.ceil(image.shape[1] / float(block)) -\
        math.ceil(float(image.shape[1] - (scale - block)) / block)
    return (padding_x, padding_y)


def reshape_image(groundtruth, image, block, scale):
    dimensions = image.shape
    padding = calculate_padding(image, block, scale)

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


def parse_name(filename):
    scale = int(re.search('(?<=SC)[0-9]*', filename).group(0))
    block = int(re.search('(?<=BK)[0-9]*', filename).group(0))
    return scale, block


def get_featurefile(imagename, block=20, scale=50, bands=[1, 2, 3],
                    feature='hog'):
    return "testfeatures/{}__BD{}-{}-{}_BK{}_SC{}_TR{}.vrt".format(
                os.path.basename(os.path.splitext(imagename)[0]), bands[0],
                bands[1], bands[2], block, scale, feature)


def create_features(image, blocks, scales, bands, features):
    cmd = 'spfeas -i {} -o testfeatures --block {} --band-positions {} {} {} --scales {} --triggers {}'

    for feature in features:
        for block in blocks:
            for scale in scales:
                spfeas = cmd.format(image, block, bands[0], bands[1], bands[2],
                                    scale, feature)
                os.system(spfeas)


def analyze_features(imagefile, blocks, scales, bands, features):
    image = np.array(read_geotiff(imagefile))
    shapefile = 'data/slums_approved.shp'
    mask = create_mask(shapefile, imagefile)[1]

    THRESHOLD = 1000000
    mask[mask > THRESHOLD] = 0

    for feature in features:
        for block in blocks:
            for scale in scales:
                featurefile = get_featurefile(imagefile, block, scale, bands,
                                              feature)
                features_ = np.array(read_geotiff(featurefile))
                groundtruth = create_groundtruth(mask, block_size=block,
                                     threshold=0)
                groundtruth = reshape_image(groundtruth, image, block, scale)

                for i, feature_ in enumerate(features_):
                    foldername = "analysis/{}_BK{}_SC{}"
                    featurename = "{}_{}_BK{}_SC{}_F{}.png"
                    base = os.path.basename(os.path.splitext(imagefile)[0])

                    featurefolder = foldername.format(base, block, scale)

                    if not os.path.exists(featurefolder):
                        os.mkdir(featurefolder)

                    name = featurename.format(base, 'boxplot', block, scale, i)
                    boxplot(groundtruth, feature_, featurefolder, name)


def analysis(imagefile, blocks, scales, bands, features):
    create_features(imagefile, blocks, scales, bands, features)
    analyze_features(imagefile, blocks, scales, bands, features)


def boxplot(groundtruth, feature, folder, name):
    dataset = create_dataset(feature, groundtruth)
    seaborn.boxplot(x="formality", y="feature", data=dataset)
    plt.title(name)
    plt.savefig(os.path.join(folder, name))
    plt.clf()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LSR features")
    parser.add_argument("imagefile",  nargs="?",
                        help="The imagefile to use")
    args = parser.parse_args()
    analysis(args.imagefile, [20, 40], [50, 100], [1, 2, 3], ['hog', 'lsr'])