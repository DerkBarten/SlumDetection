import numpy as np
import seaborn
from util import read_geotiff, concat_tiff
import matplotlib.pyplot as plt
import argparse
from groundtruth import create_mask, create_groundtruth, create_dataset
import logging
import logging.config
import math


def boxplot(dataset, axis):
    seaborn.boxplot(x="formality", y="feature", data=dataset, ax=axis)


def value_distribution(feature):
    seaborn.kdeplot(np.ravel(feature))
    plt.show()


def spatial_distribution(feature):
    plt.imshow(feature, cmap="hot")
    plt.show()


def analyze(feature):
    value_distribution(feature)
    spatial_distribution(feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LSR features")
    parser.add_argument("imagefile",  nargs="?",
                        help="The imagefile to use")
    parser.add_argument("shapefile",  nargs="?",
                        help="The shapefile to use")
    parser.add_argument("featurefile",  nargs="?",
                        help="The featurefile to use")
    parser.add_argument("--block_size",  nargs="?",
                        type=int, help="The blocksize")

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    features = np.array(read_geotiff(args.featurefile))
    mask = create_mask(args.shapefile, args.imagefile)[4]

    groundtruth = create_groundtruth(mask, block_size=args.block_size,
                                     threshold=0)

    logging.info("Number of features:\t{}".format(features.shape[0]))
    logging.info("Shape of mask:\t{}".format(mask.shape))
    logging.info("Shape of features:\t{}".format(features.shape))
    logging.info("Shape of groundtruth:\t{}".format(groundtruth.shape))

    height = features.shape[0] / 3
    width = features.shape[0] / height

    # Plot the boxplot of each feature
    fig, axes = plt.subplots(int(width), int(height), sharex='col')
    axes = np.ravel(axes)
    for i, feature in enumerate(features):
        dataset = create_dataset(feature, groundtruth)
        axes[i].set_title('feature {}'.format(i + 1))
        boxplot(dataset, axes[i])
    fig.show()

    # Plot the spatial distribution
    fig, axes = plt.subplots(int(width), int(height), sharex='col',)
    axes = np.ravel(axes)
    for i, feature in enumerate(features):
        axes[i].imshow(feature, cmap='gray')
        axes[i].set_title('feature {}'.format(i + 1))
    fig.show()

    # Plot the groundtruth
    fig, ax = plt.subplots()
    ax.imshow(groundtruth, cmap='gray')
    fig.show()

    plt.show()
