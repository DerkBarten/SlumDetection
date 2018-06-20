import matplotlib
import os
import seaborn
import logging
import numpy as np
import matplotlib.pyplot as plt

from groundtruth import create_dict
from util import Image
from groundtruth import create_mask, create_groundtruth
from groundtruth import reshape_image, create_dataset

from feature import Feature

LOG = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class Analysis:
    @staticmethod
    def analyze(feature, shapefile):
        image = feature.image
        block_size = feature.block_size
        scales = feature.scales
        feature_names = feature.feature_names
        features = feature.get()

        mask = create_mask(shapefile, image.path)

        groundtruth = create_groundtruth(mask,
                                         block_size=block_size,
                                         threshold=0.6)

        groundtruth = reshape_image(groundtruth, image.shape,
                                    block_size,
                                    max(scales))

        scale_string = Feature.get_scale_string(scales)
        feature_string = Feature.get_feature_string(feature_names)
        foldername = "analysis/{}_{}_BK{}_{}"
        LOG.info("Saving feature analysis ...")
        for i, f in enumerate(features):
            featurename = "{}_{}_{}_BK{}_{}_F{}.png"
            featurefolder = foldername.format(image.filename, feature_string,
                                              block_size, scale_string)

            # Remove infinite values
            f[f == np.inf] = 0

            if not os.path.exists(featurefolder):
                os.mkdir(featurefolder)

            dataset = create_dataset(f, groundtruth)
            name = featurename.format(image.filename, 'boxplot',
                                      feature_string, block_size, scale_string,
                                      i)
            Analysis.boxplot(dataset, featurefolder, name)

            dataset = create_dict(f, groundtruth)
            name = featurename.format(image.filename, 'kde', feature_string,
                                      block_size, scale_string, i)
            Analysis.kde(dataset, featurefolder, name)

            name = featurename.format(image.filename, 'spatial',
                                      feature_string,
                                      block_size, scale_string, i)
            Analysis.spatial_distribution(f, featurefolder, name)

    @staticmethod
    def boxplot(dataset, folder, name):
        LOG.info("Creating boxplot in %s", folder)
        seaborn.boxplot(x="formality", y="feature", data=dataset)
        plt.title(name)
        plt.savefig(os.path.join(folder, name), dpi=200)
        plt.clf()

    @staticmethod
    def kde(dataset, folder, name):
        LOG.info("Creating KDE in %s", folder)
        seaborn.kdeplot(dataset['Formal'], label="Formal")
        seaborn.kdeplot(dataset['Informal'], label="Informal")
        plt.title(name)
        plt.savefig(os.path.join(folder, name), dpi=200)
        plt.clf()

    @staticmethod
    def spatial_distribution(feature, folder, name):
        LOG.info("Creating spatial distribution in %s", folder)
        plt.imshow(feature, cmap="gray")
        plt.title(name)
        plt.savefig(os.path.join(folder, name), dpi=200)
        plt.clf()

if __name__ == "__main__":
    shapefile = 'data/slums_approved.shp'
    feature_names_list = [['rid']]
    block_size_list = [20]
    scales_list = [[50], [150]]
    images = [Image('data/section_1.tif')]
    
    for image in images:
        for feature_names in feature_names_list:
            for block_size in block_size_list:
                for scales in scales_list:
                    feature = Feature(image, block_size=block_size,
                                      scales=scales, bands=[1, 2, 3],
                                      feature_names=feature_names)
                    feature.create()
                    Analysis.analyze(feature, shapefile)
