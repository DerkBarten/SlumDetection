
import os
import seaborn
import logging
import numpy as np
import matplotlib.pyplot as plt

from groundtruth import create_dict
from util import read_geotiff
from groundtruth import create_mask, create_groundtruth
from groundtruth import reshape_image, create_dataset

from feature import Feature

LOG = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class Analysis:
    @staticmethod
    def analyze(feature, shapefile):
        image_path = feature.image_path
        block_size = feature.block_size
        scales = feature.scales
        feature_names = feature.feature_names
        features = feature.get()

        image = np.array(read_geotiff(image_path))
        image_shape = image[0].shape
        mask = create_mask(shapefile, image_path)

        groundtruth = create_groundtruth(mask,
                                         block_size=block_size,
                                         threshold=0.6)

        groundtruth = reshape_image(groundtruth, image_shape,
                                    block_size,
                                    max(scales))

        scale_string = Feature.get_scale_string(scales)
        feature_string = Feature.get_feature_string(feature_names)

        base = os.path.basename(os.path.splitext(image_path)[0])
        foldername = "analysis/{}_{}_BK{}_{}"

        for i, f in enumerate(features):
            featurename = "{}_{}_{}_BK{}_{}_F{}.png"
            featurefolder = foldername.format(base, feature_string, block_size,
                                              scale_string)

            # Remove infinite values
            f[f == np.inf] = 0

            if not os.path.exists(featurefolder):
                os.mkdir(featurefolder)

            dataset = create_dataset(f, groundtruth)
            name = featurename.format(base, 'boxplot', feature_string,
                                      block_size, scale_string, i)
            Analysis.boxplot(dataset, featurefolder, name)

            dataset = create_dict(f, groundtruth)
            name = featurename.format(base, 'kde', feature_string,
                                      block_size, scale_string, i)
            Analysis.kde(dataset, featurefolder, name)

            name = featurename.format(base, 'spatial', feature_string,
                                      block_size, scale_string, i)
            Analysis.spatial_distribution(f, featurefolder, name)

    @staticmethod
    def boxplot(dataset, folder, name):
        LOG.info("Creating boxplot in %s", folder)
        seaborn.boxplot(x="formality", y="feature", data=dataset)
        plt.title(name)
        plt.savefig(os.path.join(folder, name))
        plt.clf()

    @staticmethod
    def kde(dataset, folder, name):
        LOG.info("Creating KDE in %s", folder)
        seaborn.kdeplot(dataset['formal'], label="Formal")
        seaborn.kdeplot(dataset['informal'], label="Informal")
        plt.title(name)
        plt.savefig(os.path.join(folder, name))
        plt.clf()

    @staticmethod
    def spatial_distribution(feature, folder, name):
        LOG.info("Creating spatial distribution in %s", folder)
        plt.imshow(feature, cmap="gray")
        plt.title(name)
        plt.savefig(os.path.join(folder, name))
        plt.clf()

if __name__ == "__main__":
    shapefile = 'data/slums_approved.shp'
    feature_names_list = [['rid']]
    block_size_list = [20]
    scales_list = [[50, 100, 150]]

    for feature_names in feature_names_list:
        for block_size in block_size_list:
            for scales in scales_list:
                feature = Feature('data/section_1.tif', block_size=block_size,
                                  scales=scales, bands=[1, 2, 3],
                                  feature_names=feature_names)
                feature.create()
                Analysis.analyze(feature, shapefile)
