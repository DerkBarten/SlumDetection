from satsense.image import SatelliteImage
from satsense.extract import CellGenerator
from satsense.features.hog import HistogramOfGradients
from satsense.bands import WORLDVIEW2, RGB
from satsense.features.feature import FeatureSet
from satsense.extract import extract_features

from groundtruth import create_mask, create_groundtruth, create_dataset

from matplotlib import pyplot as plt
from analysis import analyze, boxplot
import argparse
import pickle
import numpy as np
import seaborn

MEAN_1 = "MEAN_1"
MEAN_2 = "MEAN_2"
DELTA_1 = "DELTA_1"
DELTA_2 = "DELTA_2"
BETA = "BETA"

features = {
    MEAN_1: 0,
    MEAN_2: 1,
    DELTA_1: 2,
    DELTA_2: 3,
    BETA: 4
}

available_bands = {'WORLDVIEW': WORLDVIEW2, 'RGB': RGB}


def create_feature_vector(image, bands):
    generator = CellGenerator(image, (25, 25))
    features = FeatureSet()
    features.add(HistogramOfGradients(windows=((150, 150),)))
    return extract_features(features, generator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HoG features")
    parser.add_argument("filename",  nargs="?",
                        help="The satellite TIF image to analyze")
    parser.add_argument("--load", help="Select the file to load the features\
                        from")
    parser.add_argument("--shapefile", help="Select the file shapefile to use")
    parser.add_argument("--save", help="Select name of the file where to save\
                        the calculated features.")
    parser.add_argument("--bands", choices=available_bands, default=RGB,
                        help="Select the bands of the image.")
    parser.add_argument("--feature", choices=features, default=MEAN_1,
                        help="Select the feature to analyze")

    args = parser.parse_args()
    feature_index = features[args.feature]
    bands = available_bands[args.bands]

    if (args.load):
        f = open(args.load, "rb")
        feature_vector = pickle.load(f)
    elif (args.filename):
        image = SatelliteImage.load_from_file(args.filename, bands)
        feature_vector = create_feature_vector(image, bands)
    else:
        print("ERROR: please select a file or load a featurevector")
        exit()

    feature = feature_vector[:, :, feature_index]

    if args.save:
        save_name = args.save + ".feature"
        f = open(save_name, "wb")
        pickle.dump(feature_vector, f)

    if args.shapefile:
        # Use red band for the mask (arbitrary)
        mask = create_mask(args.shapefile, args.filename)[4]
        groundtruth = create_groundtruth(mask)
        print(groundtruth.shape)
        print(feature.shape)
        dataset = create_dataset(feature, groundtruth)
        #print(dataset)
        print(dataset.head(5))
        boxplot(dataset)
        
    #analyze(feature)
