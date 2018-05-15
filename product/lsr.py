from util import read_geotiff, concat_tiff
from analysis import analyze, boxplot
import matplotlib.pyplot as plt
import numpy as np
import argparse
from groundtruth import create_mask, create_groundtruth, create_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LSR features")
    parser.add_argument("imagefile",  nargs="?",
                        help="The imagefile to use")
    parser.add_argument("shapefile",  nargs="?",
                        help="The shapefile to use")
    parser.add_argument("featurefile",  nargs="?",
                        help="The featurefile to use")
    # parser.add_argument("--blocksize",  nargs="?",
    #                     help="The blocksize")
    
    args = parser.parse_args()
    features = np.array(read_geotiff(args.featurefile))
    mask = create_mask(args.shapefile, args.imagefile, "testmask")[4]
    print(mask.shape)
    groundtruth = create_groundtruth(mask, block_size=24, threshold=0)
    plt.imshow(groundtruth)
    plt.show()
    #exit()
    
    print(groundtruth.shape)
    for feature in features:
        #print(feature.shape)
        dataset = create_dataset(feature, groundtruth)
        boxplot(dataset)
        #plt.imshow(feature, cmap='gray')
        #plt.show()

    #image = np.array(read_geotiff(args.imagefile))