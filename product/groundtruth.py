import fiona
import rasterio
import argparse
from rasterio import mask
from rasterio.features import shapes
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def create_groundtruth(mask, block_size=25, threshold=0.1):
    minum_pixels = pow(block_size, 2) * threshold
    slum_cnt = 0
    nonslum_cnt = 0
    height = mask.shape[0]
    width = mask.shape[1]
    groundtruth = np.zeros((int(math.ceil(height / block_size)),
                            int(math.ceil(width / block_size))))

    i = 0
    j = 0
    while i < groundtruth.shape[0]:
        j = 0
        while j < groundtruth.shape[1]:
            chunck = mask[i * block_size:i * block_size + block_size,
                          j * block_size:j * block_size + block_size]
            if np.count_nonzero(chunck) > minum_pixels:
                slum_cnt += 1
                groundtruth[i, j] = 1
            else:
                nonslum_cnt += 1
                groundtruth[i, j] = 0
            j += 1
        i += 1

    return groundtruth


def create_mask(shapefile, imagefile, maskname=None):
    with fiona.open(shapefile, "r") as sf:
        geoms = [feature["geometry"] for feature in sf]

    with rasterio.open(imagefile) as src:
        out_image, out_transform = mask.mask(src, geoms, crop=False, invert=False)
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
    dataset = {'feature': [], 'formality': []}

    for i in range(height):
        for j in range(width):
            if (groundtruth[i, j] != 0):
                dataset['formality'].append('informal')
            else:
                dataset['formality'].append('formal')
            dataset['feature'].append(feature[i, j])

    return pd.DataFrame.from_dict(dataset)

def create_dict(feature, groundtruth):
    height, width = groundtruth.shape
    dataset = {'formal': [], 'informal': []}

    for i in range(height):
        for j in range(width):
            if (groundtruth[i, j] != 0):
                dataset['formal'].append(feature[i, j])
            else:
                dataset['informal'].append(feature[i, j])
            
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask from shape file")
    parser.add_argument("shapefile",  nargs="?",
                        help="The shapefile to use")
    parser.add_argument("imagefile",  nargs="?",
                        help="The imagefile to use")
    parser.add_argument("maskname", nargs="?",
                        help="Select name of the file where to save the mask.")

    args = parser.parse_args()

    # use the red band
    mask = create_mask(args.shapefile, args.imagefile, args.maskname)[0]
    groundtruth = create_groundtruth(mask)
    print(groundtruth.shape)

    plt.imshow(groundtruth, cmap='gray')
    plt.title('Binary mask')    
    plt.show()
