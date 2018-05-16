import argparse
import gdal
from matplotlib import pyplot as plt
from satsense.bands import WORLDVIEW2
from satsense.image import SatelliteImage, get_rgb_bands
from scipy import ndimage, misc
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sobel Filter')
    parser.add_argument('filename',
                    help='The satellite TIF image to analyze')
    args = parser.parse_args()
    
    bands = WORLDVIEW2
    dataset = gdal.Open(args.filename, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()

    if len(array.shape) == 3:
            # The bands column is in the first position, but we want it last
        array = np.rollaxis(array, 0, 3)
    elif len(array.shape) == 2:
            # This image seems to have one band, so we add an axis for ease
            # of use in the rest of the library
        array = array[:, :, np.newaxis]

    image = array.astype('float32') 
    rgb = get_rgb_bands(image, bands)
    print(rgb.shape)
    #result = ndimage.sobel(rgb)
    plt.imshow(image)
    plt.show()
