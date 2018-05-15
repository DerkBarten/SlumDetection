#!/usr/bin/python
from satsense.image import SatelliteImage
from satsense.bands import WORLDVIEW2
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    filename = sys.argv[1]

    imagefile = filename
    bands = WORLDVIEW2 # The ordering of the bands in a worldview2 file
    
    img = SatelliteImage()
    img.load_from_file(imagefile, bands)

    # Load the file. This will give the raw gdal file as well as a numpy
    # ndarray with the bands loaded (not normalized)
    #dataset, image = load_from_file(imagefile)

    # Convert the image to an rgb image. The original image is not
    # yet normalized
    true_color = get_rgb_bands(img)

    plt.imshow(true_color)
