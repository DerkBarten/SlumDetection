import sys
import gdal
import numpy as np
import glob
import os


def read_geotiff(filename):
    """Yield the bands stored in a geotiff file as numpy arrays."""
    dataset = gdal.Open(filename)
    bands = []
    for band in range(dataset.RasterCount):
        bands.append(dataset.GetRasterBand(band + 1).ReadAsArray())
    return bands


def concat_tiff(directory, width):
    files = glob.glob(os.path.join(directory, "*.tif"))   

    array = None
    row = None
    i = 0
    for file in sorted(files):
        section = np.transpose(np.array(read_geotiff(file)))
        print(section.shape)
        if i < width:
            if row is None:
                row = section
            else:
                row = np.hstack([row, section])
            i += 1
        else:
            if array is None:
                array = section
            else:
                print(array.shape)
                print(section.shape)
                array = np.vstack([array, section])
            i = 0
    print(array.shape)
