import sys
import gdal
import numpy as np


def read_geotiff(filename):
    """Yield the bands stored in a geotiff file as numpy arrays."""
    dataset = gdal.Open(filename)
    bands = []
    for band in range(dataset.RasterCount):
        bands.append(dataset.GetRasterBand(band + 1).ReadAsArray())
    return bands
