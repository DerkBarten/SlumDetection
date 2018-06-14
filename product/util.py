import os
import sys
import cv2
import glob
import gdal
import numpy as np


class Image:
    def __init__(self, image_path):
        self._image_path = image_path
        self._image = self.__read_image(self._image_path)

    
    def __read_image(self, image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @property
    def RGB(self):
        return self._image

    @property
    def grayscale(self):
        gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        # Invert colors to find intersections as peaks instead of valleys
        return 255 - gray_image

    @property
    def shape(self):
        return self._image.shape

    @property
    def filename(self):
        return os.path.basename(os.path.splitext(self._image_path)[0])

    @property
    def path(self):
        return self._image_path


def read_geotiff(filename):
    """Yield the bands stored in a geotiff file as numpy arrays."""
    dataset = gdal.Open(filename)
    bands = []
    for band in range(dataset.RasterCount):
        bands.append(dataset.GetRasterBand(band + 1).ReadAsArray())
    return bands
