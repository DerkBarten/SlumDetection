import os
import sys
import cv2
import glob
import gdal
import numpy as np


def convert_to_grayscale(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # Invert colors to find intersections as peaks instead of valleys
    return 255 - gray_image


def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_geotiff(filename):
    """Yield the bands stored in a geotiff file as numpy arrays."""
    dataset = gdal.Open(filename)
    bands = []
    for band in range(dataset.RasterCount):
        bands.append(dataset.GetRasterBand(band + 1).ReadAsArray())
    return bands
