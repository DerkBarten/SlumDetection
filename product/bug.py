import numpy as np
import math
import gdal
from matplotlib import pyplot as plt


def read_geotiff(filename):
    dataset = gdal.Open(filename)
    return dataset.ReadAsArray()


def print_dimensions(imagefile, featurefile, block, scale):
    image = read_geotiff(imagefile)
    feature = read_geotiff(featurefile)

    print("Image:\t\t\t\t\t{}".format(imagefile))
    print("Feature:\t\t\t\t{}".format(featurefile))
    print("Dimensions of the image:\t\t{}".format(image.shape))
    print("Dimensions of a single block:\t\t{}".format(block))
    print("Old Expected feature vector dimension:\t{}x{}".format(
        image.shape[1] / block, image.shape[2] / block))

    x = math.ceil(float(image.shape[1] - (scale - block)) / block)
    y = math.ceil(float(image.shape[2] - (scale - block)) / block)
    print("New Expected feature vector dimension:\t{}x{}".format(x, y))

    print("Actual feature vector dimension:\t{}x{}".format(
        feature.shape[1], feature.shape[2]))
    print("")

# De feature file is de samenvoeging van alle losse files die spfeas geeft
# Dit bestand is gemaakt door het commando'gdalwarp * merged.tif' in de folder
# met alle losse bestanden.
# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK20_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 20)

imagefile = 'data/Bangalore.TIF'
featurefile = 'features/Bangalore__BD5-3-2_BK24_SC200_TRhog.vrt'
print_dimensions(imagefile, featurefile, 24, 200)


imagefile = 'data/Bangalore.TIF'
featurefile = 'features/Bangalore__BD5-3-2_BK24_SC100_TRhog.vrt'
print_dimensions(imagefile, featurefile, 24, 100)

imagefile = 'data/Bangalore.TIF'
featurefile = 'features/Bangalore__BD5-3-2_BK24_SC35_TRhog.vrt'
print_dimensions(imagefile, featurefile, 24, 35)

# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK24_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 24)