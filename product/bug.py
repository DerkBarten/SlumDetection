import numpy as np
import math
import gdal
from matplotlib import pyplot as plt


def read_geotiff(filename):
    dataset = gdal.Open(filename)
    return dataset.ReadAsArray()

def calculate_padding(image, block, scale):
    padding_x = math.ceil(image.shape[0] / float(block)) - math.ceil(float(image.shape[0] - (scale - block)) / block)
    
    padding_y = math.ceil(image.shape[1] / float(block)) - math.ceil(float(image.shape[1] - (scale - block)) / block)

    # padding_x = image.shape[0] - math.ceil(float(image.shape[0] - (scale - block)))
    # padding_y = math.ceil(image.shape[1] / float(block)) -\
    #             math.ceil(float(image.shape[1] - (scale - block)) / block)
    return (padding_x, padding_y)

def reshape_image(image, block, scale):
    dimensions = image.shape
    padding = calculate_padding(image, block, scale)
    px = math.ceil((scale - block) / 2.0)

    # x_start = math.ceil(px / block)
    # x_end = padding[0] - x_start

    # y_start = math.ceil(px / block)
    # y_end = padding[1] - y_start

    x_start = math.ceil(padding[0] / 2.0)
    x_end = math.floor(padding[0] / 2.0)
    y_start = math.ceil(padding[1] / 2.0)
    y_end = math.floor(padding[1] / 2.0)

    print(x_start, x_end, y_start, y_end)

    #return image[x_start:-x_end, y_start:-y_end]

def print_dimensions(imagefile, featurefile, block, scale):
    image = read_geotiff(imagefile)
    feature = read_geotiff(featurefile)

    print("Image:\t\t\t\t\t{}".format(imagefile))
    print("Feature:\t\t\t\t{}".format(featurefile))
    print("Dimensions of the image:\t\t{}".format(image.shape))
    print("Dimensions of a single block:\t\t{}".format(block))
    print("Old Expected feature vector dimension:\t{}x{}".format(
        math.ceil(image.shape[1] / float(block)), math.ceil(image.shape[2] / float(block))))

    x = math.ceil(float(image.shape[1] - (scale - block)) / block)
    y = math.ceil(float(image.shape[2] - (scale - block)) / block)
    print("New Expected feature vector dimension:\t{}x{}".format(x, y))

    print("Actual feature vector dimension:\t{}x{}".format(
        feature.shape[1], feature.shape[2]))

    reshape_image(image[0], block, scale)
    
    print("")
    
    # for band in image:
    #     plt.imshow(band)
    #     plt.show()
    
    for band in feature:
        plt.imshow(band)
        plt.show()

# De feature file is de samenvoeging van alle losse files die spfeas geeft
# Dit bestand is gemaakt door het commando'gdalwarp * merged.tif' in de folder
# met alle losse bestanden.

# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK20_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 20)

# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK24_SC200_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 24, 200)


# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK24_SC100_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 24, 100)

# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK24_SC35_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 24, 35)

# imagefile = 'data/new_section_1.tif'
# featurefile = 'features/new_section_1__BD1-2-3_BK20_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 20, 50)

# imagefile = 'data/new_section_1.tif'
# featurefile = 'testfeatures/new_section_1__BD1-2-3_BK20_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 20, 50)


imagefile = 'data/section_1.tif'
#featurefile = '/home/derk/projects/dynaslum/thesis/product/features/section_1__BD1-2-3_BK20_SC30_TRhog.vrt'
featurefile = '/home/derk/projects/dynaslum/thesis/product/features/features/section_1__BD1-2-3_BK20_SC30_TRhog/section_1__BD1-2-3_BK20_SC30__ST1-015__TL000001.tif'
print_dimensions(imagefile, featurefile, 20, 30)

# feature = read_geotiff(featurefile)
# plt.imshow(feature[2])
# plt.show()

# imagefile = 'data/Bangalore.TIF'
# featurefile = 'features/Bangalore__BD5-3-2_BK24_SC50_TRhog.vrt'
# print_dimensions(imagefile, featurefile, 24)