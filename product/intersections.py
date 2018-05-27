import sys
import cv2
import numpy as np
import math

from enum import Enum
from scipy.ndimage import zoom
from groundtruth import reshape_image
from skimage.feature import peak_local_max
from pysal.esda.getisord import G_Local
from pysal.esda.moran import Moran_Local
from pysal.weights.Distance import DistanceBand
from util import read_geotiff, read_image, convert_to_grayscale

from scipy import signal as sg
from scipy import ndimage as nd
from matplotlib import pyplot as plt


class ktype(Enum):
    """
    This enum contains the different versions of the convolution kernels.

    """
    ORIGINAL = 1
    GAUSSIAN = 2
    INCREASE = 3
    NEGATIVE = 4


class Kernel:
    """
    This class produces a kernel that can be used for the detection of road
    intersections.

    """
    def __init__(self, road_width=30, road_length=70,
                 kernel_type=ktype.GAUSSIAN):
        self.road_width = road_width
        self.road_length = road_length
        self.kernel_type = kernel_type
        self.kernel = self.create()

    def get_kernel(self):
        return self.kernel

    def create(self):
        """
        This function is the parent function in the creation of convolution
        kernels. The kernel contains the form of cross to represent the form
        of an road intersection as seen from satellite images.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        if self.kernel_type == ktype.ORIGINAL:
            return self.create_original_kernel()
        if self.kernel_type == ktype.INCREASE:
            return self.create_increase_kernel()
        if self.kernel_type == ktype.NEGATIVE:
            return self.create_negative_kernel()
        if self.kernel_type == ktype.GAUSSIAN:
            return self.create_gaussian_kernel()
        print("ERROR: Invalid kernel specified")
        exit()

    def create_original_kernel(self):
        """
        This function creates a type of kernel that was used as a proof of concept.
        The content of the kernel is a cross of ones with the remainder of the 
        kernel filled with zeros.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        # horizontal road
        hr = np.ones((self.road_width, self.road_length))
        # vertical road
        vr = np.ones((self.road_length, self.road_width))
        # road center
        cr = np.ones((self.road_width, self.road_width))
        # roadside
        rs = np.zeros((self.road_length, self.road_length))

        r1 = np.concatenate((rs, vr, rs), axis=1)
        r2 = np.concatenate((hr, cr, hr), axis=1)
        return np.concatenate((r1, r2, r1), axis=0)

    def create_increase_kernel(self):
        """
        Creates a kernel where the ends of the intersection count the most. 

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        hr1 = np.tile(np.arange(self.road_length, 0, -1), (self.road_width, 1))
        hr2 = np.flip(hr1, axis=1)
        vr1 = np.transpose(hr1)
        vr2 = np.flip(vr1, axis=0)
        cr = np.ones((self.road_width, self.road_width))
        rs = np.zeros((self.road_length, self.road_length))

        max_val = 5
        r1 = np.concatenate((rs, vr1, rs), axis=1)
        r2 = np.concatenate((hr1, cr, hr2), axis=1)
        r3 = np.concatenate((rs, vr2, rs), axis=1)
        kernel = np.concatenate((r1, r2, r3), axis=0)
        kernel[kernel > max_val] = max_val
        return kernel

    def create_negative_kernel(self):
        """
        Creates a kernel where the area outside the cross is negative.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        # horizontal road; all values are two
        hr = np.ones((self.road_width, self.road_length)) * 2
        # vertical road; all values are two
        vr = np.ones((self.road_length, self.road_width)) * 2
        # road center; all values are two
        cr = np.ones((self.road_width, self.road_width)) * 2

        min_val = -1
        # Create a staircase down from the cross to negative numbers. min_val is
        # lower bound of the negative numbers
        rs1 = np.stack([self.calculate_row_negative_kernel(i, min_val)
                        for i in range(1, self.road_length + 1)])
        rs2 = np.flip(rs1, axis=1)
        rs3 = np.flip(rs1, axis=0)
        rs4 = np.flip(rs2, axis=0)

        r1 = np.concatenate((rs4, vr, rs3), axis=1)
        r2 = np.concatenate((hr, cr, hr), axis=1)
        r3 = np.concatenate((rs2, vr, rs1), axis=1)

        kernel = np.concatenate((r1, r2, r3), axis=0)
        kernel[kernel < min_val] = min_val
        return kernel

    def calculate_row_negative_kernel(self, i, min_val):
        """
        A helper function for the negative kernel.
        """
        return np.concatenate((np.arange(-1,  i * -1, -1),
                               np.full(self.road_length - i + 1, i * -1)))

    def create_gaussian_kernel(self):
        """
        Creates a kernel where the cross of the kernel is built using two Gaussian
        distributions. The use of this distribution should create smoother results
        than the other kernels.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        kernel_width = self.road_length * 2 + self.road_width
        g1 = sg.gaussian(kernel_width, std=self.road_width / 2)
        g2 = sg.gaussian(kernel_width, std=self.road_length)

        r1 = np.tile(g1, (kernel_width, 1))
        r2 = np.transpose(r1)

        kernel = np.maximum(r1, r2)
        return kernel

    def rotate_kernel(self, kernel, degrees):
        return nd.rotate(kernel, degrees)


class RoadIntersection:
    """
    This class detects road intersection in images.

    """
    def __init__(self, image_path, road_width=30, road_length=70, 
                 peak_min_distance=150, kernel_type=ktype.GAUSSIAN):
        self.road_width = road_width
        self.road_length = road_length
        self.peak_min_distance = peak_min_distance
        self.kernel_type = kernel_type
        self.image_path = image_path
        self.intersections = self.calculate()

    def get_intersections(self):
        return self.intersections

    def visualize(self):
        print("Visualizing...")
        visualize_convolution(convolution, image, (peaks, relocated))

    def calculate(self):
        """
        This function uses convolution as a method for finding road
        intersections in an image. It gets called automatically on the creation
        of the object.


        """
        kernel = Kernel(self.road_width, self.road_length,
                        self.kernel_type).get_kernel()
        image = read_image(self.image_path)
        gray_image = convert_to_grayscale(image)
        gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY |
                                   cv2.THRESH_OTSU)[1]
        convolution = sg.convolve(gray_image, kernel, "valid")
        peaks = peak_local_max(convolution,
                               min_distance=self.peak_min_distance)

        return self.relocate_peaks(peaks, kernel.shape[0])

    def relocate_peaks(self, peaks, kernel_width):
        """
        This function relocates the peaks by the half of the kernel width. During
        the convolution, the kernel translates the image by the half of the kernel
        width. This relocation is necessary to move the peaks back to the right
        positions.

        """
        return peaks + kernel_width / 2

    def visualize(self):
        image = read_image(self.image_path)
        
        plt.imshow(image)
        plt.scatter(self.intersections[:, 1], self.intersections[:, 0], c='r',
                    alpha=0.5)
        plt.axis('off')
        plt.show()


class RoadIntersectionFeature:
    """
    This class represents the road intersection feature
    """
    def __init__(self, image_path, road_width=30, road_length=70, peak_min_distance=150,
                 kernel_type=ktype.GAUSSIAN, block_size=20, scale=150):
        self.road_width = road_width
        self.road_length = road_length
        self.peak_min_distance = peak_min_distance
        self.image_path = image_path
        self.kernel_type = kernel_type
        self.block_size = block_size
        self.scale = scale
        self.scaled_block_size = self.block_size * 4
        self.feature = self.calculate()

    def get_feature(self):
        """
        This function can be used to get the feature after the creation of the
        object

        """
        return self.feature

    def visualize(self):
        plt.imshow(self.feature)
        plt.show()

    def calculate(self):
        """
        This function calculates the road intersection feature. It gets called
        automatically on the creation of this class.

        """
        intersections = RoadIntersection(self.image_path, self.road_width,
                                         self.road_length,
                                         self.peak_min_distance,
                                         self.kernel_type).get_intersections()
        image = read_image(image_path)
        self.image_shape = image.shape

        density_map = self.create_density_map(intersections)
        radius = int(self.scale / self.block_size)
        feature = self.create_hotspot_map(density_map, radius)
        feature = self.interpolate_feature(feature)
        feature = reshape_image(feature, (image.shape[0], image.shape[1]),
                                self.block_size, self.scale)

        return feature

    def create_density_map(self, points):
        """
        This function rasterizes the intersection points to a grid built from
        blocks of size block_size and in the shape of the image. This is required
        in the creation of a hotspot map from the intersection points.

        Args:
            points:         nx2 numpy array of integers containing the points of
                            road intersection.
        Returns:
            A rasterized version of the intersection points; nxm numpy matrix

        """
        height = self.image_shape[0]
        width = self.image_shape[1]
        density_map = np.zeros((int(math.floor(float(height) /
                                    self.scaled_block_size)),
                                int(math.floor(float(width) /
                                    self.scaled_block_size))))

        for point in points:
            h = int(point[0] / self.scaled_block_size)
            w = int(point[1] / self.scaled_block_size)

            if point[0] < self.image_shape[0] and point[1] < self.image_shape[1]:
                density_map[h, w] += 1
        return density_map

    def create_hotspot_map(self, density_map, radius):
        """
        Create a hotspot map from the intersection density map.

        """
        grid = np.indices((density_map.shape[0], density_map.shape[1]))
        grid = np.stack((grid[0], grid[1]), axis=-1)
        grid = np.reshape(grid, (grid.shape[0]*grid.shape[1], 2))

        w = DistanceBand(grid, threshold=radius)
        y = np.ravel(density_map)

        g = G_Local(y, w).Zs
        return np.reshape(g, (density_map.shape[0], density_map.shape[1]))

    def interpolate_feature(self, feature):
        """
        This function resizes and interpolates the feature matrix to the dimensions
        corresponding to the image with the correct block size. A larger blocksize
        was used to create the feature matrix to reduce the computational load.

        Args:
            feature:        The hotspot map of reduced dimensions; nxm numpy matrix
                            of floats
        Returns:
            A resized and interpolated version of the feature matrix in the correct
            dimensions corresponding to the shape of the image and blocksize.

        """
        feature_shape = feature.shape
        zoom_level = [float(self.image_shape[0]) / (self.block_size *
                      feature_shape[0]),
                      float(self.image_shape[1]) / (self.block_size *
                      feature_shape[1])]

        # To compensate for the round() used in the zoom() when we want to use a
        # ceil() instead. The round() will give one off errors when the computed
        # dimensions of the interpolated feature matrix has the first decimal lower
        # than 0.5.
        if (zoom_level[0] * feature_shape[0]) % 1 < 0.5:
            zoom_level[0] = math.ceil(zoom_level[0] * feature_shape[0]) /\
                            float(feature_shape[0])
        if (zoom_level[1] * feature_shape[1]) % 1 < 0.5:
            zoom_level[1] = math.ceil(zoom_level[1] * feature_shape[1]) /\
                            float(feature_shape[1])

        return zoom(feature, zoom_level, order=3)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please supply an image")
        exit()

    image_path = sys.argv[1]

    # These parameters work well for large scale images
    # params = convolution_parameters(road_width=35, road_length=200,
    #                                 peak_min_distance=100,
    #                                 kernel_type=ktype.GAUSSIAN)

    # These parameters work well for smaller images
    # params = convolution_parameters(road_width=30, road_length=70,
    #                                 peak_min_distance=150,
    #                                 kernel_type=ktype.NEGATIVE)


    # Params for section 1
    feature = RoadIntersectionFeature(image_path, road_width=20,
                                      road_length=70,
                                      peak_min_distance=100,
                                      kernel_type=ktype.GAUSSIAN,
                                      scale=100,
                                      block_size=20).visualize()


    # Params for section 3
    # feature = RoadIntersectionFeature(image_path, road_width=30,
    #                                   road_length=70,
    #                                   peak_min_distance=150,
    #                                   kernel_type=ktype.GAUSSIAN,
    #                                   scale=150,
    #                                   block_size=20).visualize()

    # intersections = RoadIntersection(image_path, road_width=20, road_length=70, 
    #                                  peak_min_distance=100,
    #                                  kernel_type=ktype.GAUSSIAN)
    # intersections.visualize()

  








