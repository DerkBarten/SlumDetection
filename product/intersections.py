import sys
import cv2
import numpy as np

from enum import Enum
from util import read_geotiff
from skimage.feature import peak_local_max

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


class convolution_parameters():
    """
    This class contains the parameters used for convolution and peak detection.

    """
    def __init__(self, road_width=35, road_length=200, peak_min_distance=100,
                 kernel_type=ktype.GAUSSIAN):
        self.road_width = road_width
        self.road_length = road_length
        self.peak_min_distance = peak_min_distance
        self.kernel_type = kernel_type


def hough_method(image_path):
    image = read_image(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's threshold
    mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY |
                         cv2.THRESH_OTSU)[1]
    # Prefer the roads as white to find intersections as peaks instead of
    # valleys
    mask = np.invert(mask)

    rho = 1
    theta = 0.2
    threshold = 100
    minLineLength = 100
    maxLineGap = 2
    lines = cv2.HoughLinesP(mask, rho, theta, threshold, 0, minLineLength, maxLineGap)

    # Draw the hough lines on top of the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    visualize_hough(image, mask)


def visualize_hough(image, mask, save=False):
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')

    if save:
        plt.imsave('image.png', image)
        plt.imsave('mask.png', mask, cmap='gray')

    plt.show()


def rotate_kernel(kernel, degrees):
    return nd.rotate(kernel, degrees)


def relocate_peaks(peaks, kernel_width):
    """
    This function relocates the peaks by the half of the kernel width. During
    the convolution, the kernel translates the image by the half of the kernel
    width. This relocation is necessary to move the peaks back to the right
    positions.

    """
    return peaks + kernel_width / 2


def threshold_peaks(convolution, peaks, threshold):
    """
    This function removes the peaks which fall below a certain static
    threshold.

    """
    d = np.array([])
    for i, peak in enumerate(peaks):
        if convolution[peak[0], peak[1]] < threshold:
            d = np.append(d, i)

    return np.delete(peaks, d, axis=0)


def convolution_method(image_path, params):
    """
    This function uses convolution as a method for finding road intersections
    in an image.

    Args:
        image_path:     The path to the image to extract the feature from;
                        string
        params:         A convolution_parameters object specifying the
                        parameters in the convolution and detection of the
                        intersections

    """
    road_width = params.road_width
    road_length = params.road_length
    peak_min_distance = params.peak_min_distance
    kernel_type = params.kernel_type
    kernel = create_kernel(road_width, road_length, kernel_type)

    print("Reading image...")
    image = read_image(image_path)
    gray_image = convert_to_grayscale(image)
    gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY |
                               cv2.THRESH_OTSU)[1]

    print("Performing Convolution...")
    convolution = sg.convolve(gray_image, kernel, "valid")

    print("Finding peaks..")
    peaks = peak_local_max(convolution, min_distance=peak_min_distance)
    relocated = relocate_peaks(peaks, kernel.shape[0])

    print("Visualizing...")
    visualize_convolution(convolution, image, (peaks, relocated))


def create_kernel(road_width, road_length, kernel_type):
    """
    This function is the parent function in the creation of convolution
    kernels. The kernel contains the form of cross to represent the form
    of an road intersection as seen from satellite images.

    Args:
        road_width:     The width of the road in the kernel specified in
                        pixels; integer
        road_length:    The length of the road in the kernel specified in
                        pixels; integer
        kernel_type:    The variant of kernel used specified in the ktype
                        enum; integer
    Returns:
        A kernel containing the shape of a cross; nxn numpy matrix

    """
    if kernel_type == ktype.ORIGINAL:
        return create_original_kernel(road_width, road_length)
    if kernel_type == ktype.INCREASE:
        return create_increase_kernel(road_width, road_length)
    if kernel_type == ktype.NEGATIVE:
        return create_negative_kernel(road_width, road_length)
    if kernel_type == ktype.GAUSSIAN:
        return create_gaussian_kernel(road_width, road_length)
    print("ERROR: Invalid kernel specified")
    exit()


def create_original_kernel(road_width=15, road_length=15):
    """
    This function creates a type of kernel that was used as a proof of concept.
    The content of the kernel is a cross of ones with the remainder of the 
    kernel filled with zeros.

    Args:
        road_width:     The width of the road in the kernel specified in
                        pixels; integer
        road_length:    The length of the road in the kernel specified in
                        pixels; integer
     Returns:
        A kernel containing the shape of a cross; nxn numpy matrix

    """
    # horizontal road
    hr = np.ones((road_width, road_length))
    # vertical road
    vr = np.ones((road_length, road_width))
    # road center
    cr = np.ones((road_width, road_width))
    # roadside
    rs = np.zeros((road_length, road_length))

    r1 = np.concatenate((rs, vr, rs), axis=1)
    r2 = np.concatenate((hr, cr, hr), axis=1)
    return np.concatenate((r1, r2, r1), axis=0)


def create_increase_kernel(road_width=15, road_length=15):
    """
    Creates a kernel where the ends of the intersection count the most. 

    Args:
        road_width:     The width of the road in the kernel specified in
                        pixels; integer
        road_length:    The length of the road in the kernel specified in
                        pixels; integer
     Returns:
        A kernel containing the shape of a cross; nxn numpy matrix

    """
    hr1 = np.tile(np.arange(road_length, 0, -1), (road_width, 1))
    hr2 = np.flip(hr1, axis=1)
    vr1 = np.transpose(hr1)
    vr2 = np.flip(vr1, axis=0)
    cr = np.ones((road_width, road_width))
    rs = np.zeros((road_length, road_length))

    max_val = 5
    r1 = np.concatenate((rs, vr1, rs), axis=1)
    r2 = np.concatenate((hr1, cr, hr2), axis=1)
    r3 = np.concatenate((rs, vr2, rs), axis=1)
    kernel = np.concatenate((r1, r2, r3), axis=0)
    kernel[kernel > max_val] = max_val
    return kernel


def create_negative_kernel(road_width=15, road_length=15):
    """
    Creates a kernel where the area outside the cross is negative.

    Args:
        road_width:     The width of the road in the kernel specified in
                        pixels; integer
        road_length:    The length of the road in the kernel specified in
                        pixels; integer
     Returns:
        A kernel containing the shape of a cross; nxn numpy matrix

    """
    # horizontal road; all values are two
    hr = np.ones((road_width, road_length)) * 2
    # vertical road; all values are two
    vr = np.ones((road_length, road_width)) * 2
    # road center; all values are 10
    cr = np.ones((road_width, road_width)) * 2

    min_val = -1
    # Create a staircase down from the cross to negative numbers. min_val is
    # lower bound of the negative numbers
    rs1 = np.stack([calculate_row_negative_kernel(i, road_length, min_val)
                    for i in range(1, road_length + 1)])
    rs2 = np.flip(rs1, axis=1)
    rs3 = np.flip(rs1, axis=0)
    rs4 = np.flip(rs2, axis=0)

    r1 = np.concatenate((rs4, vr, rs3), axis=1)
    r2 = np.concatenate((hr, cr, hr), axis=1)
    r3 = np.concatenate((rs2, vr, rs1), axis=1)

    kernel = np.concatenate((r1, r2, r3), axis=0)
    kernel[kernel < min_val] = min_val
    return kernel


def calculate_row_negative_kernel(i, road_length, min_val):
    """
    A helper function for the negative kernel.
    """
    return np.concatenate((np.arange(-1,  i * -1, -1),
                           np.full(road_length - i + 1, i * -1)))


def create_gaussian_kernel(road_width=15, road_length=15):
    """
    Creates a kernel where the cross of the kernel is built using two Gaussian
    distributions. The use of this distribution should create smoother results
    than the other kernels.

    Args:
        road_width:     The width of the road in the kernel specified in
                        pixels; integer
        road_length:    The length of the road in the kernel specified in
                        pixels; integer
    Returns:
        A kernel containing the shape of a cross; nxn numpy matrix

    """
    kernel_width = road_length * 2 + road_width
    g1 = sg.gaussian(kernel_width, std=road_width / 2)
    g2 = sg.gaussian(kernel_width, std=road_length)

    r1 = np.tile(g1, (kernel_width, 1))
    r2 = np.transpose(r1)

    kernel = np.maximum(r1, r2)
    return kernel


def convert_to_grayscale(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # Invert colors to find intersections as peaks instead of valleys
    return 255 - gray_image


def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def visualize_convolution(convolution, image, peaks, save=False):
    original, relocated = peaks

    if save:
        plt.imshow(convolution)
        plt.scatter(original[:, 1], original[:, 0], c='r', alpha=0.5)
        plt.axis('off')
        plt.savefig('1.png', bbox_inches='tight')
        plt.clf()

        plt.imshow(image)
        plt.scatter(relocated[:, 1], relocated[:, 0], c='r', alpha=0.5)
        plt.axis('off')
        plt.savefig('2.png', bbox_inches='tight')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(convolution)
        plt.scatter(original[:, 1], original[:, 0], c='r', alpha=0.5)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.scatter(relocated[:, 1], relocated[:, 0], c='r', alpha=0.5)
        plt.axis('off')

        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please supply an image")
        exit()

    image_path = sys.argv[1]

    # These parameters work well for large scale images
    params = convolution_parameters(road_width=35, road_length=200,
                                    peak_min_distance=100,
                                    kernel_type=ktype.GAUSSIAN)

    # These parameters work well for smaller images
    # params = convolution_parameters(road_width=15, road_length=35,
    #                                 peak_min_distance=35,
    #                                 kernel_type=ktype.GAUSSIAN)

    convolution_method(image_path, params)
