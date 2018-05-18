# from __future__ import print_function, division
import sys
import cv2
import numpy as np
from util import read_geotiff
from matplotlib import pyplot as plt
from scipy import signal as sg
from skimage.feature import peak_local_max


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


def convolution_method(image_path):
    road_width = 15
    road_length = 45
    peak_min_distance = 50

    kernel = create_intersection_kernel(road_width, road_length)
    image = read_image(image_path)
    gray_image = convert_to_grayscale(image)
    convolution = convolve(gray_image, kernel)
    peaks = peak_local_max(convolution, min_distance=peak_min_distance)
    relocated = relocate_peaks(peaks, kernel.shape[0])
    visualize_convolution(convolution, image, (peaks, relocated))


def create_intersection_kernel(road_width=15, road_length=15):
    # horizontal road
    hr = np.ones((road_width, road_length))
    # vertical road
    vr = np.ones((road_length, road_width))
    # road center
    cr = np.ones((road_width, road_width))
    # roadside
    rs = np.zeros((road_length, road_length))
    # row 1
    r1 = np.concatenate((rs, vr, rs), axis=1)
    # row 2
    r2 = np.concatenate((hr, cr, hr), axis=1)
    return np.concatenate((r1, r2, r1), axis=0)


def convolve(image, kernel):
    return sg.convolve(image, kernel, "valid")


def convert_to_grayscale(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # Invert colors to find intersections as peaks instead of valleys
    return 255 - gray_image


def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def visualize_convolution(convolution, image, peaks=None):
    if peaks is not None:
        original, relocated = peaks
        
        plt.imshow(convolution)
        plt.scatter(original[:, 1], original[:, 0], c='r', alpha=0.5)
        plt.axis('off')
        plt.savefig('1.png',bbox_inches='tight')
        plt.show()

        plt.imshow(image)
        plt.scatter(relocated[:, 1], relocated[:, 0], c='r', alpha=0.5)
        plt.axis('off')
        plt.savefig('2.png',bbox_inches='tight')
        plt.show()

def relocate_peaks(peaks, kernel_width):
    return peaks + kernel_width / 2

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please supply an image")
        exit()

    image_path = sys.argv[1]
    convolution_method(image_path)
    #hough_method(image_path)
