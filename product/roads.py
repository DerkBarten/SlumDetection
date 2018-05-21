# from __future__ import print_function, division
import sys
import cv2
import numpy as np
from util import read_geotiff
from matplotlib import pyplot as plt
from scipy import signal as sg
from scipy import ndimage as nd
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

# Worked well for section_8:
# road_width = 15
# road_length = 45
# peak_min_distance = 50

# PARAMS kernel 3
# road_width = 30
# road_length = 70
# peak_min_distance = 80

# works well with kernel 4 sec 8
# road_width = 15
# road_length = 50
# peak_min_distance = 50
# also
# road_width = 15
#     road_length = 40
#     peak_min_distance = 35
# works really well:
# road_width = 15
# road_length = 35
# peak_min_distance = 35

# works well for large roads
# road_width = 30
#     road_length = 300
#     peak_min_distance = 80
# road_width = 35
#     road_length = 200
#     peak_min_distance = 100

def rotate_kernel(kernel, degrees):
    return nd.rotate(kernel, degrees)

def convolution_method(image_path):
    road_width = 35
    road_length = 200
    peak_min_distance = 100

    
    np.set_printoptions(threshold=np.nan)
    # kernel = create_intersection_kernel(road_width, road_length)
    # kernel = create_alternative_kernel(road_width, road_length)
    # kernel = create_kernel_3(road_width, road_length)
    kernel = create_kernel_4(road_width, road_length)
    
    #kernel = rotate_kernel(kernel, 45)
    # plt.imshow(kernel, cmap='gray')
    # plt.imsave('kernel.png', kernel, cmap='gray')
    # plt.show()
    # exit()

    print("Read image")
    image = read_image(image_path)
    gray_image = convert_to_grayscale(image)
    
    print("Performing Convolution")
    convolution = convolve(gray_image, kernel)
    # normalize
    
    print("Finding peaks")
    peaks = peak_local_max(convolution, min_distance=peak_min_distance)
    peaks = threshold_peaks(convolution, peaks, 0)

    relocated = relocate_peaks(peaks, kernel.shape[0])
    print("Visualizing")
    visualize_convolution(convolution, image, (peaks, relocated))


    norm_convolution = convolution / np.max(convolution)


def threshold_peaks(convolution, peaks, threshold):
    d = np.array([])
    for i, peak in enumerate(peaks):
        if convolution[peak[0], peak[1]] < threshold:
            d = np.append(d, i)
    
    # for index in d:
    #     print(peaks[int(index), :]) 
    return np.delete(peaks, d, axis=0)

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


def create_alternative_kernel(road_width=15, road_length=15):
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

def create_kernel_3(road_width=15, road_length=15):
    # horizontal road
    hr = np.ones((road_width, road_length)) * 2
    # vertical road
    vr = np.ones((road_length, road_width)) * 2
    # road center
    cr = np.ones((road_width, road_width)) * 10

    min_val = -2
    rs1 = np.stack([calc_row_kernel_3(i, road_length, min_val)
                    for i in range(1, road_length + 1)]) 
    rs2 = np.flip(rs1, axis=1)
    rs3 = np.flip(rs1, axis=0)
    rs4 = np.flip(rs2, axis=0)

    r1 = np.concatenate((rs4, vr, rs3), axis=1)
    r2 = np.concatenate((hr, cr, hr), axis=1)
    r3 = np.concatenate((rs2, vr, rs1), axis=1)

    kernel = np.concatenate((r1, r2, r3), axis=0)
    kernel[kernel < -1] = min_val
    #kernel[kernel > 0] = 2
    return kernel

# Gaussian
def create_kernel_4(road_width=15, road_length=15):
    kernel_width = road_length * 2 + road_width
    g1 = sg.gaussian(kernel_width, std=road_width / 2)
    g2 = sg.gaussian(kernel_width, std=road_length)

    r1 = np.tile(g1, (kernel_width, 1))
    #for i in range(0, kernel_width):
    #    r1[:, i] = r1[:, i] * g2

    r2 = np.transpose(r1)
    kernel = np.maximum(r1, r2)



    #plt.plot(np.arange(0, kernel_width, 1), rw)
    #plt.show()
    # plt.imshow(kernel)
    # plt.show()
    return kernel

def calc_row_kernel_3(i, road_length, min_val):
    return np.concatenate((np.arange(-1,  i * -1, -1),
                           np.full(road_length - i + 1, i * -1)))

def convolve(image, kernel):
    return sg.convolve(image, kernel, "valid")


def convert_to_grayscale(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # Invert colors to find intersections as peaks instead of valleys
    return 255 - gray_image


def read_image(image_path):
    # image = read_geotiff(image_path)

    # r = image[:, :, 5]
    # g = image[:, :, 3]
    # b = image[:, :, 2]
    # rgb = np.stack([r, g, b], axis=2)
    # plt.imshow(rgb)
    # plt.show()
    # exit()

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

        # plt.subplot(1, 3, 3)
        # plt.imshow(convert_to_grayscale(image), cmap='gray')
        # plt.axis('off')
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
