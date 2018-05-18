from __future__ import print_function, division
import cv2
import numpy as np
from util import read_geotiff
from matplotlib import pyplot as plt
from PIL import Image
import operator
import pandas as pd

from scipy import signal as sg

from skimage.feature import peak_local_max\

from matplotlib.widgets import Slider


def update(val):
    peaks = peak_local_max(output, min_distance=int(min_distance.val))
    
    plotax.clear()
    plotax.imshow(build_image)
    plotax.scatter(peaks[:, 1], peaks[:, 0], c='r')

def update_roadwidth(val):
    sw = int(road_width.val)
    r1 = np.concatenate((np.zeros((sw, sw)), np.ones((sw, sw)), np.zeros((sw, sw))), axis=1)
    r2 = np.ones((sw, sw * 3))
    kernel = np.concatenate((r1, r2, r1))

    output = sg.convolve(build_gray, kernel, "valid")
    peaks = peak_local_max(output, min_distance=int(min_distance.val))
    
    plotax.clear()
    r = build_image[:, :, 2]
    g = build_image[:, :, 1]
    b = build_image[:, :, 0]
    rgb = np.stack([r, g, b], axis=2)

    plotax.imshow(rgb)
    plotax.scatter(peaks[:, 1], peaks[:, 0], c='r')


if __name__ == '__main__':
    build_image = cv2.imread("data/roads/section_8.png")
    build_gray = cv2.cvtColor(build_image, cv2.COLOR_BGR2GRAY)
    build_gray = 255 - build_gray

    sw = 15
    r1 = np.concatenate((np.zeros((sw, sw)), np.ones((sw, sw)), np.zeros((sw, sw))), axis=1)
    r2 = np.ones((sw, sw * 3))
    kernel = np.concatenate((r1, r2, r1))

    output = sg.convolve(build_gray, kernel, "valid")

    peaks = peak_local_max(output, min_distance=70)

    plotax = plt.axes()
    axmindist = plt.axes([0.25, 0.1, 0.65, 0.03])
    axsw = plt.axes([0.25, 0.05, 0.65, 0.03])    
    min_distance = Slider(axmindist, 'Min Distance', 1, 150, valinit=70, valstep=5)
    road_width = Slider(axsw, 'Road Width', 1, 40, valinit=15, valstep=1)


    r = build_image[:, :, 2]
    g = build_image[:, :, 1]
    b = build_image[:, :, 0]
    rgb = np.stack([r, g, b], axis=2)

    plotax.imshow(rgb)
    plotax.scatter(peaks[:,1], peaks[:,0], c='r')
    min_distance.on_changed(update)
    road_width.on_changed(update_roadwidth)
    plt.show()