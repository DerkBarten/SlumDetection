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

def canny():
    filename = "data/Bangalore.TIF"
    #filename = "data/section_1_fixed.tif"
    image = np.array(read_geotiff(filename))
    r = image[4]
    g = image[2]
    b = image[1]

    image = np.uint8(255*((0.21 * r + 0.72 * g + 0.07 * b) / 2047))
    edges = []
    edges = cv2.Canny(image, 100, 200)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    #cv2.imwrite('hough.jpg',image)

    #plt.imshow(image, cmap='gray')
    #plt.show()
    plt.imshow(edges, cmap='gray')
    plt.show()

def opencv():
    #reading image directly from current working directly
    build_image = cv2.imread("data/roads/section_8.png")

    #build_image = cv2.bilateralFilter(build_image, 9, 75, 75)
    #Doing MeanShift Filtering
    #shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    #GrayScale Conversion
    build_gray = cv2.cvtColor(build_image, cv2.COLOR_BGR2GRAY)

    

    #OTSU Thresholding
    thresh = cv2.threshold(build_gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = np.invert(thresh)

    cv2.imwrite("plots/roadmask.jpg", thresh)

    minLineLength = 100
    maxLineGap = 2
    lines = cv2.HoughLinesP(thresh, 1, 0.2, 100, 0, minLineLength, maxLineGap)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(build_image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('plots/hough.jpg', build_image)
    #plt.imshow(build_image)
    #plt.show()

    #cv2.imshow("OTSU Thresholded Image", thresh)
    # cv2.imwrite("outimg.jpg", thresh)        

    # #Checking Coordinates of White Pixels
    # build_white =  np.argwhere(thresh == 255)

    # #Creating an array
    # build_data= np.array(build_white)
    # np.savetxt('build_coords.csv', build_data,delimiter=",")


    # #Checking Coordinates of White Pixels
    # road_white =  np.argwhere(thresh == 0)

    # #Creating an array
    # road_data = np.array(road_white)

    # print(road_data)

    # #Saving vector of roads in a csv file
    # np.savetxt('road_coords.csv', road_data,delimiter=",")

#opencv()
#canny()


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
    plotax.imshow(build_image)
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


    plotax.imshow(build_image)
    plotax.scatter(peaks[:,1], peaks[:,0], c='r')
    min_distance.on_changed(update)
    road_width.on_changed(update_roadwidth)
    plt.show()

    # filtered_output = maximum_filter(output, size=(30,30))
    # print(filtered_output)
    # plt.imshow(filtered_output)
    # plt.show()
    # output[output < 1e4] = 0
    #plt.imshow(output)
    #plt.show()
        
