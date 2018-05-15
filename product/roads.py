from __future__ import print_function, division
import cv2
import numpy as np
from util import read_geotiff
from matplotlib import pyplot as plt
from PIL import Image
import operator
import pandas as pd


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
    build_image = cv2.imread("data/image-input/section_8.png")


    #Doing MeanShift Filtering
    #shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    #GrayScale Conversion
    build_gray = cv2.cvtColor(build_image, cv2.COLOR_BGR2GRAY)

    build_gray = cv2.bilateralFilter(build_gray, 6, 75, 75)

    #OTSU Thresholding
    thresh = cv2.threshold(build_gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = np.invert(thresh)

    cv2.imwrite("outimg.jpg", thresh)

    minLineLength = 150
    maxLineGap = 2
    lines = cv2.HoughLinesP(thresh, 1, np.pi/300, 400, 0, minLineLength, maxLineGap)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(build_image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('hough.jpg',build_image)


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

opencv()
#canny()