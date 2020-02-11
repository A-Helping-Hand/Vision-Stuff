import sys
import argparse
import cv2
import math
from scipy.spatial import distance
from scipy.stats import variation
import numpy as np
import matplotlib.pyplot as plt

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Path to the image to be processed', default='./images/1.jpg')
    return parser.parse_args(argv)

def findContourCentroid(contour):
    # compute centroid of contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = [cX, cY]

    return centroid

def centroidnp(arr):
    length, dim = arr.shape

    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

def findRadii(contour, centroid):
    rs = []

    # take 360 evenly spaced points from contour
    points = takespread(contour, 360)

    # find radius for every point
    for point in points:
        r = distance.euclidean(point[0], centroid)
        rs.append(r)

    return rs

def takespread(sequence, num):
    # take num evenly spaced elements from sequence
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(math.ceil(i * length / num))]

def main(args):
    print(args.filename)

    border = 100
    kernel = np.ones((3,3),np.uint8)

    image = cv2.imread(args.filename)
    # image = cv2.resize(image, (400,400))

    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect circles in image
    circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # crop to centroid of detected circles and resize to 400*400
    circles_centroid = centroidnp(circles[0,:])
    ymin = int(circles_centroid[1] - circles_centroid[2] - border)
    ymax = int(circles_centroid[1] + circles_centroid[2] + border)
    xmin = int(circles_centroid[0] - circles_centroid[2] - border)
    xmax = int(circles_centroid[0] + circles_centroid[2] + border)
    crop_image = image[ymin:ymax, xmin:xmax]
    crop_image = cv2.resize(crop_image, (400, 400))

    grey_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)

    contours_image = np.copy(crop_image)

    # apply Gaussian blur and Otsu thresholding
    # grey_crop_image = cv2.GaussianBlur(grey_crop_image, (11, 11), 0)
    grey_crop_image = cv2.GaussianBlur(grey_crop_image, (5, 5), 0)
    ret, thresh_image = cv2.threshold(grey_crop_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # use largest contour with no children
    holes = [contours[i] for i in range(len(contours)) if hierarchy[0][i][2] == -1]
    contour = max(holes, key = cv2.contourArea)
    centroid = findContourCentroid(contour)

    # draw contour and label centroid
    cv2.drawContours(contours_image, [contour], -1, BLUE, 2)
    # cv2.drawContours(contours_image, contours, -1, BLUE, 2)
    # cv2.drawContours(contours_image, holes, -1, BLUE, 2)
    cv2.circle(contours_image, (centroid[0], centroid[1]), 7, BLUE, -1)
    cv2.putText(contours_image, "centroid", (centroid[0] - 20, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)

    cv2.imshow('circle', crop_image)
    cv2.imshow('thresh', thresh_image)
    cv2.imshow('contours', contours_image)
    plt.hist(grey_crop_image.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)

    radii = findRadii(contour, centroid)
    CV = variation(radii)
    SD = np.std(radii)
    print("SD = {}\nCV = {}".format(CV, SD))

    x = np.arange(0, len(radii), 1)
    plt.plot(x, radii)
    plt.show()

if __name__ == "__main__":
    args = parseArguments(sys.argv[1:])
    main(args)
