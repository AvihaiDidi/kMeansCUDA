# CPU based implementation of k-means. VERY slow. Not recomended for actual use. Use the CUDA one instead
# none of the other code files in this rep even import anything from this file, it's only here for completeness' sake
# this code does work, if you really want to use it

import numpy as np
import imageio
import os.path

import sysAndConsts as sac

def closestCentroidIndex(pixel, centroids):
    # return the centroid index which is closest to pixel
    closest = -1
    min_dist = np.Infinity
    for index, centroid in enumerate(centroids):
        dist = np.linalg.norm(pixel - centroid)
        if dist < min_dist:
            min_dist = dist
            closest = index
    return closest

def getCentroids(iterations: int, k: int, img, printprog: bool = True):
    """
    finds the K means and returns them, does {iterations} iterations
    img is the zeroth elemenet of the array loadImage(path) returns
    """
    DATA_DIMMENSION = len(img[0])
    centroids = sac.initCentroids(k, DATA_DIMMENSION)
    for iteration in range(iterations):
        if printprog:
            print(centroids)
        new_centroids = np.zeros((k, DATA_DIMMENSION))
        new_centroids_count = np.zeros((k, 1))
        # count the number of pixels each centroid has
        for pixel in img:
            centroid_index = closestCentroidIndex(pixel, centroids)
            new_centroids[centroid_index] += pixel
            new_centroids_count[centroid_index] += 1
        # update the centroids
        for index in range(k):
            if 0 < new_centroids_count[index]:
                centroids[index] = new_centroids[index] / new_centroids_count[index]
    return centroids

def newImg(imgdata, centroids):
    # replace each pixel in img with the closest centroid, imgdata is what loadImage returns
    new_img = []
    height = imgdata[1]
    width = imgdata[2]
    for y in range(height):
        new_img.append([])
        for x in range(width):
            pixel = imgdata[0][x + (y * width)]
            closest = centroids[closestCentroidIndex(pixel, centroids)]
            new_img[y].append(closest)
    return new_img