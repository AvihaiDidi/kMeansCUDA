# Fast Approximate K means for image processing using Nvidia's numba library
# This main cause of slowness is the normal GPU implementation of kmeans is the conditional statements, so in this version I completly got rid of those
# Instead, I use euclidian distance between pixels as a measure of similarity (essentially, two identical pixels are assigned a value of 1, and two pixels that are as far apart as possible are assigned a value of 0
# Naturally, that means this algorithem isn't really K-means anymore, but it's close enough

import numpy as np
from numba import jit, cuda
from math import sqrt

import sysAndConsts as sac

# Internal functions are prefixed with _ and arn't supposed to be used directly (you can use them anyways if you want to tho)

# finds the K means and returns them, does {iterations} iterations
# img is the zeroth elemenet of the array _loadImage(path) returns
def getCentroids(iterations, k, img, printprog = False):
    dim = len(img[0])
    img_device = cuda.to_device(img)
    centroids = sac.initCentroids(k, dim)
    centroids_device = cuda.to_device(centroids)
    for iteration in range(iterations):
        if printprog:
            print(centroids_device.copy_to_host())
        new_centroids = np.zeros((k, dim), dtype = 'float32')
        new_centroids = new_centroids.reshape(k * dim)
        new_centroids_device = cuda.to_device(new_centroids)
        new_centroids_count_device = cuda.to_device(np.zeros((k), dtype = 'float32'))
        _new_centroids_kernel[sac.BLOCKS_PER_GRID, sac.THREADS_PER_BLOCK](img_device, centroids_device, new_centroids_device, new_centroids_count_device)
        cuda.synchronize()
        new_centroids_count = new_centroids_count_device.copy_to_host()
        new_centroids = new_centroids_device.copy_to_host()
        new_centroids = new_centroids.reshape((k, dim))
        centroids = centroids_device.copy_to_host()
        for index in range(k):
            if 0 < new_centroids_count[index]:
                centroids[index] = new_centroids[index] / new_centroids_count[index]
    return centroids

# creates a new image using the original image and the centroids
# imgdata is the output of _loadImage
def newImg(imgdata, centroids):
    dim = len(imgdata[0][0])
    height = imgdata[1]
    width = imgdata[2]
    new_img_device = cuda.device_array_like(np.reshape(imgdata[0],  height * width * dim))
    img_device = cuda.to_device(imgdata[0])
    centroids_device = cuda.to_device(centroids)
    _new_image_kernel[sac.BLOCKS_PER_GRID, sac.THREADS_PER_BLOCK](img_device, new_img_device, centroids_device)
    cuda.synchronize()
    new_img = new_img_device.copy_to_host()
    cuda.synchronize()
    return np.reshape(new_img, (height, width, dim))

# gpu kernel for finding the centroids
# note that new_centroids_device is flattened, this is done to make working with the atomic functions easier
@cuda.jit
def _new_centroids_kernel(img_device, centroids_device, new_centroids_device, new_centroids_count_device):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, img_device.shape[0], stride):
        pixel = img_device[i]
        # inline closestCentroidIndex {
        centroid_index = -1
        min_dist = np.Infinity
        for index, centroid in enumerate(centroids_device):
            #dist = np.linalg.norm(pixel - centroid) {
            dist = 0
            for k in range(len(pixel)):
                dist += (pixel[k] - centroid[k])**2
            # }
            if dist < min_dist:
                min_dist = dist
                centroid_index = index
        # }
        for j in range(len(pixel)):
            cuda.atomic.add(new_centroids_device, j + centroid_index*len(pixel), pixel[j])
        cuda.atomic.add(new_centroids_count_device, centroid_index, np.float32(1.0))

# gpu kernel for generating the new image
# note that new_centroids_device is flattened
@cuda.jit
def _new_image_kernel(img_device, new_img_device, centroids_device):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, img_device.shape[0], stride):
        pixel = img_device[i]
        # inline closestCentroidIndex {
        centroid_index = -1
        min_dist = np.Infinity
        for index, centroid in enumerate(centroids_device):
            #dist = np.linalg.norm(pixel - centroid) {
            dist = 0
            for k in range(len(pixel)):
                dist += (pixel[k] - centroid[k])**2
            # }
            if dist < min_dist:
                min_dist = dist
                centroid_index = index
        # }
        closest = centroids_device[centroid_index]
        for j in range(len(closest)):
            new_img_device[(i * len(closest)) + j] = closest[j]