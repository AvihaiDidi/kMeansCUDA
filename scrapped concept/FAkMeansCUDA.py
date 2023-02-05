# Fast Approximate K means for image processing using Nvidia's numba library
# This main cause of slowness is the normal GPU implementation of kmeans is the conditional statements, so in this version I completly got rid of those
# Instead, I use euclidian distance between pixels as a measure of similarity (essentially, two identical pixels are assigned a value of 1, and two pixels that are as far apart as possible are assigned a value of 0
# Naturally, that means this algorithem isn't really K-means anymore, but it's similar

"""
        The equation I chose to use to determine the influence strength is:
        Influence_i = 2/N - D/totaldis
        where N is the number of centroid and D is the distance between the pixel and centroid
        (the sum of the total influence here is one, and inversly proportional to the distance
"""

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
        _new_centroids_kernel[sac.BLOCKS_PER_GRID, sac.THREADS_PER_BLOCK](img_device, centroids_device, new_centroids_device)
        cuda.synchronize()
        new_centroids = new_centroids_device.copy_to_host()
        new_centroids = new_centroids.reshape((k, dim))
        centroids = new_centroids
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
def _new_centroids_kernel(img_device, centroids_device, new_centroids_device):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, img_device.shape[0], stride):
        pixel = img_device[i]
        # first, add up all distances
        totaldis = 0
        for centroid in centroids_device:
            for k in range(len(pixel)):
                totaldis += (pixel[k] - centroid[k])**2
        # then, modify each centroid by an ammount that is inversly proportional to the distance
        for centroid_index, centroid in enumerate(centroids_device):
            dis = 0
            for k in range(len(pixel)):
                dis += (pixel[k] - centroid[k])**2
            Influence_i = 2.0 / len(centroid) - dis / totaldis
            # Normalize for the number of pixels
            Influence_i /= img_device.shape[0]
            for j in range(len(pixel)):
                cuda.atomic.add(new_centroids_device, j + centroid_index*len(pixel), 
                                pixel[j] * Influence_i)

# gpu kernel for generating the new image
# note that new_centroids_device is flattened
@cuda.jit
def _new_image_kernel(img_device, new_img_device, centroids_device):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, img_device.shape[0], stride):
        pixel = img_device[i]
        # first, add up all distances
        totaldis = 0
        for centroid in centroids_device:
            for k in range(len(pixel)):
                totaldis += (pixel[k] - centroid[k])**2
        # then, modify the pixel by an ammount that is inversly proportional to the distance
        for j in range(len(pixel)):
                new_img_device[(i * len(pixel)) + j] = pixel[j]
        for centroid_index, centroid in enumerate(centroids_device):
            dis = 0
            for k in range(len(pixel)):
                dis += (pixel[k] - centroid[k])**2
            Influence_i = 2.0 / len(centroid) - dis / totaldis
            for j in range(len(pixel)):
                new_img_device[(i * len(pixel)) + j] += centroid[j] * Influence_i
        # normalize the influence of the centroids by the number of centroids
        for j in range(len(pixel)):
                new_img_device[(i * len(pixel)) + j] /= len(centroids_device)
