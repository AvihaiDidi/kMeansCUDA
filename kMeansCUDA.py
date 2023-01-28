# CUDA accelerated k-means for image processing using Nvidia's numba library

import numpy as np
from imageio.v2 import imread, imwrite
from os.path import isfile, join
from os import listdir
from numba import jit, cuda
from math import sqrt
from skimage import img_as_ubyte
import imghdr

# Constant used trough the code
RANDOM_SEED = 1
BLOCKS_PER_GRID = 256
THREADS_PER_BLOCK = 256
IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

# Internal functions are prefixed with _ and arn't supposed to be used directly (you can use them anyways if you want to tho)

# get a k by dim matrix of random numbers that will be used as the initial centroids
def _initCentroids(K, dim):
    np.random.seed(RANDOM_SEED) # this is here to keep behavior consistent when using the same inputs
    return np.random.rand(K, dim)

# loads and normalize an image
# returns an array whose zeroth element is the flattened image and the next two are its original height and width
def _loadImage(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.
    img_size = img.shape
    return [img.reshape(img_size[0] * img_size[1], img_size[2]), img_size[0], img_size[1]]

# save an image as filename
def _saveImage(img, filename):
    imwrite(f"{filename}.png", img_as_ubyte(img))

# finds the K means and returns them, does {iterations} iterations
# img is the zeroth elemenet of the array _loadImage(path) returns
def _getCentroids(iterations, k, img, printprog = False):
    dim = len(img[0])
    img_device = cuda.to_device(img)
    centroids = _initCentroids(k, dim)
    centroids_device = cuda.to_device(centroids)
    for iteration in range(iterations):
        if printprog:
            print(centroids_device.copy_to_host())
        new_centroids = np.zeros((k, dim), dtype = 'float32')
        new_centroids = new_centroids.reshape(k * dim)
        new_centroids_device = cuda.to_device(new_centroids)
        new_centroids_count_device = cuda.to_device(np.zeros((k), dtype = 'float32'))
        _new_centroids_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](img_device, centroids_device, new_centroids_device, new_centroids_count_device)
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
def _newImg(imgdata, centroids):
    dim = len(imgdata[0][0])
    height = imgdata[1]
    width = imgdata[2]
    new_img_device = cuda.device_array_like(np.reshape(imgdata[0],  height * width * dim))
    img_device = cuda.to_device(imgdata[0])
    centroids_device = cuda.to_device(centroids)
    _new_image_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](img_device, new_img_device, centroids_device)
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

# External functions for end user
            
# process a single image, return -1 if couldn't and 0 otherwise
def singleImg(path, iterations, k, printLog = False, dest = ""):
    if printLog:
        print(f"Attempting to load {path}")
    try:
        img_data = _loadImage(path)
    except:
        if printLog:
            print(f"Failed to open {path}, ignoring.")
        return -1
    if printLog:
        print(f"{path} loaded, making centroids now")
    centroids = _getCentroids(iterations, k, img_data[0], printprog = False)
    if printLog:
        print("Done getting centroids, making new image")
    new_img = _newImg(img_data, centroids)
    if printLog:
        print("New image ready, saving")
    if dest == "":
        _saveImage(new_img, f"{path.split('.')[0]} - k = {k}")
        if printLog:
            print(f"New image saved as {path.split('.')[0]} - k = {k}")
    else:
        _saveImage(new_img, f"{dest}/{path.split('/')[-1].split('.')[0]} - k = {k}")
        if printLog:
            print(f"New image saved as {dest}/{path.split('/')[-1].split('.')[0]} - k = {k}")
    return 0

# process a list of images, returns the number of images that were processed succsusfully
def multiImg(pathlist, iterations, k, printLog = False, dest = ""):
    count = 0
    for path in pathlist:
        ret = singleImg(path, iterations, k, printLog, dest)
        if ret == 0:
            count += 1
    return count

# get a path to a folder and process all images in that folder, save the result in {dest}
def allInFolder(path, iterations, k, printLog = False, dest = ""):
    pathlist = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f)) and f.lower().endswith(IMAGE_FORMATS)]
    if printLog:
        print(f"The folder {path} contains {len(pathlist)} images.")
    count = multiImg(pathlist, iterations, k, printLog = False, dest = f"{dest}")
    if printLog:
        print(f"Done processing the folder {path} for k = {k}\n{count}/{len(pathlist)} images were processed succsusfully.")