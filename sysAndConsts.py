"""
    This file is just for some simple functions and constants used by both FAkMeansCUDA.py and kMeansCUDA.py
"""

from skimage import img_as_ubyte
from imageio.v2 import imread, imwrite
from numpy import random, float32

# Constant used trough the code
RANDOM_SEED = 1
BLOCKS_PER_GRID = 256
THREADS_PER_BLOCK = 256
IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

# get a k by dim matrix of random numbers that will be used as the initial centroids
def initCentroids(K, dim):
    random.seed(RANDOM_SEED) # this is here to keep behavior consistent when using the same inputs
    return random.rand(K, dim)

# loads and normalize an image
# returns an array whose zeroth element is the flattened image and the next two are its original height and width
def loadImage(path):
    img = imread(path)
    img = img.astype(float32) / 255.
    img_size = img.shape
    return [img.reshape(img_size[0] * img_size[1], img_size[2]), img_size[0], img_size[1]]

# save an image as filename
def saveImage(img, filename):
    imwrite(f"{filename}.png", img_as_ubyte(img))