# CUDA accelerated k-means for image processing using Nvidia's numba library

import sysAndConsts as sac
import FAkMeansCUDA
import kMeansCUDA

from os.path import isfile, join
from os import listdir
import imghdr

# process a single image located at 'path', returns -1 if couldn't and 0 otherwise
def singleImg(path, iterations, k, printLog = False, dest = "", fast = False):
    if printLog:
        print(f"Attempting to load {path}")
    try:
        img_data = sac.loadImage(path)
    except:
        if printLog:
            print(f"Failed to open {path}, ignoring.")
        return -1
    if printLog:
        print(f"{path} loaded, making centroids now")
    if fast:
        centroids = FKkMeansCUDA.getCentroids(iterations, k, img_data[0], printprog = False)
    else:
        centroids = kMeansCUDA.getCentroids(iterations, k, img_data[0], printprog = False)
    if printLog:
        print("Done getting centroids, making new image")
    if fast:
        new_img = FKkMeansCUDA.newImg(img_data, centroids)
    else:
        new_img = kMeansCUDA.newImg(img_data, centroids)
    if printLog:
        print("New image ready, saving")
    if dest == "":
        sac.saveImage(new_img, f"{path.split('.')[0]} - k = {k}")
        if printLog:
            print(f"New image saved as {path.split('.')[0]} - k = {k}")
    else:
        sac.saveImage(new_img, f"{dest}/{path.split('/')[-1].split('.')[0]} - k = {k}")
        if printLog:
            print(f"New image saved as {dest}/{path.split('/')[-1].split('.')[0]} - k = {k}")
    return 0

# process a list of images given in 'pathlist', returns the number of images that were processed succsusfully
def multiImg(pathlist, iterations, k, printLog = False, dest = "", fast = False):
    count = 0
    for path in pathlist:
        ret = singleImg(path, iterations, k, printLog, dest, fast)
        if ret == 0:
            count += 1
    return count

# get a path to a folder and process all images in that folder, save the result in {dest}
def allInFolder(path, iterations, k, printLog = False, dest = "", fast = False):
    pathlist = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f)) and f.lower().endswith(sac.IMAGE_FORMATS)]
    if printLog:
        print(f"The folder {path} contains {len(pathlist)} images.")
    count = multiImg(pathlist, iterations, k, printLog = False, dest = f"{dest}", fast = False)
    if printLog:
        print(f"Done processing the folder {path} for k = {k}\n{count}/{len(pathlist)} images were processed succsusfully.")