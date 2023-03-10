# CUDA accelerated k-means for image processing using Nvidia's numba library

import sysAndConsts as sac
import kMeansCPU
import kMeansCUDA

from os.path import isfile, join
from os import listdir
import imghdr


def singleImg(path: str, iterations: int, k: int, printLog: bool = False, dest: str = "", CPU: bool = False) -> int:
    # process a single image located at 'path', returns -1 if couldn't and 0 otherwise
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
    if CPU:
        centroids = kMeansCPU.getCentroids(iterations, k, img_data[0], printprog = False)
    else:
        centroids = kMeansCUDA.getCentroids(iterations, k, img_data[0], printprog = False)
    if printLog:
        print("Done getting centroids, making new image")
    if CPU:
        new_img = kMeansCPU.newImg(img_data, centroids)
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

def multiImg(pathlist: list, iterations: int, k: int, printLog: bool = False, dest: str = "", CPU: bool = False) -> int:
    # process a list of images given in 'pathlist', returns the number of images that were processed succsusfully
    count = 0
    for path in pathlist:
        ret = singleImg(path, iterations, k, printLog, dest, CPU)
        if ret == 0:
            count += 1
    return count

def allInFolder(path: str, iterations: int, k: int, printLog: bool = False, dest: str = "", CPU: bool = False):
    # get a path to a folder and process all images in that folder, save the result in {dest}
    pathlist = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f)) and f.lower().endswith(sac.IMAGE_FORMATS)]
    if printLog:
        print(f"The folder {path} contains {len(pathlist)} images.")
    count = multiImg(pathlist, iterations, k, printLog = False, dest = f"{dest}", CPU = False)
    if printLog:
        print(f"Done processing the folder {path} for k = {k}\n{count}/{len(pathlist)} images were processed succsusfully.")