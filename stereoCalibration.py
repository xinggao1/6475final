import numpy as np
import cv2
import os
from glob import glob

M = np.array( [[  1.92949648e+03,   0.00000000e+00,   1.20508238e+03],
 [  0.00000000e+00,   1.92774278e+03,   7.79828847e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

dis = np.array([[-0.18403257,  0.04334305, -0.00142467, -0.00183455,  0.32721892]])

def readImages(image_dir):
    """ This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
        image_dir (str): The image directory to get images from.

    Returns:
        images(list): List of images in image_dir. Each image in the list is of
                      type numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]

    return images


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints0 = [] # 2d points in image plane.
imgpoints1 = [] # 2d points in image plane.
imgpoints = [imgpoints0, imgpoints1]
images = readImages("stereoCalibrate2")


i=0
track = 1
cur = 1

for img in images:
    curImgpoints = imgpoints[cur]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
# If found, add object points, image points (after refining them)
    print ret
    if ret == True:
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        curImgpoints.append(corners)
   
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
        cv2.imwrite('steCalimg'+ str(i) + " " + str(cur)+".jpg",img)
        
        i+=1
    track = (track + 1) % 2
    if track == 0:
        cur = 1 - cur
        objpoints.append(objp)
print len(objpoints), len(imgpoints0), len(imgpoints1)
print gray.shape[::-1]
        
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints0, imgpoints1, M, dis, M, dis, gray.shape[::-1])
print retval
print R 
print T 
print E 
print F 

        