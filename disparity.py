import numpy as np

import cv2


def blend2(img1, img2):
    img = None
    img = (img1/2 + img2/2)
    #print img[10:20, 1050:1060]
    return img


imgL = cv2.pyrDown( cv2.imread('img1Cal.png') )  # downscale images for faster processing
imgR = cv2.pyrDown( cv2.imread('img2Cal.png') )
#===============================================================================
# stereo = cv2.StereoSGBM_create(minDisparity=30, numDisparities=256, blockSize=25)
#===============================================================================
#===============================================================================
# stereo = cv2.StereoBM_create(numDisparities=320, blockSize=21)
#===============================================================================


#===============================================================================
# stereo = cv2.StereoBM_create(numDisparities=320, blockSize=35)
#===============================================================================

window_size = 4
min_disp = 16
num_disp = 64-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 3,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 25,
    speckleWindowSize = 300,
    speckleRange = 10
)

def calDisparity(img1, img2, name):
    cv2.imwrite(name + "blend.png", blend2(img1, img2))
    disparity = stereo.compute(img1,img2)
    print np.min(disparity), np.max(disparity)
    cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(name + "disparity.png", disparity) 
    
calDisparity(imgL, imgR, "cal")

