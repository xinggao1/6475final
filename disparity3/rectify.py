import numpy as np

import cv2






try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create()
            print "use ORB"
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)
            

M = np.array( [[  1.92949648e+03,   0.00000000e+00,   1.20508238e+03],
 [  0.00000000e+00,   1.92774278e+03,   7.79828847e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
 
dis = np.array([[-0.18403257,  0.04334305, -0.00142467, -0.00183455,  0.32721892]])
 
#===============================================================================
# R = np.array([[  9.99994906e-01,  -3.18560762e-03 , -1.99977206e-04],
#  [  3.18783948e-03 ,  9.99918267e-01  , 1.23813564e-02],
#  [  1.60518718e-04 , -1.23819308e-02 ,  9.99923328e-01]])
# T = np.array( [[-3.28237759],
#  [-0.02718054],
#  [ 0.07851459]])
# E = np.array([[ -2.54654900e-04  ,-7.81716268e-02 , -2.81505711e-02],
#  [  7.90410747e-02 , -4.08922888e-02,   3.28211022e+00],
#  [  1.67167067e-02 , -3.28219589e+00  ,-4.06457222e-02]])
# F = np.array([[  1.70826316e-09  , 5.24864009e-07  ,-4.69999697e-05],
#  [ -5.30701701e-07 ,  2.74810909e-07,  -4.20949169e-02],
#  [  1.95427696e-04  , 4.16744508e-02,  1.00000000e+00]])
#===============================================================================

R =np.array([[ 0.99998848, -0.00436078 ,-0.00200511],
 [ 0.00437696 , 0.9999573 , 0.00813895],
 [ 0.00196953, -0.00814763 , 0.99996487]])
T =np.array([[-3.20702999],
 [-0.02845684],
 [ 0.05973959]])
E =np.array([[ -3.17524434e-04,  -5.95051866e-02 , -2.89420561e-02],
 [  6.60552411e-02  ,-2.63902141e-02  , 3.20679753e+00],
 [  1.44194643e-02 , -3.20701714e+00,  -2.61589169e-02]])
F =np.array( [[  4.31021768e-09  , 8.08484653e-07  , 1.22371943e-04],
 [ -8.97478888e-07  , 3.58884563e-07 , -8.32666761e-02],
 [  3.17013426e-04  , 8.28199414e-02  , 1.00000000e+00]])

         
            
def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
  """ Draws the matches between the image_1 and image_2.
  This function is provided to you for visualization because there were
  differences in the OpenCV 3.0.0-alpha implementation of drawMatches and the
  2.4.9 version, so we decided to provide the functionality ourselves.
  Note: Do not edit this function, it is provided for you for visualization
  purposes.
  Args:
    image_1 (numpy.ndarray): The first image (can be color or grayscale).
    image_1_keypoints (list): The image_1 keypoints, the elements are of type
                              cv2.KeyPoint.
    image_2 (numpy.ndarray): The image to search in (can be color or grayscale).
    image_2_keypoints (list): The image_2 keypoints, the elements are of type
                              cv2.KeyPoint.
  Returns:
    output (numpy.ndarray): An output image that draws lines from the input
                            image to the output image based on where the
                            matching features are.
  """
  # Compute number of channels.
  num_channels = 1
  if len(image_1.shape) == 3:
    num_channels = image_1.shape[2]
  # Separation between images.
  margin = 10
  # Create an array that will fit both images (with a margin of 10 to separate
  # the two images)
  joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                           image_1.shape[1] + image_2.shape[1] + margin,
                           3))
  if num_channels == 1:
    for channel_idx in range(3):
      joined_image[:image_1.shape[0],
                   :image_1.shape[1],
                   channel_idx] = image_1
      joined_image[:image_2.shape[0],
                   image_1.shape[1] + margin:,
                   channel_idx] = image_2
  else:
    joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
    joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

  for match in matches:
    image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                     int(image_1_keypoints[match.queryIdx].pt[1]))
    image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] + \
                         image_1.shape[1] + margin),
                   int(image_2_keypoints[match.trainIdx].pt[1]))

    cv2.circle(joined_image, image_1_point, 5, (0, 0, 255), thickness = -1)
    cv2.circle(joined_image, image_2_point, 5, (0, 255, 0), thickness = -1)
    cv2.line(joined_image, image_1_point, image_2_point, (255, 0, 0), \
             thickness = 3)
  return joined_image
            
def drawlines(img1,img2,lines,pts1,pts2):
     ''' img1 - image on which we draw the epilines for the points in img2
     lines - corresponding epilines '''
     r,c = img1.shape
     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
     for r,pt1,pt2 in zip(lines,pts1,pts2):
         color = tuple(np.random.randint(0,255,3).tolist())
         x0,y0 = map(int, [0, -r[2]/r[1] ])
         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
     return img1,img2
            
def remapImg(img1, size, h1):
    MinvH1M = np.dot(np.linalg.inv(M), np.dot(h1, M))
    mapx1, mapy1 = cv2.initUndistortRectifyMap(M, dis, MinvH1M, M, size, cv2.CV_32FC1)
    img1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
    return img1

img1 = cv2.imread('l.jpg',0)#[700:1300, 380:2030]  #queryimage # left image
img2 = cv2.imread('r.jpg',0)#[700:1300, 380:2030] #trainimage # right image4

img1 = cv2.undistort(img1, M, dis)
img2 = cv2.undistort(img2, M, dis)

#===============================================================================
# width = img1.shape[1]
# height = img1.shape[0]
# one = np.array([1])
# zero = np.array([0])
# ones = np.tile(one, [height, width/2])
# zeros = np.tile(zero, [height, width - ones.shape[1]])
# maskLeft = np.concatenate((zeros, ones), axis = 1)
# maskRight = np.concatenate((ones, zeros), axis = 1)
# 
# print maskRight.astype(np.int8)
# kp1L, des1L = SIFT.detectAndCompute(img1,maskRight.astype(np.int8))
# kp1R, des1R = SIFT.detectAndCompute(img1,maskLeft.astype(np.int8))
# kp1 = kp1L + kp1R
# des1 = des1L + des1R
# print len(kp1L), len(kp1)
#===============================================================================

kp1, des1 = SIFT.detectAndCompute(img1,None)
kp2, des2 = SIFT.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)


flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)



good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


output = drawMatches(img1, kp1, img2, kp2, sorted(good, key = lambda x:x.distance)[:100])
cv2.imwrite("output.png", output)        
        
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print F

print pts1.shape;
pts1 = pts1[mask.ravel()==1]
print pts1.shape;
pts2 = pts2[mask.ravel()==1]
 

 # Find epilines corresponding to points in right image (second image) and
     # drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

     # Find epilines corresponding to points in left image (first image) and
     # drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

   

pp1 = pts1.reshape((pts1.shape[0] * 2, 1))
pp2 = pts2.reshape((pts2.shape[0] * 2, 1))

size =  img1.shape[1], img1.shape[0]

suc, h1, h2  = cv2.stereoRectifyUncalibrated(pp1, pp2, F, size)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M, dis, M, dis, size, R, T) 
print validPixROI1, validPixROI2

cv2.imwrite("img2p.png", img3)
cv2.imwrite("img1p.png", img5) 


img5Un = cv2.warpPerspective(img5, h1, size)
img3Un = cv2.warpPerspective(img3, h2, size)
cv2.imwrite("img2r Un.png", img3Un)
cv2.imwrite("img1r Un.png", img5Un) 

img1 = cv2.imread('l.jpg',-1)
img2 = cv2.imread('r.jpg',-1)

img1Un = remapImg(img1, size, h1)
img2Un = remapImg(img2, size, h2)


def remapImgCalibrated(img1, size, R1, P1):
    mapx1, mapy1 = cv2.initUndistortRectifyMap(M, dis, R1, P1, size, cv2.CV_32FC1)
    img1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
    return img1

rowmin = max(validPixROI1[1], validPixROI2[1])
rowmax = min(validPixROI1[3], validPixROI2[3])
colmin = max(validPixROI1[0], validPixROI2[0])
colmax = min(validPixROI1[2], validPixROI2[2])

img1Cal = remapImgCalibrated(img1, size, R1, P1)[rowmin:rowmax, colmin:colmax]
img2Cal = remapImgCalibrated(img2, size, R2, P2)[rowmin:rowmax, colmin:colmax]

#===============================================================================
# img5Cal = remapImgCalibrated(img5, size, R1, P1)[rowmin:rowmax, colmin:colmax]
# img3Cal = remapImgCalibrated(img3, size, R2, P2)[rowmin:rowmax, colmin:colmax]
# cv2.imwrite("img2r cal.png", img3Cal)
# cv2.imwrite("img1r cal.png", img5Cal) 
#===============================================================================

print img1.shape
#print img2.shape
 

#===============================================================================
# print disparity[200:220, 600:620]
#===============================================================================
cv2.imwrite("img1Un.png", img1Un)
cv2.imwrite("img2Un.png", img2Un)
cv2.imwrite("img1Cal.png", img1Cal)
cv2.imwrite("img2Cal.png", img2Cal) 
 




