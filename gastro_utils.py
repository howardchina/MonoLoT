import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

"""locateView
locate view and format it with x, y, w, h
Args:
    img
    th
    show
Return:
    x, y, w, h
"""
def locateView(img, th=15, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, th, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        if show is True:
            contour_img = img.copy()
            # draw the biggest contour (c) in green
            cv2.rectangle(contour_img,(x,y),(x+w,y+h),(0,255,0),2)
            plt.figure()
            plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        return x, y, w, h
    else:
        return -1, -1, -1, -1

# examples:
# x, y, w, h = locateView(image, show=True)
# print(x, y, w, h)
# croped_img = image[y:y+h, x:x+w]
# plt.figure()
# plt.imshow(cv2.cvtColor(croped_img, cv2.COLOR_BGR2RGB))

"""calibrateCamera
calculate intrinsic matrix and distortion params
Args:
    images
    show
    crop
    crop_params
    resize
    resize_params
Return:
    matrix, distortion
"""
def calibrateCamera(images, show=False, crop=False, crop_params=None, resize=False, resize_params=None):
    # Define the dimensions of checkerboard
    CHECKERBOARD = (8, 11)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    # 3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    for filename in images:
        image = cv2.imread(filename)
        if crop is True:
            x, y, w, h = crop_params
            image = image[y:y+h, x:x+w]
        if resize is True:
            image = cv2.resize(image, resize_params)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            if show is True:
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
                plt.figure()
                plt.axis('off')
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    
    mean_error = 0
    for i in range(len(threedpoints)):
        imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
        error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print("total error: ", mean_error / len(threedpoints), "valid image number: ", len(threedpoints))
    
    return matrix, distortion