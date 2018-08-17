#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:50:57 2018

@author: Sandee Patil
"""
import glob
import numpy as np
import cv2
import pickle
import os

def calib(srcFolder, nx, ny):
    
    #Read list of files in the input folder
    fileList = glob.glob(os.path.join(srcFolder,"*.jpg"))
    print("calib:Info: Total no. of input images for calibration: ",
          len(fileList))
    
    cornerList = []
    imgshape = ()
    for file in fileList:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgshape = img.shape
        ret, corners = cv2.findChessboardCorners(
                img, 
                (nx, ny), 
                None
                )
        if ret == True:
            cornerList.append(corners)
        else:
            print("calib:Warning: Couldn't find expeced corners on ", file)
    
    objpts = [] # 3D points in the real world space
    imgpts = [] # 2D points in teh image plane
    
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    for corners in cornerList:
        objpts.append(objp)
        imgpts.append(corners)
    
    #Calculate Calibration Matrix and Distortion Matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpts, 
            imgpts, 
            imgshape, 
            None, 
            None
            )
    
    calibData = {
            "ret": ret, 
            "mtx": mtx, 
            "dist": dist, 
            "rvecs": rvecs, 
            "tvecs": tvecs, 
            "nx": nx, 
            "ny": ny
            }

    #Save Calibration Data on Hard Disk
    outputfile = os.path.join(srcFolder,"calibData.p")
    pickle.dump(calibData, open(outputfile, "wb"))
    print("calib:Info: Calibration data stored at ", outputfile)
    return ret, mtx, dist, rvecs, tvecs

##Test
#calib("../camera_cal", 9, 6)