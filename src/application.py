#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:34:26 2018

@author: Sandeep Patil
"""
import pickle
import os
import advLane_calib
import advLane_imgproc as im
import advLane_detection as ln
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

#Camera Calibration
calibFolder = "../camera_cal/"
nx = 9
ny = 6

if os.path.exists(os.path.join(calibFolder,"calibData.p")):
    print("app:Info: Calib data found and the same will be loaded")
else:
    advLane_calib.calib(calibFolder, nx, ny)
    
calibData = {}
with open(os.path.join(calibFolder,"calibData.p"), "rb") as f:
    calibData = pickle.load(f)

#Read Sample Input Images and apply pipline
input_img_folder = "../test_images"
input_vid_folder = "../test_videos"
output_img_folder = "../output_images"
output_vid_folder = "../output_videos"

input_img_wildcard = "*.jpg"
input_vid_wildcard = "project*.mp4"

imgFileLst = glob.glob(os.path.join(input_img_folder, input_img_wildcard))
imgLst = []
for file in imgFileLst:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgLst.append({"filename": file.split("/")[-1], "img": img})

vidFileLst = glob.glob(os.path.join(input_vid_folder, input_vid_wildcard))
vidLst = []
for file in vidFileLst:
    vidLst.append({"filename": file.split("/")[-1]})

SAVE_SUFIX = ".jpg"
from collections import deque
centroidsDeque = deque([])
uesQueue = False
initCentroid = []
def detectionPipeline(img):
    #Pipeline Start

    # Undistort image
    undist = im.undistort(img, calibData)

    #uncomment below lines to draw Perspective transform choosen
    # src, dst, size = im.getSrcDstPts(undist)
    # lineThickness = 1
    # cv2.line(undist, (src[0][0],src[0][1]), (src[1][0],src[1][1]), (0,255,0), lineThickness)
    # cv2.line(undist, (src[1][0],src[1][1]), (src[2][0],src[2][1]), (0,255,0), lineThickness)
    # cv2.line(undist, (src[2][0],src[2][1]), (src[3][0],src[3][1]), (0,255,0), lineThickness)
    # cv2.line(undist, (src[3][0],src[3][1]), (src[0][0],src[0][1]), (0,255,0), lineThickness)

    # cv2.line(undist, (dst[0][0],dst[0][1]), (dst[1][0],dst[1][1]), (0,255,255), lineThickness)
    # cv2.line(undist, (dst[1][0],dst[1][1]), (dst[2][0],dst[2][1]), (0,255,255), lineThickness)
    # cv2.line(undist, (dst[2][0],dst[2][1]), (dst[3][0],dst[3][1]), (0,255,255), lineThickness)
    # cv2.line(undist, (dst[3][0],dst[3][1]), (dst[0][0],dst[0][1]), (0,255,255), lineThickness)

    #Select Color and gradiant
    colorFilter = im.colorSelect(undist)

    #Perform Perspective Transform
    transform = im.perspectiveTransform(colorFilter)

    #Find Centroids of sliding window for the lane detected
    if (len(centroidsDeque)):
        centroids, wh, ww = ln.findWindowCentroids(transform, prevCentroids=centroidsDeque[-1])
    else:
        centroids, wh, ww = ln.findWindowCentroids(transform)
    initCentroid.append(centroids[0])
    plt.plot(initCentroid)
    #Average over centroids to avoid wobbling of lanes in case of video
    if uesQueue:
        if len(centroidsDeque) > 3:
            centroidsDeque.popleft()
        centroidsDeque.append(centroids)
        centroids = np.average(centroidsDeque, axis=0)

    #Fit curve and calculate radius of curvataure and shift between center of lanes
    fit, radius, center_diff = ln.polyFit(colorFilter, centroids, wh, ww)
    #pipeline End

    #Perform Inverse Transform
    rev = im.inverseTransform(fit)

    #overlay lanes detected on the original image
    output = cv2.addWeighted(undist, 1, rev, 0.5, 0.0) # overlay the orignal road image with window results
    side_pos = "left"
    if center_diff <= 0:
        side_pos = "right"
    
    cv2.putText(output, "Radius of curvature: "+str(round(radius,3))+"(m)",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(output, "Vehicle is "+str(abs(round(center_diff,3)))+"m "+side_pos+" of center",(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)    
    return output
    #Uncomment for Debugging
    #return np.concatenate((undist,np.dstack((transform,transform,transform)),np.dstack((transform,fit[:,:,1],fit[:,:,0])), output),axis=1)
    #return np.concatenate((undist,np.dstack((colorFilter,colorFilter,colorFilter))),axis=1)

#Work on still images
for i in range(len(imgLst)):
    out = detectionPipeline(imgLst[i]["img"])
    outfile = os.path.join(output_img_folder,imgLst[i]["filename"].split(".")[-2]+SAVE_SUFIX)
    cv2.imwrite(outfile, cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
    print("app:Info: ",outfile, " Saved.")

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#work on videos
for i in range(len(vidLst)):
    vidInFile = os.path.join(input_vid_folder, vidLst[i]["filename"])
    vidOutFile = os.path.join(output_vid_folder, vidLst[i]["filename"])
    
    #Create queue for averaging the centroids
    centroidsDeque = deque([])
    uesQueue = True
    clip1 = VideoFileClip(vidInFile)#.subclip(20,26)
    white_clip = clip1.fl_image(detectionPipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(vidOutFile, audio=False)
    uesQueue = False

