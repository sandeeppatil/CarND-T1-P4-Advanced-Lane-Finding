#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:19:44 2018

@author: Sandeep Patil
"""
import cv2
import numpy as np

#def colorSelect(img, s_thresh=(170, 255), sx_thresh=(25, 35), rgb_low_th=(193,193,0), rgb_high_th=(255,255,255)):
def colorSelect(img, s_thresh=(150, 255), sx_thresh=(25, 35), rgb_low_th=(193,193,0), rgb_high_th=(255,255,255)):
    img = np.copy(img)

    img = cv2.GaussianBlur(img,(5,5),0)

    # Get R, G and B channel information
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]

    # Convert to HSV color space and separate the V channel
    # Get L and S from HLS color format
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255
    
    # Threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    # Threshold RGB channels
    rgb_binary = np.zeros_like(r_channel)
    rgb_binary[
        ((r_channel >= rgb_low_th[0]) & (r_channel <= rgb_high_th[0])) &
        ((g_channel >= rgb_low_th[1]) & (g_channel <= rgb_high_th[1])) &
        ((b_channel >= rgb_low_th[2]) & (b_channel <= rgb_high_th[2]))
        ] = 255

    color_binary = np.maximum(rgb_binary, sxbinary, s_binary)
    return color_binary

def undistort(img, calibData):
    img = cv2.undistort(img, calibData["mtx"], calibData["dist"], None, calibData["mtx"])
    return img

def getSrcDstPts(img):
    img_size = (img.shape[1], img.shape[0])
    width = img.shape[1]
    height = img.shape[0]
    src = np.float32(
        [[(width / 2) - 62, height / 2 + 100],
        [((width / 6) - 10), height],
        [(width * 5 / 6) + 60, height],
        [(width / 2 + 68), height / 2 + 100]])
    dst = np.float32(
        [[(width / 4)-80, 0],
        [(width / 4)-80, height],
        [(width * 3 / 4)+80, height],
        [(width * 3 / 4)+80, 0]])    
    return src, dst, img_size

def perspectiveTransform(img):
    # Given src and dst points, calculate the perspective transform matrix
    src, dst, img_size = getSrcDstPts(img)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped

def inverseTransform(img):
    src, dst, img_size = getSrcDstPts(img)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped
