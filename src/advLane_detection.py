#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 18:31:57 2018

@author: sandeep
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def findWindowCentroids(image, window_width=35, window_height=120, margin=100, prevCentroids = None):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(2*image.shape[0]/3):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(2*image.shape[0]/3):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    #Threshold to remove noise in centroids
    th = 10000

    if prevCentroids != None: #if history available
        l_maxSum = np.convolve(window,l_sum).max()
        r_maxSum = np.convolve(window,r_sum).max()
        #Use Previous l_center in case of noise
        #print("max: ", l_maxSum)
        #print("max: ", r_maxSum)
        if l_maxSum < 250000:
            #print("skiped: ", l_maxSum)
            l_center = prevCentroids[0][0]
        if r_maxSum < 250000:
            #print("skiped: ", r_maxSum)
            r_center =  prevCentroids[0][1]

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        prev_l = l_center
        prev_r = r_center
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        l_maxSum = conv_signal[l_min_index:l_max_index].max()
        #Use Previous l_center in case of noise
        if l_maxSum < th:
            # print("skiped: ", l_maxSum)
            l_center = prev_l
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        r_maxSum = conv_signal[r_min_index:r_max_index].max()
        #Use Previous l_center in case of noise
        if r_maxSum < th:
            # print("skiped: ", r_maxSum)
            r_center = prev_r
        # Add what we found for that layer
        #print(l_maxSum,r_maxSum)
        window_centroids.append((l_center,r_center))
    return window_centroids, window_height, window_width

def polyFit(img, centroids, wh, ww):
    
    # Fit a second order polynomial to pixel positions
    ploty = np.linspace(0, img.shape[0]-1, num=img.shape[0])# to cover same y-range as image
    lx = [row[0] for row in centroids]
    rx = [row[1] for row in centroids]
    
    leftx = []
    rightx = []
    for i in range(len(centroids)):
        for j in range(wh):
            leftx.append(lx[i])
            rightx.append(rx[i])
            
    leftx = np.array(leftx[::-1])
    rightx = np.array(rightx[::-1])
    
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty*ploty + left_fit[1]*ploty + left_fit[2]
    left_fitx = np.array(left_fitx)
    
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty*ploty + right_fit[1]*ploty + right_fit[2]
    right_fitx = np.array(right_fitx)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)-10
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    a = ploty*ym_per_pix
    b = leftx*xm_per_pix
    c = rightx*xm_per_pix
    left_fit_cr = np.polyfit(a, b, 2)
    right_fit_cr = np.polyfit(a, c, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # average radius of curvature is in meters
    radius = (left_curverad+right_curverad)/2

    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = ((camera_center-(img.shape[1]/2)))*xm_per_pix

    #Debug code
    # for i in range(len(leftx)):
    #     cv2.putText(color_warp, "x",(int(leftx[i]),int(ploty[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    #     cv2.putText(color_warp, "o",(int(rightx[i]),int(ploty[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    return color_warp, radius, center_diff


#############
#Test Code
#############
'''

img = cv2.imread("./example.jpg")
img = img[:,:,1]
#plt.imshow(img[:,:,1],cmap='gray')
cent, wh, ww = findWindowCentroids(img)
out, a, b = polyFit(img,cent,wh,ww)
#out = markCentroids(img, ww=50, wh=80, margin=100)

plt.imshow(out)
plt.title('window fitting results')
#plt.gca().invert_yaxis() # to visualize as we do the images
plt.show()

# Plot up the fake data
#mark_size = 3
#plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
#plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, img.shape[0], color='green', linewidth=3)
plt.plot(right_fitx, img.shape[0], color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
'''