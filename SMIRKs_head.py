#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:13:15 2018

@author: jinzhao
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage
from collections import Counter



def convert(image):
    ''' 
    # Attempt 1
    m, n = image.shape
    
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if image[i, j] == 255:
                result[i, j] = 0
                
            if image[i, j] == 0:
                result[i, j] = 255
    
    return result
    '''
    # Attempt 2
    result = copy.copy(image)
    result[image == 255] = 0
    result[image == 0] = 255
    
    return result  


def get_most_common(data):
# =============================================================================   
#   data_ = sorted(data)

#   half = len(data_) // 2
#   return (data_[half] + data_[half+1]) / 2
# =============================================================================
     most_common = Counter(data).most_common(1)
     
     return most_common[0][0] 

    
def get_col_hist(img):
    height, width = img.shape
    hist = [0] * width    
    
    for j in range(width):
        hist[j] = np.sum(img[:, j] / 255)  
    
    return hist
        

def get_staff_lines(img, dash_filter, staff_line_filter):
    
    # Dilation: Smooth the horizontal lines
    temp = cv2.dilate(img, dash_filter, iterations = 1)
    # Erosion: keep the horizontal lines only
    img_staff_lines = cv2.erode(temp, staff_line_filter, iterations = 1)
    
    height, width = img_staff_lines.shape
    
    hist_row = [0] * height
    for i in range(height):
        hist_row[i] = np.sum(img_staff_lines[i, :] / 255)  
    
#     plt.figure
#     plt.bar(range(height), hist_row)
      
    idx_staff = []
    for i in range(height):
        if hist_row[i] > 0.1 * width:
            idx_staff.append(i)
            
    # modfiy the index of staff to fit different size of images
    idx_staff_new = []
    flag = True
    for i in range(len(idx_staff) - 1):
        
        if idx_staff[i+1] - idx_staff[i] == 1:
            flag = False
        else:
            flag = True
            
        if flag:
            idx_staff_new.append(idx_staff[i])
            
    idx_staff_new.append(idx_staff[-1])
    
    return img_staff_lines, idx_staff_new

def remove_staff_lines(img, staff_lines, diff_staff):
    image_result = copy.copy(img)
    image_result[staff_lines == 255] = 0
    
    # Use closing to fill up missing parts
    tmp = diff_staff // 2 + 1
    # 1. Vertical closing
    vertical_filter = np.ones([tmp, 1]) 

    image_result = cv2.dilate(image_result, vertical_filter, iterations = 1)
    image_result = cv2.erode(image_result, vertical_filter, iterations = 1)

    # 2. Horizontal closing
    horizontal_filter = np.ones([1, tmp])

    image_result = cv2.dilate(image_result, horizontal_filter, iterations = 1)
    image_result = cv2.erode(image_result, horizontal_filter, iterations = 1)
    
    return image_result

def generate_disk_filter(radius):
    
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype = 'uint8')
    
    return disk

def opening(image, mor_filter):
    
    temp = cv2.erode(image, mor_filter, iterations = 1)
    image_result = cv2.dilate(temp, mor_filter, iterations = 1)
    
    return image_result
 
def determine_note_pos(index_staff, moment_row_ind):
    # Add the 6th staff line
    diff_staff = index_staff[1] - index_staff[0]
    index_staff = index_staff + [index_staff[-1] + diff_staff]
    
    # Partition the staff line
    step = diff_staff // 4

    if moment_row_ind >= index_staff[5]-step and moment_row_ind < index_staff[5]+step:
        return 1 #Do
    if moment_row_ind >= index_staff[4]+step and moment_row_ind < index_staff[5]-step:
        return 2 #Re
    if moment_row_ind >= index_staff[4]-step and moment_row_ind < index_staff[4]+step:
        return 3 #Mi
    if moment_row_ind >= index_staff[3]+step and moment_row_ind < index_staff[4]-step:
        return 4 #Fa
    if moment_row_ind >= index_staff[3]-step and moment_row_ind < index_staff[3]+step:
        return 5 #Sol
    if moment_row_ind >= index_staff[2]+step and moment_row_ind < index_staff[3]-step:
        return 6 #La
    if moment_row_ind >= index_staff[2]-step and moment_row_ind < index_staff[2]+step:
        return 7 #Ti
    
    if moment_row_ind >= index_staff[1]+step and moment_row_ind < index_staff[2]-step:
        return 8 #Do
    if moment_row_ind >= index_staff[1]-step and moment_row_ind < index_staff[1]+step:
        return 9 #Re
    if moment_row_ind >= index_staff[0]+step and moment_row_ind < index_staff[1]-step:
        return 10 #Mi
    if moment_row_ind >= index_staff[0]-step and moment_row_ind < index_staff[0]+step:
        return 11 #Fa
    
def compute_moments(contours):
    n_notes = len(contours)
    moments = np.empty((0, 2))
    
    for i in range(n_notes):
        cnt = contours[i]
        M = cv2.moments(cnt)
        col_ind = int(M['m10']/M['m00'])
        row_ind = int(M['m01']/M['m00'])  # We only care about its row index
        centroid = np.array([row_ind, col_ind])
        moments = np.vstack((moments, centroid))
        
    tmp_arg = np.argsort(moments[:, 1]) 
    moments = moments[tmp_arg]
    
    return moments

def determine_edge(hist, median):
    width = len(hist)
    
    count = 0
    for i in range(width):        
        if hist[i] > median and count == 0:
            count += 1
            start = i - 1
            
        if hist[i] <= median and count == 1:
            end = i - 1
            break
    return start, end