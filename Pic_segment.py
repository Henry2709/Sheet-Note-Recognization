#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:31:58 2018

@author: yifengluo

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
# Import libraries
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

"""
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
# Pre-defines functions
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
"""
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
    result[ image == 255] = 0
    result[ image == 0] = 255
    
    return result

def remove_staff_lines(image, staff_lines):
    
    image[staff_lines == 255] = 0
    
    return image

def binarize_and_convert(img):
    m,n = img.shape
    result = np.zeros([m, n], dtype = np.uint8)
    
    for i in range(m):
        for j in range(n):
            
            if img[i, j] > 200:
                result[i, j] = 0
            
            if img[i,j] < 200:
                result[i, j] = 255

    return result




"""
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    # Main function (Image Processing)
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    
    cv2.imshow('test', )
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    """

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
# Set up
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==

def segment(img):


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = convert(img_bw)

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    # Pre-process Image
    # (Morphological Filtering)
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==

    dash_filter = np.ones([1, 2])
    staff_line_filter = np.ones([1, 40])

    # Erosion: keep the horizontal lines only
    im_temp_ero = cv2.dilate(img_bw, dash_filter, iterations = 1)
    # Dilation: smooth the horizontal lines
    im_temp_dil = cv2.erode(im_temp_ero, staff_line_filter, iterations = 1)

    img_staff_lines = im_temp_dil

    notes = remove_staff_lines(img_bw, img_staff_lines)

    # Use closing to fill up missing parts
    # 1. Vertical
    vertical_filter = np.ones([3, 1])

    notes = cv2.dilate(notes, vertical_filter, iterations = 1)
    notes = cv2.erode(notes, vertical_filter, iterations = 1)

    # 2. Horizontal
    horizontal_filter = np.ones([1, 4])

    notes = cv2.dilate(notes, horizontal_filter, iterations = 1)
    notes = cv2.erode(notes, horizontal_filter, iterations = 1)
    note_copy = copy.copy(notes)
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    # Extract information (Find contours and segmentation)
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==

    img2, contours, hierarchy = cv2.findContours(note_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros(img_bw.shape)
    cv2.drawContours(canvas, contours, -1, 1)

    # Segmentation
    seg_set = []
    for i in range(len(contours)):
        # Use a list to store all contours
        contour_list = list(contours[i])
        
        # Coordinate of each contour
        row = []
        column = []
        
        for j in range(len(contour_list)):
            
            row.append(contour_list[j][0, 1])
            column.append(contour_list[j][0, 0])
        
        r_min = np.min(row)
        r_max = np.max(row)
        
        c_min = np.min(column)
        c_max = np.max(column)
        
        seg_set.append(notes[r_min-3 : r_max+3, c_min-3 : c_max+3])
    return seg_set,notes











