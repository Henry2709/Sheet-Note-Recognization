#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:01:58 2018

@author: jinzhao
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage
from SMIRKs_head import *
    
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Setup
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
img = cv2.imread('music.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw = convert(img_bw)
height, width = img_bw.shape

cv2.imshow('test', img_bw)
cv2.waitKey(1)
cv2.destroyAllWindows() 

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
#  Define some filters
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
dash_filter = np.ones([1, 2])  
staff_line_filter = np.ones([1, width//10])

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
#  Partition the image into individual staff lines
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

img_staff_lines, idx_staff = get_staff_lines(img_bw, dash_filter, staff_line_filter)

cv2.imshow('test', img_bw)
cv2.imshow('test1', img_staff_lines)
cv2.waitKey(1)
cv2.destroyAllWindows()

num_staff = len(idx_staff) // 5
diff_staff = (idx_staff[4] - idx_staff[0])  // 4

img_div_set = []
for i in range(num_staff):
    idx_start = idx_staff[5*i] - 2 * diff_staff
    idx_end = idx_staff[5*i+4] + 2 * diff_staff
    
    if idx_start < 0:
        idx_start = 0
        
    if idx_end > height:
        idx_end = height
    
    img_div = img_bw[idx_start:idx_end, :]
    img_div_set.append(img_div)
    
    
# Show!
for i in range(len(img_div_set)):    
    cv2.imshow('test'+str(i), img_div_set[i])
    
cv2.waitKey(1)
cv2.destroyAllWindows()

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Only keep the main part of staff line
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
for it in range(len(img_div_set)):
    img_div = img_div_set[it]
    hei_div, wid_div = img_div.shape
    
    hist_col = [0] * wid_div    
    for j in range(wid_div):
        hist_col[j] = np.sum(img_div[:, j] / 255)  
    
    # Get the most common element in hist_col, which corresponds to the staff line
    median = get_most_common(hist_col)
    
    # Find the right and left edge of the staff
    # Right edge
    for i in range(wid_div):    
        
        if hist_col[i] == 0: 
            if hist_col[i+2] == median: 
                right_start = i
                right_end = i+2
                break
            
            if hist_col[i+3] == median: 
                right_start = i
                right_end = i+3
                break
    
    # Left edge
    for i in range(wid_div//2, wid_div):
        
        if hist_col[i] == 0:
            left_end = i - 1
            break
        
    for i in range(wid_div//2, left_end)[::-1]:
        
        if hist_col[i] == median:
            left_start = i
            break
      
    img_tmp = img_div[:, right_end : left_start]
    img_div_set[it] = img_div[:, right_end : left_start]
    
# Show!
for i in range(len(img_div_set)):    
    cv2.imshow('test'+str(i), img_div_set[i])   
cv2.waitKey(1)
cv2.destroyAllWindows()

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
# Process each staff line individually 
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 1. Determine the type of clef (treble or bass) 
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

output_clef = [] # The length of the output should be same as len(img_division_set)
  
for it in range(len(img_div_set)):
    img_div = img_div_set[it]
    hei_div, wid_div = img_div.shape
    
    div_staff_lines, idx_staff_set = get_staff_lines(img_div, dash_filter, staff_line_filter)
    
    hist_col = [0] * wid_div    
    for j in range(wid_div):
        hist_col[j] = np.sum(img_div[:, j] / 255)  
    
    # Get median from hist_col
    median = get_most_common(hist_col)
    
    # Find the right and left edge of the clef
    clef_start, clef_end = determine_edge(hist_col, median)
    
    
    staff_5 = idx_staff_set[-1]
    staff_4 = idx_staff_set[-2]   
    area = (staff_5 - staff_4) * (clef_end - clef_start)   
    
    n_pixels = 0
    for i in range(staff_4, staff_5):
        for j in range(clef_start, clef_end):
            if img_div[i, j] == 255:
                n_pixels += 1
                
    density = n_pixels / area  
    if density > 0.25:
        type_clef = 1; # Treble       
    else:
         type_clef = 0; # Bass
         
    output_clef.append(type_clef) 
    
    # Deal with the special situation when the type of clef is 'bass'
    if type_clef == 0:
        count = 0
        for i in range(clef_end, wid_div):
            
            if hist_col[i] > median and count == 0:
                count += 1
                
            if hist_col[i] <= median and count == 1:
                clef_end = i
                break
                
    img_tmp = img_div[:, clef_end : -1]           
    # After determine the type of clef, we no longer need them.
    img_div_set[it] = img_div[:, clef_end : -1]

# Show!
for i in range(len(img_div_set)):    
    cv2.imshow('test'+str(i), img_div_set[i]) 
# =============================================================================
# cv2.imshow('test', img_bw)
# cv2.imshow('test', img_tmp)
# =============================================================================
cv2.waitKey(1)
cv2.destroyAllWindows()


# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 2. Delete Beats and key signature before notes 
# (This part would be used to detect the type of key signatures (sharp or flat) and the Beats in the future)
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Only the 1st and 2nd staff line (1st segmentation and 2nd segmentation) are taken into consideration
for it in range(0, 2):
    img_div = img_div_set[it]
    hei_div, wid_div = img_div.shape
    
    div_staff_lines, idx_staff_set = get_staff_lines(img_div, dash_filter, staff_line_filter)
    diff_staff = idx_staff_set[1] - idx_staff_set[0]
     
    horizontal_filter = np.ones([1, diff_staff])
    img_tmp = cv2.dilate(img_div, horizontal_filter, iterations = 1)

    hist_tmp = get_col_hist(img_tmp)
    hist_div = get_col_hist(img_div)     
    median = get_most_common(hist_div)
    key_start, key_end = determine_edge(hist_tmp, median)
    
    img_div_set[it] = img_div[:, key_end+1 : -1]
    
# Show!!!
for i in range(len(img_div_set)):    
    cv2.imshow('test'+str(i), img_div_set[i]) 
# cv2.imshow('test1', img_tmp)
cv2.imshow('test_bw', img_bw)
cv2.waitKey(1)
cv2.destroyAllWindows()  










# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 2. Determine the position and type of each note
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====  

output_note_pos = []
output_note_type = []
for it in range(len(img_div_set)):
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    # Detect types
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    img_div = img_div_set[2]
    div_staff_lines, idx_staff = get_staff_lines(img_div, dash_filter, staff_line_filter)
    diff_staff = idx_staff[1] - idx_staff[0]
       
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # a) Detect quarter and eighth notes
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    
    # Quarter note + Eighth note
    disk_filter = generate_disk_filter(diff_staff//2.5)
    img_note1 = opening(img_div, disk_filter) 
    im1, contours1, hierarchy1 = cv2.findContours(img_note1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    moments1 = compute_moments(contours1)
    
    # Sequential Eighth notes 
    disk_filter = generate_disk_filter(diff_staff//4)
    img_note3 = opening(img_div, disk_filter) 
    im3, contours3, hierarchy3 = cv2.findContours(img_note3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_area_set = []
    for i in range(len(contours3)):
        cnt = contours3[i]
        area = cv2.contourArea(cnt)
        contour_area_set.append(area)
        
    median = np.median(contour_area_set)
    std = np.std(contour_area_set)
    
    contour_dash = []
    canvas = np.zeros((img_note3.shape))
    for i in range(len(contours3)):
        if contour_area_set[i] > median + 1.3 * std:
            contour_dash.append(contours3[i])   
            
    cv2.drawContours(canvas, contour_dash, -1, 1)  
 
    moments_dash = compute_moments(contour_dash)
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # b) Detect whole and half notes
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    img_note = remove_staff_lines(img_div, div_staff_lines, diff_staff)
    img_note = cv2.dilate(img_note, np.ones((2, 2)), iterations = 1)
    
    img_note_fill = scipy.ndimage.binary_fill_holes(img_note).astype('uint8')
    img_note_fill[img_note_fill == 1] = 255
    img_note2 = img_note_fill - img_note
    
    disk1 = generate_disk_filter(diff_staff // 4)
    disk2 = generate_disk_filter(diff_staff // 2)
    tmp = cv2.erode(img_note2, disk1, iterations = 1)
    img_note_new = cv2.dilate(tmp, disk2, iterations = 1)
    
    im2, contours2, hierarchy2 = cv2.findContours(img_note_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)          
    moments2 = compute_moments(contours2)   
  
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    # 1) Determine the position
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    
    # Sort to make sure the notes are in order
    moments = np.vstack((moments1, moments2))  
    tmp_arg = np.argsort(moments[:, 1]) 
    moments = moments[tmp_arg]
    
    note_pos = []
    for i in range(len(moments)):
        row_idx = moments[i, 0]
        output = determine_note_pos(idx_staff, row_idx)
        note_pos.append(output)   
        
    output_note_pos.append(note_pos)
    
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    # 2) Determine the type
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    
    # Quarter and eighth note
    n_note1 = moments1.shape[0]
    note_type1 = np.column_stack((moments1[:, 1], [4] * n_note1))  
    
    for i in range(len(moments_dash)):
        for j in range(len(moments1)-1):
            if moments1[j, 1] < moments_dash[i, 1] and moments1[j+1, 1] > moments_dash[i, 1]:
                note_type1[j, 1] = 8
                note_type1[j+1, 1] = 8
                break
            
    # Half note
    n_note2 = moments2.shape[0]
    note_type2 = np.column_stack((moments2[:, 1], [2] * n_note2)) 
    
    # Combine and sort
    note_type = np.vstack((note_type1, note_type2))
    
    tmp_arg = np.argsort(note_type[:, 0]) 
    note_type = note_type[tmp_arg]
    
    # Add to output
    note_type = list(note_type[:, 1])   
    output_note_type.append(note_type)
    
    
cv2.imshow('test', img_bw)
cv2.imshow('test1', img_note3)
cv2.imshow('test2', img_div)

cv2.waitKey(1)
cv2.destroyAllWindows() 


