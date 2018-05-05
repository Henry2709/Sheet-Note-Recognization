#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:31:58 2018

@author: yifengluo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os


"""
 ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
SEGMENT
 ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
""" 


def get_staff_lines(img, dash_filter, staff_line_filter):
    
    # Dilation: Smooth the horizontal lines
    temp = cv2.dilate(img, dash_filter, iterations = 1)
    # Erosion: keep the horizontal lines only
    img_staff_lines = cv2.erode(temp, staff_line_filter, iterations = 1)
    
    height, width = img_staff_lines.shape
    
    hist_row = [0] * height
    for i in range(height):
        hist_row[i] = np.sum(img_staff_lines[i, :] / 255)  
    
    idx_staff = []
    for i in range(height):
        if hist_row[i] > 0.1 * width:
            idx_staff.append(i)
            
    # modfiy the index of staff to fit different size of images
    idx_staff_new = []
    flag = True
    for i in range(len(idx_staff)-1):
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

def convert(image):

    result = copy.copy(image)
    result[image == 255] = 0
    result[image == 0] = 255
    
    return result

def binary(pic):
    m = pic.shape[0]
    n = pic.shape[1]
    new_pic = np.copy(pic)
    for i in range(m):
        for j in range(n):
            if pic[i][j] >= 120:
                new_pic[i][j]= 0
            elif pic[i][j]< 120:
                new_pic[i][j] = 255
    return new_pic

def area_detect(pic):
    m = pic.shape[0]
    n = pic.shape[1]
    new_pic = np.copy(pic)
    new_pic2 = np.copy(pic)
    lines = []
    for i in range(m-1):
        sum = 0
        for j in range(n):
            if pic[i][j] == 0:
                new_pic[i][j]= 1
                sum+=1
                
            elif pic[i][j]== 255:
                new_pic[i][j] = 0
        if sum < (n * 0.4):
            new_pic2[i,:] = 0
            lines.append(i) 
    return new_pic2,lines

def closing(pic,index):

    n = pic.shape[1]
    for i in (index):
        for j in range(n):
            if pic[i-1][j] == 255:
                pic[i][j] = 255
            elif pic[i+1][j] == 255:
                pic[i][j] == 255
    return pic




def Seg(gray):

    b_g = binary(gray)

    (gray_new,index) = area_detect(b_g)
    final = closing(gray_new,index)
    return final

def K_NN(X_train,y_train,X_test,k):
    m=np.shape(X_test)[0]
    a=np.shape(X_train)[0]
    b=np.shape(X_train)[1]
    List=[]
    Rank=[]
    for i in range(m):
        weight=[]
        
        #X_test[i,k]
        for j in range(a):
            diff=0
            for e in range(b):
                diff+=abs(X_train[j,e]-X_test[i,e])
            weight.append(diff)
        temp=[]
    
        Inf=1000000000
    
        for i in range(k):
           
            temp.append(weight.index(min(weight)))
        
            weight[weight.index(min(weight))]=Inf
        result=[y_train[x] for x in temp]
        result_1=sum(result)/len(result)
        if result_1<-0.5:
            R=-1
        else:
            R=0
        List.append(R)
        Rank.append(y_train[temp])
    return List,list(Rank)
        

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

def Noise_eliminate(List) :  
    for i in range(len(List)):
        if len(List[i])>1:
            New = [List[i][a].shape[0]*List[i][a].shape[1] for a in range(len(List[i]))]
            a = Rank(New)
            List[i] = [List[i][int(a[-1,-1])]]
    return List


def segment(img_gray):
    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = convert(img_bw)

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    # Pre-process Image
    # (Morphological Filtering)
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
    
    dash_filter = np.ones([1, 2])
    staff_line_filter = np.ones([1, 25])
    
    # Erosion: keep the horizontal lines only
    im_temp_ero = cv2.dilate(img_bw, dash_filter, iterations = 1)
    # Dilation: smooth the horizontal lines
    im_temp_dil = cv2.erode(im_temp_ero, staff_line_filter, iterations = 1)
    
    img_staff_lines = im_temp_dil
    
    notes = remove_staff_lines(img_bw, img_staff_lines)
    save_notes = copy.copy(notes)
    
   
    # Use closing to fill up missing parts
    # 1. Vertical
    vertical_filter = np.ones([5, 1])
    
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
        
        seg_set.append(save_notes[r_min-3 : r_max+3, c_min-3 : c_max+3])
    return seg_set



def Rank(List):
    List_copy = copy.copy(List)
    lens = len(List_copy)
    New = np.zeros([lens,2])
    for i in range(lens):
        Min = np.min(List_copy)
        Max = np.max(List_copy) 
        Index=List_copy.index(Min)
        New[i,0]=List_copy[Index]
        New[i,1]=Index
        List_copy[Index] = Max+1    
    return New

def Remove(List):
    New_list = []
    for i in range(len(List)):
        m,n = List[i][0].shape
        Min = np.min([m,n])
        if Min != 0:
            New_list.append(List[i])
    return New_list
def Norm_to_one(List):
    lens = len(List)
    New_list = []
    for i in range(lens):
        m,n = List[i].shape
        a = np.zeros([m,n])
        a[List[i]>=1]=1
        New_list.append(a)
    return New_list
            


"""
 ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
SIFT
 ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==
""" 

def Data_create(Name):
    Database = {}     
    for i in range(len(Name)):
        name = [] 
        for filename in os.listdir(r'%s'%(Name[i])):
            name.append(filename) 
        name.remove('.DS_Store')
        img_data = []
        for j in range(len(name)):
            img_data.append(cv2.cvtColor(cv2.imread(r'%s/%s'%(Name[i],name[j])), cv2.COLOR_BGR2GRAY))            
        Database['%s'%(Name[i])]=img_data
    return Database
def Test(Test_data,Database): 
#############################    
#    Test_data=Test_processed
#    Database=DB_processed
#    
############################3    
    for nn,m in enumerate(Test_data):
        List_1 = []
        for n in range(len(Test_data[m])):
            gray = Test_data[m][n]
            ######

            ######
            finaltest=siftlearn(gray)
            bf = cv2.BFMatcher()
            Num = {}
            for num,i in enumerate(Database):
                List = []
                for j in range(len(Database[i])):
                    if len(DB_feature[i][j][0]) !=0:
                        match = bf.knnMatch(finaltest[1], DB_feature[i][j][1],k=2) 
                        if len(match)!=0:
                            if len(match[0]) !=1:
                                List.append(good(match))
                
                Num[i]=List
            
            Average = {}
            for num,i in enumerate(Num):
                if len(Num[i]) ==0:
                    Average[i] = 0
                else:
                    Sum = 0
                    for j in range(len(Num)):
                        Sum += Num[i][j]
    
    
                        Average[i] = Sum/len(Num[i])
            List_1.append(Average)
    return List_1
def processed(Database):
    DB_processed = {}
    for num,i in enumerate(Database):
        List = []
        for j in range(len(Database['%s'%(i)])):
            L= Seg(Database['%s'%(i)][j])

            List.append(L)


        DB_processed['%s'%(i)] = List
    return DB_processed

def siftlearn(gray):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    _,des = sift.compute(gray,kp)  
    return [kp,des]

def good(matches):
    good = []  
    for m, n in matches:  
        if m.distance < 0.90 * n.distance:  
            good.append([m]) 
    return len(good)


########################################learning from database






Name = ['quad','quad1','double','double1']
Database = Data_create(Name)

Name = ['Test']
Test_data = Data_create(Name)

DB_processed=processed(Database)

Test_processed=processed(Test_data)
#########################
DB_1={}
TEST_1={}
for num,i in enumerate(DB_processed):
        List = []
        for j in range(len(DB_processed[i])):
            gray = DB_processed[i][j] 
            
            new = cv2.resize(gray,(gray.shape[0]*2,gray.shape[1]*2))
            new[new>=1]=255
            List.append(new)
        DB_1[i]=List
        
for num,i in enumerate(Test_processed):
        List = []
        for j in range(len(Test_processed[i])):
            gray = Test_processed[i][j] 
            new = cv2.resize(gray,(gray.shape[0]*2,gray.shape[1]*2))
            new[new>=1]=255
            List.append(new)
        TEST_1[i]=List        
        

#########################



DB_feature = {}


for num,i in enumerate( DB_1):
    List = []
    for j in range(len( DB_1[i])):
        List.append(siftlearn( DB_1[i][j]))
    DB_feature[i]=List
  
List = Test(TEST_1, DB_1)   

##################show the feature detection
'''
gray = Test_processed['Test'][0]
gray = Test_data['Test'][0]
finaltest=siftlearn(gray)

gray=cv2.resize(gray,(120,54))
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
_,des = sift.compute(gray,kp)  
kp_image = cv2.drawKeypoints(gray,kp,None)

cv2.imshow('kp',kp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

'''




for j in range(len(DB_processed['double'])):
    cv2.imshow('a',DB_processed['double'][j])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for j in range(len(Test_processed['Test'])):
    cv2.imshow('a',Test_processed['Test'][j])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''





