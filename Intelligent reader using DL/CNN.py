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
import h5py
import tensorflow as tf

import keras

import cv2

from matplotlib import pyplot as plt

from keras.datasets import mnist

from keras.utils import to_categorical

from keras import layers 

from keras import optimizers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input



def Data_create(Name):
    Database = {}     
    for i in range(len(Name)):
        name = [] 
        for filename in os.listdir(r'Database/%s'%(Name[i])):
            name.append(filename) 
        if '.DS_Store' in name:
            name.remove('.DS_Store')
        img_data = []
        for j in range(len(name)):
            img_data.append(cv2.cvtColor(cv2.imread(r'Database/%s/%s'%(Name[i],name[j])), cv2.COLOR_BGR2GRAY))            
        Database['%s'%(Name[i])]=img_data
    return Database

def processed(List):
    New_list = []
    for [pic,name] in List:
        New_pic = np.ones(pic.shape)
        New_pic = cv2.resize(pic,(50,50),interpolation = cv2.INTER_CUBIC)
        if 'eighth' in name:
            Label = 0
        if 'half' in name:
            Label = 1       
        if 'quarter' in name:
            Label = 2
        New_list.append([New_pic,Label])
    return New_list


def Seg_processed(Database):
    DB_processed = {}
    for num,i in enumerate(Database):
        List = []
        for j in range(len(Database['%s'%(i)])):
            L= Seg(Database['%s'%(i)][j])

            List.append(L)


        DB_processed['%s'%(i)] = List
    return DB_processed

def Seg(gray):
    b_g = binary(gray)
    (gray_new,index) = area_detect(b_g)
    final = closing(gray_new,index)
    return final
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








'''
=======================
Import dataset 
=======================

'''

Name = ['eighth_note','half_note','quarter_note']  #change the folders here to learn the different note
Database = Data_create(Name)
Database = Seg_processed(Database)



Train_data = []
for name in Database:
    for pic in Database[name]:
        Train_data.append([pic,name])
np.random.shuffle(Train_data)
Train =  processed(Train_data)




Name = ['test eighth_note','test quarter_note','test half_note']                             #Test note should put in the TEST folder
Test_database = Data_create(Name)
Test_database = Seg_processed(Test_database)

Test_data = []
for name in Test_database:
    for pic in Test_database[name]:
        Test_data.append([pic,name])
np.random.shuffle( Test_data)
Test =  processed( Test_data)   

'''
=======================
Create .h5 files to store them
=======================

'''
num_train = len(Train)
X_train = np.zeros([num_train,50,50])
Y_train = np.zeros([num_train,1])
for num,[pic,label] in enumerate(Train):
    X_train[num,:,:] = pic
    Y_train[num,0] = label
X_train = np.expand_dims(X_train,axis = -1)

num_test = len(Test)
X_test = np.zeros([num_test,50,50])
Y_test = np.zeros([num_test,1])
for num,[pic,label] in enumerate(Test):
    X_test[num,:,:] = pic
    Y_test[num,0] = label
X_test= np.expand_dims(X_test,axis = -1)


f = h5py.File('data.h5', 'w')
f.create_dataset("X_train",data = X_train)
f.create_dataset("X_test",data = X_test)
f.create_dataset("Y_train",data = Y_train)
f.create_dataset("Y_test",data = Y_test)
f.close()

'''
=======================
Import processed data
=======================
'''
file=h5py.File('data.h5','r')
X_train =file['X_train'][:]
X_test =file['X_test'][:]
Y_train =file['Y_train'][:]
Y_test =file['Y_test'][:]
file.close()
'''
=======================
Begin to create deep neural network
=======================
'''

m,n_h,n_w,n_c = X_train.shape
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
Y_train = to_categorical(Y_train) #test = tf.one_hot(train_labels,depth  = 3)
Y_test = to_categorical(Y_test)

def NoteModel(input_shape):

    X_input = Input(input_shape)
    
    X = ZeroPadding2D((2,2))(X_input) #tf.pad(tensor, paddings, mode='CONSTANT', name=None)

    X = Conv2D(32,(5,5),strides = (1,1), name = 'Conv0')(X) # tf.nn.conv2D(input,filter,strides,padding,name)
    
    X = BatchNormalization(axis = -1, name = 'bn0')(X) #tf.layers.batch_normalization(inputs,axis = -1, nimentum = 0.99,epsilon = 0.001,name)
    
    X = Activation('relu')(X) # tf.nn.relu(features,name)
        
    X = MaxPooling2D((3,3), name = 'Max_pool0')(X)

    X = ZeroPadding2D((3,3))(X) #tf.pad(tensor, paddings, mode='CONSTANT', name=None)
    
    X = Conv2D(64,(7,7),strides = (1,1), padding = 'valid',name = 'Conv1')(X) # tf.nn.conv2D(input,filter,strides,padding,name)
    
    X = BatchNormalization(axis = -1, name = 'bn1')(X) #tf.layers.batch_normalization(inputs,axis = -1, nimentum = 0.99,epsilon = 0.001,name)
    
    X = Activation('relu')(X) # tf.nn.relu(features,name)
    
    
    X = Conv2D(128,(7,7),strides = (1,1), padding = 'valid',name = 'Conv2')(X) # tf.nn.conv2D(input,filter,strides,padding,name)
    
    X = BatchNormalization(axis = -1, name = 'bn2')(X) #tf.layers.batch_normalization(inputs,axis = -1, nimentum = 0.99,epsilon = 0.001,name)
    
    X = Activation('relu')(X) # tf.nn.relu(features,name)
 
    X = Flatten()(X) # tf.layers.flattten(input, name)
    
    X = Dense(3,activation = 'softmax',name = 'fc0', kernel_initializer='glorot_uniform')(X)  #tf.layers.dense(inputs, units,activation,kernal_initializer,name,)
    # kernel initializer 初始化： glorot_normal 为 Xavier 正态分布初始化
    
    model = Model(inputs = X_input, outputs = X, name = 'CNN')
    
    return model    

Note_recognize = NoteModel(X_train.shape[1:])

Note_recognize.summary()

adam =  optimizers.Adam(lr = 0.001)

Note_recognize.compile(optimizer = adam ,loss = 'mean_squared_error',metrics = ["accuracy"])

Note_recognize.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 64)

preds = Note_recognize.evaluate(x = X_test, y = Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

Note_recognize.save('Note_recognition.h5')

Note_recognize = load_model('Note_recognition.h5')







Sequence = ['eight','half','quarter']


Note_recognize.predict(np.expand_dims(X_test[0,:,:,:],axis = 0)) * 100
plt.imshow(X_test[0,:,:,0])



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



for j in range(len(DB_processed['double'])):
    cv2.imshow('a',DB_processed['double'][j])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for j in range(len(Test_processed['Test'])):
    cv2.imshow('a',Test_processed['Test'][j])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''





