#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import imutils
import random
import json
import os
from shutil import copyfile








# parse a string of annotation 
def new_img_ann(line, img_folder , filename):
    
    img = {'object':[]}
    img['filename'] = filename
            
    image = cv2.imread(img_folder + img['filename']+ '.pgm')
    img['width'], img['height'] = image.shape[:2]
    return img
        


# In[6]:


def parse_lables (ann, img_folder, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    with open(ann) as f:
        lines = f.readlines()
        
        
    img = new_img_ann(sorted(lines)[0], img_folder , sorted(lines)[0].split()[0])
#     prev = img
    
    for line in sorted(lines):
        split_line = line.split()
            
        filename =  split_line[0] 
        
        
#         if the image is new - create a new one, in not - add to the exsisting one
        if filename != img['filename']:
           
            if len(img['object']) > 0:
                all_imgs += [img]
            
            img = new_img_ann(line, img_folder , filename)
            
            
            
            
        
        
#         read the box annotation
        obj = {}
        obj['xmin'] = int(round(float(split_line[1])))
        obj['ymin'] = int(round(float(split_line[2])))
        
        obj['box_height'] = int(round(float(split_line[3])))
        obj['box_width'] = int(round(float(split_line[4])))
        
        obj['xmax'] = int(round(float(split_line[3]))) + int(round(float(split_line[1])))
        obj['ymax'] = int(round(float(split_line[4]))) + int(round(float(split_line[2])))
        
        obj['label'] = split_line[5]
        
        img['object'] += [obj]
        
#         prev = img # previous object filename
        
        
        
        if obj['label'] in seen_labels:
            seen_labels[obj['label']] += 1
        else:
            seen_labels[obj['label']] = 1
    
    
       
    
    return all_imgs, seen_labels




# In[ ]:


def draw_boxes(ann,img_folder):
    
    img = cv2.imread(img_folder+ann['filename'])


#     image_h, image_w, _ = img.shape

    for box in ann['object']:
       

        cv2.rectangle(img, (box['xmin'],box['ymin']), (box['xmax'],box['ymax']), (155,120*int(box['label']),0), 3)
       
        
    return img         


# In[ ]:


# tmp = images[idx]
# # img = cv2.imread(tmp['filename'])

# # plt.imshow(img) 
# plt.imshow(draw_boxes(tmp,img_folder))




# import os
# # rotate and save images into corresponding 
# def split_data(img_folder, train_images, val_images):
    
#     val_dir = 'data/val/'
#     os.makedirs(val_dir)
    
#     for i in val_images:
#         img =  cv2.imread(img_folder + i['filename'] +'.pgm')
# #         img =  cv2.imread()
# #         copyfile(img_folder + i['filename'], dst)
#         cv2.imwrite( val_dir + i['filename'] , img);
    
#     train_dir = 'data/train/'
#     os.makedirs(train_dir)
    
#     for i in val_images:
# #          img = open(img_folder + i['filename'], 'wb')
#         img =  cv2.imread(img_folder + i['filename'] + '.pgm')
#         cv2.imwrite( train_dir + i['filename'], img);
    
    
#     return 0
    
    


# In[ ]:


# split_data(img_folder, train_images, val_images)


# In[ ]:


# rotate the images  90 degrees

 
# # read image as grey scale
# img = cv2.imread('data/depth/seq0_0000_0.pgm')

# plt.imshow(img) 
# plt.title('original')

# Draw a diagonal blue line with thickness of 5 px
# boximg = cv2.rectangle(img,(-1,5),(-1+517,5+258),(0,255,0),3)
# plt.imshow(boximg)
    
    
# cv2.drawContours(img,[box],0,(0,0,255),2)


# # get image height, width
# (h, w) = img.shape[:2]

# # calculate the center of the image
# center = (w / 2, h / 2)
 



# In[ ]:





# In[ ]:


# Perform the counter clockwise rotation holding at the center
# 90 degrees
# M = cv2.getRotationMatrix2D(center, 90, 1.0)
# rotated90 = cv2.warpAffine(img, M, (h, w))
 
# cv2.imwrite( "data/rotated.png", rotated90 );

# plt.imshow(rotated90) 
# plt.title('rotated')


# In[ ]:


# rotated = imutils.rotate_bound(img, 270)


# plt.imshow(rotated) 
# plt.title('rotated')


# # In[ ]:


# # cv2.imshow('Original Image',img)
# # cv2.waitKey(0) # waits until a key is pressed
# # cv2.destroyAllWindows() # destroys the window showing image
 
# cv2.imshow('Image rotated by 90 degrees',rotated90)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image
 


# # In[ ]:


# f = open(annot,'r')
# while True:
#     text = f.readline()
#     if 'rawr' in text:
#         print text


# # In[ ]:


# from os import listdir
# from os.path import isfile, join
# all = [f for f in listdir('data/')]


# # In[ ]:


# print onlyfiles


# In[ ]:


# randomly split to  test/validation making sure we have all the labels

