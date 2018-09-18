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
def new_img_ann(line, filename):
    
    img = {'object':[]}
    img['filename'] = filename
            
    image = cv2.imread(img['filename'])
    img['height'], img['width'] = image.shape[:2]
    return img
                


def parse_lables (ann, img_folder):
    all_imgs = []
    seen_labels = {}
    
    with open(ann) as f:
        lines = f.readlines()
        
        
    img = new_img_ann(sorted(lines)[0], img_folder + sorted(lines)[0].split()[0] + '.pgm')
    
    for line in sorted(lines):
        split_line = line.split()
            
        filename = img_folder + split_line[0]  + '.pgm' 
        
        
    #         if the image is new - create a new one, in not - add to the exsisting one
        if filename != img['filename']:
           
            if len(img['object']) > 0:
                all_imgs += [img]
            
            img = new_img_ann(line,  filename)
            

        
    #            read the box annotation
        obj = {}
        obj['xmin'] = int(round(float(split_line[1])))
        obj['ymin'] = int(round(float(split_line[2])))
        
        obj['box_width'] = int(round(float(split_line[3])))
        obj['box_height'] = int(round(float(split_line[4])))
        
        obj['xmax'] = obj['xmin'] + obj['box_width']
        obj['ymax'] = obj['ymin'] + obj['box_height']
        
        obj['name'] = split_line[5]
        
        img['object'] += [obj]
        
    #           prev = img # previous object filename
        
        
        
        if obj['name'] in seen_labels:
            seen_labels[obj['name']] += 1
        else:
            seen_labels[obj['name']] = 1
    
    
    return all_imgs, seen_labels



    
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
        
        obj['name'] = split_line[5]
        
        img['object'] += [obj]
        
    #            prev = img # previous object filename
        
        
        
        if obj['name'] in seen_labels:
            seen_labels[obj['name']] += 1
        else:
            seen_labels[obj['name']] = 1
    
    
       
    
    return all_imgs, seen_labels






def draw_boxes(ann,img_folder):
    
    img = cv2.imread(img_folder+ann['filename'])


#     image_h, image_w, _ = img.shape

    for box in ann['object']:
       

        cv2.rectangle(img, (box['xmin'],box['ymin']), (box['xmax'],box['ymax']), (155,120*int(box['label']),0), 3)
       
        
    return img         


# split into training and validation 
def split_data (images, val_proportion):
    
    val_inx = random.sample(xrange(len(images)), int(round(len(images)*val_proportion)))

    val_images = [images[i] for i  in val_inx]
    train_images = [images[i] for i in xrange(len(images)) if i not in val_inx]
    
    return train_images, val_images