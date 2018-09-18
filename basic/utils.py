#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import os
import tensorflow as tf
import copy
import cv2


def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[2]:

def aug_img(train_instance):
    path = train_instance['filename']
    all_obj = copy.deepcopy(train_instance['object'][:])
    img = cv2.imread(path)
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    t  = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t)

    img = img * (1 + t)
    img = img / (255. * 2.)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img[:,:,::-1]

    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(NORM_W) / w)
            obj[attr] = max(min(obj[attr], NORM_W), 0)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(NORM_H) / h)
            obj[attr] = max(min(obj[attr], NORM_H), 0)

        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = NORM_W - obj['xmax']
            obj['xmax'] = NORM_W - xmin

    return img[:,:,1].reshape((img.shape[0],img.shape[1],1)), all_obj





def data_gen(img_anns, batch_size):
    num_img = len(img_anns)
    shuffled_indices = np.random.permutation(np.arange(num_img))
    l_bound = 0
    r_bound = batch_size if batch_size < num_img else num_img

    while True:
        if l_bound == r_bound:
            l_bound  = 0
            r_bound = batch_size if batch_size < num_img else num_img
            shuffled_indices = np.random.permutation(np.arange(num_img))

        batch_size = r_bound - l_bound
        currt_inst = 0
        x_batch = np.zeros((batch_size, IMAGE_W, IMAGE_H, 1))
        y_batch = np.zeros((batch_size, GRID_W, GRID_H, BOX, 5+CLASS))

        for index in shuffled_indices[l_bound:r_bound]:
            train_instance = img_anns[index]

            # augment input image and fix object's position and size
            img, all_obj = aug_img(train_instance)
            #for obj in all_obj:
            #    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (1,1,0), 3)
            #plt.imshow(img); plt.show()

            # construct output from object's position and size
            for obj in all_obj:
                box = []
                center_x = .5*(obj['xmin'] + obj['xmax']) #xmin, xmax
                center_x = center_x / (float(IMAGE_W) / GRID_W)
                center_y = .5*(obj['ymin'] + obj['ymax']) #ymin, ymax
                center_y = center_y / (float(IMAGE_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_idx = int(obj['label'])-1
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

                    y_batch[currt_inst, grid_y, grid_x, :, 0:4]        = BOX * [box]
                    y_batch[currt_inst, grid_y, grid_x, :, 4  ]        = BOX * [1.]
                    y_batch[currt_inst, grid_y, grid_x, :, 5: ]        = BOX * [[0.]*CLASS]
                    y_batch[currt_inst, grid_y, grid_x, :, 5+obj_idx] = 1.0

            # concatenate batch input from the image
            x_batch[currt_inst] = img
            currt_inst += 1

            del img, all_obj

        yield x_batch, y_batch

        l_bound  = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_img: r_bound = num_img


# In[ ]:




