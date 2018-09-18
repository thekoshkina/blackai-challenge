
# coding: utf-8

# tiny yolo 2 for single channel data trained on the provided data only 
# 
# git reposotories used: 
# 
# https://github.com/joycex99/tiny-yolo-keras/blob/master/Tiny%20Yolo%20Keras.ipynb
# 
# https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb

# In[1]:


from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import pickle
from utils import data_gen
from preprocessing import parse_lables, new_img_ann
import numpy as np
import json
import copy
import cv2
# import labels

global IMAGE_H, IMAGE_W
# import os, cv2
# from utils import BatchGenerator
# from utils import WeightReader, decode_netout, draw_boxes

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

get_ipython().magic(u'matplotlib inline')


# In[2]:


LABELS = ['1', '2']
# IMAGE_H, IMAGE_W = 416, 416
IMAGE_H, IMAGE_W =  480,640
GRID_H, GRID_W = 15 , 20
BOX              = 5
CLASS            = len(LABELS)

CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

TRESHOLD = 0.3
# OBJ_THRESHOLD    = 0.3#0.5
# CLASS_THRESHOLD    = 0.3#0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

SCALE_NOOB  = 1.0
SCALE_OBJECT     = 5.0
SCALE_COOR      = 1.0
SCALE_CLASS     = 1.0

BATCH_SIZE       = 16
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50


# In[3]:


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
#     img = cv2.resize(img, (IMAGE_H, IMAGE_W))
#     img = img[:,:,::-1]

    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(IMAGE_W) / w)
            obj[attr] = max(min(obj[attr], IMAGE_W), 0)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(IMAGE_H) / h)
            obj[attr] = max(min(obj[attr], IMAGE_H), 0)

        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = IMAGE_W - obj['xmax']
            obj['xmax'] = IMAGE_W - xmin

    return  img[:,:,1].reshape((image.shape[0],image.shape[1],1)) , all_obj


# In[4]:


# wt_path = 'yolo.weights'                      
img_dir = 'data/train/'
train_ann = 'data/train_ann.json'

val_img_dir = 'data/val/'
val_ann = 'data/val_ann.json'


# In[5]:


with open(train_ann) as f:
    anns = json.load(f)


# In[6]:


print anns[1]

image = cv2.imread(anns[1]['filename'])
print  image.shape

plt.imshow(image)

tmp= image[:,:,1]
plt.imshow(tmp)
print  tmp.shape


# In[7]:


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


# In[8]:


input_image = Input(shape=(IMAGE_H, IMAGE_W, 1))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))


# In[9]:


model = Sequential()

# Layer 1
model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(IMAGE_H, IMAGE_W,1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - 5
for i in range(0,4):
    model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

# Layer 7 - 8
for _ in range(0,2):
    model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

# Layer 9
model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
model.add(Activation('linear'))
model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))


# In[10]:


model.summary()


# In[11]:


# connecting_layer = model.layers[-4].output

# top_model = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal') (connecting_layer)
# top_model = Activation('linear') (top_model)
# top_model = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)) (top_model)


# In[12]:


def custom_loss(y_true, y_pred):
    ### Adjust prediction
    # adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    
    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    print("Y_pred shape: {}".format(y_pred.shape))
    
    ### Adjust ground truth
    # adjust x and y
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(IMAGE_W)/GRID_W), (float(IMAGE_H)/GRID_H)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)
    
    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(IMAGE_W), float(IMAGE_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    
    # adjust confidence
    true_box_prob = y_true[:,:,:,:,5:]
    
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    print("Y_true shape: {}".format(y_true.shape))
    #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
    
    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor
    
    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_OBJECT * true_box_conf
    
    weight_prob = tf.concat(CLASS * [true_box_conf], 4) 
    weight_prob = SCALE_CLASS * weight_prob 
    
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    print("Weight shape: {}".format(weight.shape))
    
    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    
    return loss


# In[13]:


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
        x_batch = np.zeros((batch_size, IMAGE_H, IMAGE_W, 1))
        y_batch = np.zeros((batch_size, GRID_H, GRID_W, BOX, 5+CLASS))

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


# In[14]:


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)


# In[15]:


sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=sgd)



model.fit_generator(generator = data_gen(anns, BATCH_SIZE), 
                    steps_per_epoch = int(len(anns)/BATCH_SIZE), 
                    epochs = 10, 
                    verbose = 2,
                    callbacks = [early_stop, checkpoint],
                    max_queue_size = 3)


# In[ ]:


# evaluate model using IoU (Intersection over union)


# In[ ]:


# def yolo_eval(val_anns,
#               anchors,
#               num_classes,
#               image_shape,
#               max_boxes=20,
#               score_threshold=.6,
#               iou_threshold=.5):
#     """Evaluate YOLO model on given input and return filtered boxes."""
#     num_layers = len(yolo_outputs)
#     input_shape = K.shape(yolo_outputs[0])[1:3] * 32
#     boxes = []
#     box_scores = []
#     for l in range(val_anns):
#         _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
#             anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
#         boxes.append(_boxes)
#         box_scores.append(_box_scores)
#     boxes = K.concatenate(boxes, axis=0)
#     box_scores = K.concatenate(box_scores, axis=0)

#     mask = box_scores >= score_threshold
#     max_boxes_tensor = K.constant(max_boxes, dtype='int32')
#     boxes_ = []
#     scores_ = []
#     classes_ = []
#     for c in range(num_classes):
#         # TODO: use keras backend instead of tf.
#         class_boxes = tf.boolean_mask(boxes, mask[:, c])
#         class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
#         nms_index = tf.image.non_max_suppression(
#             class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
#         class_boxes = K.gather(class_boxes, nms_index)
#         class_box_scores = K.gather(class_box_scores, nms_index)
#         classes = K.ones_like(class_box_scores, 'int32') * c
#         boxes_.append(class_boxes)
#         scores_.append(class_box_scores)
#         classes_.append(classes)
#     boxes_ = K.concatenate(boxes_, axis=0)
#     scores_ = K.concatenate(scores_, axis=0)
#     classes_ = K.concatenate(classes_, axis=0)

#     return boxes_, scores_, classes_

