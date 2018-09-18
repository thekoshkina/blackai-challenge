
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
import numpy as np

import pickle, json, copy, cv2
from preprocessing import parse_lables, new_img_ann, split_data

from generator import DataGenerator


# import labels

global IMAGE_H, IMAGE_W, THRESHOLD #, SCALE_NOOB, SCALE_OBJECT, SCALE_COOR, SCALE_CLASS




# In[2]:


exec(open("./utils.py").read())


# In[3]:


LABELS = ['1', '2']
# IMAGE_H, IMAGE_W = 416, 416
IMAGE_H, IMAGE_W =  480,640
GRID_H, GRID_W = 15 , 20
BOX              = 5
CLASS            = len(LABELS)

CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

THRESHOLD = 0.3
# OBJ_THRESHOLD    = 0.3#0.5
# CLASS_THRESHOLD    = 0.3#0.45
# ANCHORS          = [149,74, 194,97, 282,141, 392,196, 511,255]
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]



SCALE_NOOB  = 1.0
SCALE_OBJECT     = 5.0
SCALE_COOR      = 1.0
SCALE_CLASS     = 1.0

BATCH_SIZE       = 16
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50


# In[4]:


# wt_path = 'yolo.weights'                      
img_dir = 'data/depth/'
labels = 'data/labels.txt'




# In[5]:


imgs, seen_labels = parse_lables (labels, img_dir)

# split into training and validation 
train_imgs, val_imgs = split_data (imgs, 0.2)
    
 ## write parsed annotations to pickle for fast retrieval next time
with open('train_imgs', 'wb') as fp:
    pickle.dump(train_imgs, fp)
# write parsed annotations to pickle for fast retrieval next time
with open('val_imgs', 'wb') as fp:
    pickle.dump(val_imgs, fp)

   


# In[6]:


# ## read saved pickle of parsed annotations
# with open ('val_imgs', 'rb') as fp:
#     val_imgs = pickle.load(fp)

# ## read saved pickle of parsed annotations
# with open ('train_imgs', 'rb') as fp:
#     train_imgs = pickle.load(fp)


# In[7]:


# print train_imgs[1]

# image = cv2.imread(train_imgs[1]['filename'])
# print  image.shape

# plt.imshow(image)

# tmp= image[:,:,1]
# plt.imshow(tmp)
# print  tmp.shape


# In[8]:


input_image = Input(shape=(IMAGE_H, IMAGE_W, 1))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))


# In[9]:


# Layer 1
x = Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False)(input_image)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


# Layer 2 - 5
for i in range(0,4):
    x = Conv2D((32*(2**i)), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
# Layer 6
x = Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7 - 8
for _ in range(0,2):
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

# Layer 9
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
x = Activation('linear')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

output = Lambda(lambda args: args[0])([output, true_boxes])


# In[10]:


model = Model([input_image, true_boxes], output)


# In[11]:


# model.summary()


# In[12]:


# connecting_layer = model.layers[-4].output

# top_model = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal') (connecting_layer)
# top_model = Activation('linear') (top_model)
# top_model = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)) (top_model)


# In[13]:


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_H), [GRID_W]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,1,2,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    
    true_box_conf = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * SCALE_COOR
    
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * SCALE_NOOB
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * SCALE_OBJECT
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * SCALE_CLASS       
    
    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < SCALE_COOR/2.)
    seen = tf.assign_add(seen, 1.)
    
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """    
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    
    return loss


# In[14]:


# def custom_loss(y_true, y_pred):
#     ### Adjust prediction
#     # adjust x and y      
#     pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
#     # adjust w and h
#     pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
#     pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))
    
#     # adjust confidence
#     pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
#     # adjust probability
#     pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    
#     y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
#     print("Y_pred shape: {}".format(y_pred.shape))
    
#     ### Adjust ground truth
#     # adjust x and y
#     center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
#     center_xy = center_xy / np.reshape([(float(IMAGE_W)/GRID_W), (float(IMAGE_H)/GRID_H)], [1,1,1,1,2])
#     true_box_xy = center_xy - tf.floor(center_xy)
    
#     # adjust w and h
#     true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
#     true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(IMAGE_W), float(IMAGE_H)], [1,1,1,1,2]))
    
#     # adjust confidence
#     pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
#     pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
#     pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
#     pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
#     true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
#     true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
#     true_box_ul = true_box_xy - 0.5 * true_tem_wh
#     true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
#     intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
#     intersect_br = tf.minimum(pred_box_bd, true_box_bd)
#     intersect_wh = intersect_br - intersect_ul
#     intersect_wh = tf.maximum(intersect_wh, 0.0)
#     intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
#     iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
#     best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
#     best_box = tf.to_float(best_box)
#     true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    
#     # adjust confidence
#     true_box_prob = y_true[:,:,:,:,5:]
    
#     y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
#     print("Y_true shape: {}".format(y_true.shape))
#     #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
    
#     ### Compute the weights
#     weight_coor = tf.concat(4 * [true_box_conf], 4)
#     weight_coor = SCALE_COOR * weight_coor
    
#     weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_OBJECT * true_box_conf
    
#     weight_prob = tf.concat(CLASS * [true_box_conf], 4) 
#     weight_prob = SCALE_CLASS * weight_prob 
    
#     weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
#     print("Weight shape: {}".format(weight.shape))
    
#     ### Finalize the loss
#     loss = tf.pow(y_pred - y_true, 2)
#     loss = loss * weight
#     loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS)])
#     loss = tf.reduce_sum(loss, 1)
#     loss = .5 * tf.reduce_mean(loss)
    
#     return loss


# In[15]:


generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}


# In[16]:


def normalize(image):
    return image / 255.


# In[17]:


# train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)
# val_batch = BatchGenerator(val_imgs, generator_config, norm=normalize)


# In[18]:


# Generators
train_gen = DataGenerator(train_imgs, generator_config)
val_gen = DataGenerator(val_imgs, generator_config)


# In[19]:


early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_anchors.hdf5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


# In[20]:


if not os.path.exists('./logs'): os.makedirs('./logs')
    
tb_counter  = len([log for log in os.listdir(os.path.expanduser('./logs/')) if 'depth_' in log]) + 1


# In[21]:


tensorboard = TensorBoard(log_dir=os.path.expanduser('.logs/') + 'depth_' + '_' + str(tb_counter), 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False)


# In[22]:


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)


# In[23]:


config = tf.ConfigProto(device_count={"CPU": 16})
K.tensorflow_backend.set_session(tf.Session(config=config))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


model.compile(loss=custom_loss, optimizer=optimizer)


# In[ ]:


model.fit_generator(generator        = train_gen, 
                    steps_per_epoch  = len(train_gen), 
                    epochs           = 100, 
                    verbose          = 1,
                    validation_data  = val_gen,
                    validation_steps = len(val_gen),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)

