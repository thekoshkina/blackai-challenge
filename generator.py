from keras.utils import Sequence
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt

# from utils import BoundBox
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
	x = x - np.max(x)
	
	if np.min(x) < t:
		x = x/np.min(x)*t
		
	e_x = np.exp(x)
	return e_x / e_x.sum(axis, keepdims=True)



def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b

	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3		  



def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
	
	intersect = intersect_w * intersect_h

	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	
	union = w1*h1 + w2*h2 - intersect
	
	return float(intersect) / union


class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		
		self.c	 = c
		self.classes = classes

		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
		
		return self.label
	
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
			
		return self.score


class DataGenerator(Sequence):
	
	def __init__(self, imgs,config,	shuffle=True,	norm=None):
		'Initialization'
		
		self.imgs = imgs
		self.config = config

		self.shuffle = shuffle
		self.norm		= norm
		self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

		self.shuffle = shuffle
		self.on_epoch_end()


	### augmentors by https://github.com/aleju/imgaug
	# sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		# self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.imgs)

	def load_image(self, i):
		'Loads an image'
		return cv2.imread(self.imgs[i]['filename'])

	

	def num_classes(self):
		return len(self.config['LABELS'])

	 

	def size(self):
		return len(self.imgs)		
	


	def load_annotation(self, i):
		anns = []

		for obj in self.imgs[i]['object']:
			ann = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
			anns += [ann]
		
		if len(anns) == 0: anns = [[]]

		return np.array(anns)

	

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(float(len(self.imgs))/self.config['BATCH_SIZE']))	 



	def aug_img(self, train_instance):
		path = train_instance['filename']
		all_obj = copy.deepcopy(train_instance['object'][:])
		img = cv2.imread(path)
		

		if img is None: print('Cannot find ', path)

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

		# # re-color
		# t	= [np.random.uniform()]
		# t += [np.random.uniform()]
		# t += [np.random.uniform()]
		# t = np.array(t)

		# img = img * (1 + t)
		# img = img / (255. * 2.)

		# # resize the image to standard size
		# img = cv2.resize(img, (self.config['IMAGE_H'], self.config['IMAGE_W']))
		# img = img[:,:,::-1]

		# fix object's position and size
		for obj in all_obj:
			for attr in ['xmin', 'xmax']:
				obj[attr] = int(obj[attr] * scale - offx)
				obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
				obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

			for attr in ['ymin', 'ymax']:
				obj[attr] = int(obj[attr] * scale - offy)
				obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
				obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

		if flip > 0.5:
			xmin = obj['xmin']
			obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
			obj['xmax'] = self.config['IMAGE_W'] - xmin

		return	img , all_obj	


	

	def __data_generation(self, l_bound, r_bound):

		'Generates data containing batch_size samples' 

		# Initialization
		instance_count = 0

		x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))	# input imgs
		b_batch = np.zeros((r_bound - l_bound, 1		 , 1		 , 1		,	self.config['TRUE_BOX_BUFFER'], 4))	 # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
		y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],	self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))								# desired network output

	 
		for train_instance in self.imgs[l_bound:r_bound]:
			# print train_instance
			# print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
			# augment input image and fix object's position and size
			img, all_obj = self.aug_img(train_instance)
			
			
			# construct output from object's x, y, w, h
			true_box_index = 0
			for obj in all_obj:
				
				if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
					
					center_x = .5*(obj['xmin'] + obj['xmax'])
					center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
					center_y = .5*(obj['ymin'] + obj['ymax'])
					center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])			

					grid_x = int(np.floor(center_x))
					grid_y = int(np.floor(center_y))	


				if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
					obj_indx	= self.config['LABELS'].index(obj['name'])
						# obj_indx = int(obj['label'])-1
						
					center_w = obj['box_width'] / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
					center_h = obj['box_height'] / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
					
					box = [center_x, center_y, center_w, center_h]	
					# print box									

					# find the anchor that best predicts this box
					best_anchor = -1
					max_iou		 = -1

					shifted_box = BoundBox(0, 0, center_w, center_h)
					
					for i in range(len(self.anchors)):
							anchor = self.anchors[i]
							iou		= bbox_iou(shifted_box, anchor)
							
							if max_iou < iou:
								best_anchor = i
								max_iou		 = iou
								
					# assign ground truth x, y, w, h, confidence and class probs to y_batch
					y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
					y_batch[instance_count, grid_y, grid_x, best_anchor, 4	] = 1.
					y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
					
					# print y_batch
					# print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"


					# assign the true box to b_batch
					b_batch[instance_count, 0, 0, 0, true_box_index] = box
					
					true_box_index += 1
					true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
			
			
			# print len(all_obj)
			# assign input image to x_batch
			if self.norm != None: 
				x_batch[instance_count] = self.norm(img[:,:,1].reshape((img.shape[0],img.shape[1],1)))
			else:
				# plot image and bounding boxes for sanity check
				# for obj in all_obj:
				# 	# print img.shape
				# 	if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
				# 		# tmp = img[:,:,0]
						# cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (155,120*int(obj['name']),0), 3)
						# cv2.putText(img, obj['name'], 
						# 			(obj['xmin']+2, obj['ymin']+12), 
						# 			0, 1.2e-3 * img.shape[0], 
						# 			(155,120*int(obj['name']),0), 2)
						# plt.imshow(img)
							
				x_batch[instance_count] = img[:,:,1].reshape((img.shape[0],img.shape[1],1)) # increase instance counter in current batch
				instance_count += 1  

		return [x_batch, b_batch], y_batch


	# def data_gen(self, l_bound, r_bound): 
	# 	return self.__data_generation(l_bound, r_bound)

	def __getitem__(self, idx):
		'Generate one batch of data'
		
		l_bound = idx*self.config['BATCH_SIZE']
		r_bound = (idx+1)*self.config['BATCH_SIZE']

		if r_bound > len(self.imgs):
			r_bound = len(self.imgs)
			l_bound = r_bound - self.config['BATCH_SIZE']
	 
		# # Generate data
		# X, y = self.__data_generation(list_IDs_temp)

		[x_batch, b_batch], y_batch = self.__data_generation(l_bound, r_bound)

		return [x_batch, b_batch], y_batch


