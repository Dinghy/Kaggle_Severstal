import numpy as np
import torch
import random
import os
import cv2

# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def rle2mask(mask_rle, width, height):
	'''
	mask_rle: run-length as string formated (start length)
	shape: (width,height) of array to return 
	Returns numpy array, 1 - mask, 0 - background
	'''
	shape = (width, height)
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	return img.reshape(shape).T

def mask2rle(img):
	'''
	img: numpy array, 1 - mask, 0 - background
	Returns run length as string formated
	'''
	pixels= img.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)


def plot_mask(image, mask, ax=None):
	'''
	plot the image and mask for one id(Int)
	Class: yellow 1; cyan 2; purple 3; red 4
	'''
	image, mask = np.uint8(image*255), np.uint8(mask*255)
	# plot one sample image
	if ax is None:
		fig, ax = plt.subplots(1,1, figsize = (16,3))
	palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
	for classid in range(4):
		mask_ = mask[:,:,classid]
		contours, _ = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		for contour in contours:
			cv2.polylines(image, contour, True, palet[classid], 2)
	ax.imshow(image)
	
	
def analyze_labels(stat_df, rows = float('inf')):
	'output label distribution information, require Class 1~4 column'
	# sanity check
	for i in range(4):
		if 'Class {:d}'.format(i+1) not in stat_df.columns:
			raise ValueError('Property not found in the input')
	# computation
	total_num = stat_df.shape[0]
	pos_ratio = [stat_df.apply(lambda x:x['Class {:d}'.format(i+1)] > 0, axis=1).sum()/total_num for i in range(4)]
	label_nums = [stat_df.apply(lambda x:int(x['Class 1'] > 0) + \
										int(x['Class 2'] > 0) + \
										int(x['Class 3'] > 0) + \
										int(x['Class 4'] > 0) == i, axis=1).sum() for i in range(5)]
	# output the information
	sres = 'Four labels ratio (1~4): {:.3f},{:.3f},{:.3f},{:.3f}\n'.format(*pos_ratio) + \
            'Label Num (Zero labels~four Labels): {:d},{:d},{:d},{:d},{:d}\n'.format(*label_nums) + \
            'Label ratio (Zero labels~four Labels): {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(*[item/total_num for item in label_nums])
	return sres


def seed_everything(seed=1234):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def print2file(text, filename):
    with open(filename, 'a') as fopen:
        fopen.write(text)

	
