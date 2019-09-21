import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, sampler

from tqdm import tqdm
from utils import rle2mask


def random_crop_shift_pad(image, mask, p = 0.4):
	# randomly crop a part from the image in width direction
	# randomly shift in the width direction
	# pad the empty part with zero
	height, width = image.shape[:2]
	if np.random.uniform() >= p:
		return image, mask
	
	image_sum = np.sum(image, axis = (0,2))/3
	pos = np.where(image_sum <= 10)[0] # black positions
	if len(pos) <= 5: # full image
		beg, end = 0, width-1
	elif pos[0] != 0: # image in the front
		beg, end = 0, pos[0]-1 
	else:			 # image in the back
		beg, end = pos[-1]+1, width-1
	
	if end-beg+1 >= 800:
		dx   = np.random.randint(200, end-beg+1)  # width of the crop
		image_new, mask_new = np.random.uniform(size = image.shape)*0.01, np.zeros(mask.shape)
		if np.random.uniform() >= 0.5:  # cat to left
			image_new[:, :dx] = image[:,beg:beg+dx]
			mask_new[:, :dx] = mask[:,beg:beg+dx]
		else:						   # cat to right
			image_new[:, -dx:] = image[:,beg:beg+dx]
			mask_new[:, -dx:] = mask[:,beg:beg+dx]
		return image_new, mask_new
	else:
		return image, mask


class SteelDataset(Dataset):
	def __init__(self, fpaths ,args,
					 mask_df  = None,
					 height   = 256,
					 width	= 1600,
					 channel  = 3,
					 category = 4,
					 augment  = None):
		self.mask_df = mask_df
		self.fpaths  = fpaths
		# basic parameters
		self.category = category
		self.height, self.width, self.channel = height, width, channel
		# augmentations
		self.augment = augment
		self.args = args
		self.augTag = self.augment is not None and len([item for item in self.augment]) > 2 and self.args.augment == 2	
	
	def __getitem__(self, idx):
		'get one image along with its masks on four categories'
		fpath = self.fpaths[idx]
		fname = fpath.split('/')[-1]
		# get the image
		image = cv2.imread(fpath)
		mask  = np.zeros((self.height, self.width, self.category))
		# if this is for training
		if self.mask_df is not None:
			# get the masks
			for classid in range(self.category):
				rle = self.mask_df.loc[fname + '_{:d}'.format(classid + 1), 'EncodedPixels']
				mask[:,:,classid] = rle2mask(rle, self.width, self.height)
		image, mask = np.uint8(image), np.uint8(mask)

		# if there is augmentation
		if self.augment is not None:
			augmented = self.augment(image = image, mask = mask)
			image, mask = augmented['image'], augmented['mask']
			# do additional augmentations for training data set
			if self.augTag:
				image, mask = random_crop_shift_pad(image, mask)

		# do simple normalization
		else:
			image, mask = image/255, mask
		
		# do regression task
		if self.args.output == 1:
			area = np.array([self.stat_mask(mask[:,:,j]) for j in range(self.category)]).astype(np.float32) 
			return image.astype(np.float32), mask.astype(np.float32), area
		# do classification task
		elif self.args.output == 2:
			area = np.array([self.stat_mask(mask[:,:,j])>0 for j in range(self.category)]).astype(np.float32)
			return image.astype(np.float32), mask.astype(np.float32), area
		# vanilla version (0,3)
		else: 	
			return image.astype(np.float32), mask.astype(np.float32)
	
	
	def __len__(self):
		return len(self.fpaths)

		
	def stat_mask(self, mask):
		return np.sum(mask)/self.height/self.width 
	

	def stat_images(self, rows = float('inf')):
		'generate a file storing all inforamtion'
		# record the ratio
		dicStat = {**{'ImageId':[], 'mean':[], 'std':[]}, 
				   **{'Class '+str(classid+1):[] for classid in range(self.category)}} # ratios for the images
		for i in tqdm(range(min(rows, len(self.fpaths)))):
			fpath = self.fpaths[i]
			fname = fpath.split('/')[-1]
			if self.args.output == 0:
				image, mask = self.__getitem__(i)
			else:
				image, mask, _ = self.__getitem__(i)
			dicStat['ImageId'].append(fname)
			dicStat['mean'].append(np.mean(image))
			dicStat['std'].append(np.std(image))
			for j in range(self.category):
				dicStat['Class '+str(j+1)].append(self.stat_mask(mask[:,:,j]))
		self.stat_df = pd.DataFrame(dicStat)
		return self.stat_df
