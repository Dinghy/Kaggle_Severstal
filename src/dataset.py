import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, sampler

from tqdm import tqdm
from utils import rle2mask


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
		# do simple normalization
		else:
			image, mask = image/255, mask/255
		
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
