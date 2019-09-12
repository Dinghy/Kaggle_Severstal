# check the data
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from albumentations import (Compose, Flip, HorizontalFlip, Normalize, 
	RandomBrightness, RandomContrast, RandomGamma, OneOf, ToFloat, 
	RandomSizedCrop, ShiftScaleRotate)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SteelDataset
from unet import Unet
from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask, analyze_labels, seed_everything, print2file


def evaluate_batch(labels, outputs, threshold = 0.5):
	labels  = labels.detach().cpu().numpy()
	outputs = (torch.sigmoid(outputs).detach().cpu().numpy() > threshold).astype(int)
	return dice_metric(labels, outputs)


def evaluate_loader(net, device, criterion, dataloader):
	loss, dice = 0, 0
	with torch.no_grad():
		for data in dataloader:
			images, labels = data[0].to(device), data[1].to(device)
			images = images.permute(0, 3, 1, 2)
			labels = labels.permute(0, 3, 1, 2)
			outputs = net(images)
			loss += criterion(outputs, labels).item()			   
			dice += evaluate_batch(labels, outputs)
	return dice, loss


def evaluate_loader_post(net, device, dataloader, args, bplot = True):
	net.eval()
	dicPred = {'Class '+str(classid+1):[] for classid in range(4)}
	dice, preds = 0.0, []
	iplot = 0
	fig, axs = plt.subplots(args.batch, 2, figsize=(16,16))

	with torch.no_grad():
		for data in tqdm(dataloader):
			images, labels = data[0], data[1]
			for image_raw, label_raw in zip(images, labels):
				# flip and predict
				output_merge = predict_flip(image_raw, net, device)
				# append the result
				if args.bayes_opt:
					preds.append(output_merge)
				# using simple threshold and output the result
				output_thres = post_process(output_merge)
				# record the predicted labels
				for j in range(4):
					dicPred['Class {:d}'.format(j+1)].append(output_thres[:,:,j].sum()/args.height/args.width)
				
				dice += dice_metric(label_raw.detach().numpy(), output_thres)
				# plot
				if iplot < args.batch:
					ax = axs[iplot, 0]
					plot_mask(image_raw.detach().numpy(), output_thres, ax)
					ax.axis('off')

					ax = axs[iplot, 1]
					plot_mask(image_raw.detach().numpy(), label_raw.detach().numpy(), ax)
					ax.axis('off')
					iplot += 1
			# save some memory
			if len(preds) > 480:
				break
	plt.savefig('../output/evaluate_image.png')
	return dice, dicPred


def train_net(net, criterion, optimizer, device, args):
	# output information
	history = {'Train_loss':[], 'Train_dice':[], 'Valid_loss':[], 'Valid_dice':[]}
	
	# main iteration
	for epoch in range(args.epoch):  # loop over the dataset multiple times
		net.train()
		running_loss, running_dice = 0.0, 0.0
		tk0 = tqdm(enumerate(trainloader), total = len(trainloader), leave = False)

		# zero the gradient
		optimizer.zero_grad()
		# iterate over all samples
		for i, data in tk0:
			# get the inputs; data is a list of [inputs, labels]
			images, labels = data[0].to(device), data[1].to(device)
			images = images.permute(0, 3, 1, 2)
			labels = labels.permute(0, 3, 1, 2)

			# forward + backward + optimize
			outputs = net(images)
			# accumulate the gradient
			if not args.accumulate:
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				batch_loss = loss.item()
			else:
				acc_step = 64//args.batch
				loss = criterion(outputs, labels)/acc_step
				loss.backward()
				if (i+1)%acc_step == 0:
					optimizer.step()
					optimizer.zero_grad()
				batch_loss = loss.item()*acc_step

			# print statistics
			batch_dice = evaluate_batch(labels, outputs)
			running_loss += batch_loss
			running_dice += batch_dice
			tk0.set_postfix(info = 'Loss {:.3f}, Dice {:.3f}'.format(batch_loss, batch_dice))

		# after every epoch, print the statistics
		net.eval()
		val_dice, val_loss = evaluate_loader(net, device, criterion, validloader)

		# update the learning rate
		# scheduler.step()

		# update the history and output message
		history['Train_loss'].append(running_loss/len(trainloader))
		history['Valid_loss'].append(val_loss / len(validloader))
		history['Train_dice'].append(running_dice / len(TRAIN_FILES) / 4) # four categories
		history['Valid_dice'].append(val_dice / len(VALID_FILES) / 4)	 # four categories
		print('Epoch {:d} '.format(epoch)+' '.join(key+':{:.3f}'.format(val[-1]) for key,val in history.items()))

	return history


def post_process(pred, thres_seg = 0.5, size_seg = 100):
	# TTA: thresholding
	assert(pred.shape[2] == 4)
	for j in range(4):
		pred[:,:,j] = pred[:,:,j] > thres_seg   
		# TTA: combining classification and size thresholding
		nsize = pred[:,:,j].sum()
		if nsize < size_seg:
			pred[:,:,j] *= 0  
	return pred


def predict_flip(image_raw, net, device):
	output_merge = np.zeros((args.height, args.width, 4))
	for i in range(4):
		lr, ud = divmod(i, 2)
		image = image_raw.detach().numpy()
		if lr == 1: # flip left to right
			image = np.fliplr(image)
		if ud == 1: # flip up to down
			image = np.flipud(image)

		image_flip = torch.from_numpy(image.copy()).unsqueeze(0).to(device)
		# obtain the prediction
		outputs = torch.sigmoid(net(image_flip.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).detach().cpu().numpy()[0]

		# flip the predicted results
		if lr == 1: # flip the prediction from right to left
			outputs = np.fliplr(outputs)
		if ud == 1:
			outputs = np.flipud(outputs)
		# merge the result
		output_merge += outputs/4
	return output_merge



if __name__ == '__main__':
	# argsparse
	class args:
		test_run  = True	  # do a test run
		eda	   = True	  # run eda from start
		epoch	 = 3		 # the number of epochs
		augment   = False	 # whether using augmentations 
		normalize = False	 # whether do normalization on the data
		height	= 256
		width	 = 1600
		batch	 = 8
		model	 = 'resnet34'
		accumulate= True
		bayes_opt = False
		load_mod  = False

	# input folder paths
	TRAIN_PATH  = '../input/severstal-steel-defect-detection/train_images/'
	TEST_PATH   = '../input/severstal-steel-defect-detection/test_images/'
	TRAIN_MASKS = '../input/severstal-steel-defect-detection/train.csv'
	MODEL_LOAD_PATH = '../input/severstal-model/model.pth'
	
	# ouput folder paths
	VALID_ID_FILE = '../output/validID.csv'
	MODEL_FILE   = '../output/model.pth'
	HISTORY_FILE = '../output/history.csv'
	LOG_FILE     = '../output/log.txt'

	# find all files in the directory
	TRAIN_FILES_ALL = sorted(glob.glob(TRAIN_PATH+'*.jpg'))
	TEST_FILES  = sorted(glob.glob(TEST_PATH+'*.jpg'))
	print(len(TRAIN_FILES_ALL), len(TEST_FILES))
	print(len(os.listdir(TRAIN_PATH)))
	########################################################################
	# Train test split
	if args.eda:
		# read in the masks
		mask_df = pd.read_csv(TRAIN_MASKS).set_index(['ImageId_ClassId']).fillna('-1')
		print(mask_df.head())
		# get train and test for statistics
		steel_ds	   = SteelDataset(TRAIN_FILES_ALL, mask_df = mask_df)
		steel_ds_test  = SteelDataset(TEST_FILES)

		# get the statistics of the images
		if args.test_run:
			rows = 100
		else:
			rows = float('inf')

		# there is a deviation in mean in this data set.
		stat_df_test = steel_ds_test.stat_images(rows)
		stat_df = steel_ds.stat_images(rows)
		train_mean, train_std = stat_df['mean'].mean(), stat_df['std'].std()
		test_mean, test_std   = stat_df_test['mean'].mean(), stat_df_test['std'].std()

		# get the labels from the data
		labels = list(stat_df.apply(lambda x:int(x['Class 1'] != 0) + \
											int(x['Class 2'] != 0) + \
											int(x['Class 3'] != 0) + \
											int(x['Class 4'] != 0) != 0, axis=1))
		# save the statistics
		print(stat_df.shape, len(labels))
		X_train, X_valid, _, _ = train_test_split(np.arange(stat_df.shape[0]), labels, test_size = 0.16, random_state = 1234)
		valid_df = pd.DataFrame({'Valid':X_valid})
		valid_df.to_csv(VALID_ID_FILE)
		# print statistics
		sout =  '========   Train Stat ==========\n' + analyze_labels(stat_df.iloc[X_train,:]) +\
				'========Validation Stat ==========\n' + analyze_labels(stat_df.iloc[X_valid,:])
		print2file(sout, LOG_FILE)
		# plot the distributions
		fig, axs = plt.subplots(1,2, figsize=(16,5))
		sns.distplot(stat_df['mean'], ax=axs[0], kde_kws={"label": "Train"}); axs[0].set_title('Distribution of mean');
		sns.distplot(stat_df['std'] , ax=axs[1], kde_kws={"label": "Train"}); axs[1].set_title('Distribution of std');
		sns.distplot(stat_df_test['mean'], ax=axs[0], kde_kws={"label": "Test"}); 
		sns.distplot(stat_df_test['std'] , ax=axs[1], kde_kws={"label": "Test"}); 
		plt.savefig('../output/Distribution.png')
	else: # load previous result
		train_mean, train_std = 0.3438812517320016, 0.056746666005067205
		test_mean, test_std = 0.25951299299868136, 0.051800296725619116
		# load validation id
		X_valid = list(pd.read_csv(VALID_ID_FILE)['Valid'])
		X_train = list(set(np.arange(TRAIN_FILES)) - set(X_valid))


	# not using sophisticated normalize
	if not args.normalize:
		train_mean, train_std = 0, 1
		test_mean, test_std = 0, 1
		
	# get the train and valid files
	TRAIN_FILES = [TRAIN_FILES_ALL[i] for i in X_train]
	VALID_FILES = [TRAIN_FILES_ALL[i] for i in X_valid]
	

	sout = 'Train/Test {:d}/{:d}\n'.format(len(TRAIN_FILES_ALL), len(TEST_FILES)) + \
			'Train mean/std {:.3f}/{:.3f}\n'.format(train_mean, train_std) + \
			'Test mean/std {:.3f}/{:.3f}\n'.format(test_mean, test_std) +\
			'Train num/sample {:d}'.format(len(TRAIN_FILES)) + ' '.join(TRAIN_FILES[:2]) + \
			'\nValid num/sample {:d}'.format(len(VALID_FILES)) + ' '.join(VALID_FILES[:2])
	print2file(sout, LOG_FILE)

	########################################################################
	# Augmentations
	augment_train = Compose([
		Flip(p=0.5),   # Flip vertically or horizontally or both
		ShiftScaleRotate(rotate_limit = 5, p = 0.25),
		Normalize(mean = (train_mean, train_mean, train_mean), std  = (train_std, train_std, train_std)),
		ToFloat(max_value=1.)
	],p=1)

	# validation
	augment_valid = Compose([
		Normalize(mean=(train_mean, train_mean, train_mean), std=(train_std, train_std, train_std)),
		ToFloat(max_value=1.)],p=1)

	# normal prediction
	augment_test = Compose([
		Normalize(mean=(test_mean, test_mean, test_mean), std=(test_std, test_std, test_std)),
		ToFloat(max_value=1.)],p=1)

	# do some simple checking
	if args.test_run:
		# check rle2mask and mask2rle
		mask_df = pd.read_csv(TRAIN_MASKS).set_index(['ImageId_ClassId']).fillna('-1')
		for i, pixel in enumerate(mask_df['EncodedPixels']):
			if pixel != '-1':
				rle_pass = mask2rle(rle2mask(pixel, 1600, 256))
				if rle_pass != pixel:
					print(i)
					
		# check dataloader			
		steel_ds = SteelDataset(TRAIN_FILES, mask_df = mask_df)
		steel_ds_train = SteelDataset(TRAIN_FILES, mask_df = mask_df, augment = augment_train)
		steel_ds_valid = SteelDataset(VALID_FILES, mask_df = mask_df, augment = augment_valid)
		image, mask = steel_ds_train[1]
		print(image.shape, image.min(), image.max())
		print(mask.shape, mask.min(), mask.max())
		image, mask = steel_ds_valid[1]
		print(image.shape, image.min(), image.max())
		print(mask.shape, mask.min(), mask.max())

		# check on the images
		nplot = 4
		fig, axs = plt.subplots(nplot, 2, figsize=(16,nplot*2))
		for i in range(nplot):
			ax = axs[divmod(i, 2)]
			ax.axis('off')
			plot_mask(*steel_ds[i], ax)

			ax = axs[divmod(i + nplot, 2)]
			plot_mask(*steel_ds_train[i], ax)
			ax.axis('off')
		plt.savefig('../output/Dataset_augment.png')

	########################################################################
	# Prepare dataset -> dataloader
	# creat the data set
	steel_ds_train = SteelDataset(TRAIN_FILES, mask_df = mask_df, augment = augment_train)
	steel_ds_valid = SteelDataset(VALID_FILES, mask_df = mask_df, augment = augment_valid)	
		
	# create the dataloader
	trainloader = torch.utils.data.DataLoader(steel_ds_train, batch_size = args.batch, shuffle = True, num_workers = 4)
	validloader = torch.utils.data.DataLoader(steel_ds_valid, batch_size = args.batch, shuffle = False, num_workers = 4)

	# cpu or gpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# checking the dataloader
	if args.test_run:
		data = next(iter(trainloader))
		inputs, labels = data[0].to(device), data[1].to(device)
		print(inputs.shape, inputs.min(), inputs.max(), inputs.dtype)
		print(labels.shape, labels.min(), labels.max(), labels.dtype)

	########################################################################
	# Model
	if args.model == 'resnet34':
		net = Unet("resnet34", encoder_weights="imagenet", classes = 4, activation = None).to(device)  # pass model specification to the resnet32
	else:
		raise NotImplementedError

	########################################################################
	# Define a Loss function and optimizer
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(net.parameters(), lr = 0.001)
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch//2, args.epoch*3//4], gamma = 0.5)

	########################################################################
	# Train the network
	if args.load_mod:
		history = {'Train_loss':[], 'Train_dice':[], 'Valid_loss':[], 'Valid_dice':[]}
		net.load_state_dict(torch.load(MODEL_LOAD_PATH))
	else:
		history = train_net(net, criterion, optimizer, device, args)
	
	# save the final result
	print('Finished Training')
	history_df = pd.DataFrame(history)
	history_df.to_csv(HISTORY_FILE)
	torch.save(net.state_dict(),MODEL_FILE)

	# show the curve
	fig, axs = plt.subplots(1,2,figsize=(16,4))
	axs[0].plot(history['Train_loss'], label = 'Train Loss')
	axs[0].plot(history['Valid_loss'], label = 'Valid Loss')
	axs[0].legend();axs[0].grid()
	axs[0].set_title('Loss')

	axs[1].plot(history['Train_dice'], label = 'Train Dice')
	axs[1].plot(history['Valid_dice'], label = 'Valid Dice')
	axs[1].legend();axs[1].grid()
	axs[1].set_title('Dice')
	plt.savefig('../output/loss_dice.png')

	########################################################################
	# Evaluate the network
	# get all predictions of the validation set: maybe a memory error here.
	dice, dicPred = evaluate_loader_post(net, device, validloader, args)

	# evaluate the prediction
	sout = 'Final Dice {:.3f}\n'.format(dice/len(VALID_FILES)/4) +\
			'==============Predict===============\n' + \
			analyze_labels(pd.DataFrame(dicPred)) +\
			'==============True===============\n' + \
			analyze_labels(stat_df.iloc[X_valid,:])
	print2file(sout, LOG_FILE)
