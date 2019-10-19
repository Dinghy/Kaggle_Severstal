import glob
import os
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from albumentations import (Compose, Flip, HorizontalFlip, Normalize, 
	RandomBrightnessContrast, RandomBrightness, RandomContrast, RandomGamma, OneOf, ToFloat, 
	RandomSizedCrop, ShiftScaleRotate)
import copy
from bayes_opt import BayesianOptimization

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SteelDataset, InfiniteSampler
from unet import Unet
from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask, analyze_labels, seed_everything, print2file
from loss import criterion_wbce_dice, criterion_wbce_lovasz, criterion_wmse, criterion_wbce
from loss_vat import VATLoss
from evaluate import Evaluate, evaluate_batch, evaluate_loader


def train_net(net, criterion, optimizer, device, args, LOG_FILE, MODEL_FILE):
	# output regression information
	history = {'Train_loss':[], 'Train_dice':[], 'Train_other':[],  'Valid_loss':[], 'Valid_dice':[], 'Valid_other':[]}	

	# scheduler
	if args.sch == 1:
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epoch//2, args.epoch*3//4], gamma = 0.4)
	elif args.sch == 2:
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, 1.6e-4)

	# criterion
	criterion_seg, criterion_other = criterion[0], criterion[1]
	
	# VAT semi-supervised training
	if args.VAT:
		vat = VATLoss(xi = args.vat_xi)
		vat_iter = iter(vatloader)

	val_dice_best = -float('inf')
	# main iteration
	for epoch in range(args.epoch):  # loop over the dataset multiple times
		net.train()
		running_loss, running_dice, running_other = 0.0, 0.0, 0.0
		tk0 = tqdm(enumerate(trainloader), total = len(trainloader), leave = False)

		# zero the gradient
		optimizer.zero_grad()
		# iterate over all samples
		for i, data in tk0:
			# get the inputs; data is a list of [inputs, labels]
			images, masks = data[0].to(device), data[1].to(device)
			if args.output == 1 or args.output == 2:
				labels = data[2].to(device)
			images = images.permute(0, 3, 1, 2)
			masks  = masks.permute(0, 3, 1, 2)

			# vat loss, drawn from unlabeled data
			if args.VAT:
				images_vat = next(vat_iter)[0].to(device).permute(0, 3, 1, 2)
				loss_vat = vat(net, images_vat)
			else:
				loss_vat = 0

			# forward + backward + optimize
			outputs = net(images)
			# do not accumulate the gradient
			if not args.accumulate:
				# different ways of handling the outputs
				if args.output == 0:
					loss = criterion_seg(outputs, masks)
				elif args.output == 1 or args.output == 2:
					loss = criterion_seg(outputs[0], masks, weight = args.wlovasz) + 0.2 * criterion_other(outputs[1], labels)
				# add vat loss
				loss += loss_vat
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				batch_loss = loss.item()
			# do accumulation
			else:
				acc_step = 64//args.batch
				# different ways of handling the outputs
				if args.output == 0:
					loss = criterion_seg(outputs, masks)/acc_step
				elif args.output == 1 or args.output == 2:
					loss = (criterion_seg(outputs[0], masks, weight = args.wlovasz) + 0.2 * criterion_other(outputs[1], labels))/acc_step 
				# add vat loss
				loss += loss_vat
				loss.backward()
				if (i+1)%acc_step == 0:
					optimizer.step()
					optimizer.zero_grad()
				batch_loss = loss.item() * acc_step

			# print statistics
			batch_dice, batch_other  = evaluate_batch(data, outputs, args)
			running_loss += batch_loss
			running_dice += batch_dice
			running_other += batch_other
			tk0.set_postfix(info = 'Loss {:.3f}, Dice {:.3f}, Other {:.3f}'.format(batch_loss, batch_dice, batch_other))
		
		# stochastic weight averaging
		if args.swa > 0 and epoch >= args.epoch-args.swa:
			epoch_tmp = args.epoch-args.swa
			if epoch == epoch_tmp:
				net_swa = copy.deepcopy(net.state_dict())
			else:
				for key, val in net_swa.items():
					net_swa[key] = ((epoch-epoch_tmp)*val+net.state_dict()[key])/(epoch-epoch_tmp+1)   

		# after every epoch, print the statistics
		net.eval()
		val_loss, val_dice, val_other = evaluate_loader(net, device, criterion, validloader, args)
		
		# save the best up to now
		if val_dice > val_dice_best:
			print('Improving val_dice from {:.3f} to {:.3f}, saving the model'.format(val_dice_best/len(VALID_FILES)/args.category, val_dice/len(VALID_FILES)/args.category))
			val_dice_best = val_dice
			torch.save(net.state_dict(),MODEL_FILE)

		# update the learning rate
		if args.sch > 0:
			scheduler.step()

		# update the history and output message
		history['Train_loss'].append(running_loss   / len(trainloader))
		history['Valid_loss'].append(val_loss       / len(validloader))
		history['Train_dice'].append(running_dice   / len(TRAIN_FILES) / args.category) # four categories
		history['Valid_dice'].append(val_dice       / len(VALID_FILES) / args.category) 
		history['Train_other'].append(running_other / len(TRAIN_FILES) / args.category)
		history['Valid_other'].append(val_other     / len(VALID_FILES) / args.category) 
		sout = '\nEpoch {:d} :'.format(epoch)+' '.join(key+':{:.3f}'.format(val[-1]) for key,val in history.items())
		print2file(sout, LOG_FILE)
		print(sout)
	if args.swa > 0:
		return net_swa, history	
	else:
		return net.state_dict(), history


def concatenate_model_file(args, model_type):
	# concate a folder path with the model type
	dicSpec = {'dec_':args.decoder, 'm_':model_type, 'e_':args.epoch, 'norm_':args.normalize, 'sch_':args.sch, 'loss_':args.loss, 'out_':args.output, 'seed_':args.seed}
	strSpec = '_'.join(key+str(val) for key,val in dicSpec.items())
	return '../input/weights/model_swa_{:s}.pth'.format(strSpec)


if __name__ == '__main__':
	# argsparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_split',   action = 'store_false', default = True,    help = 'Rerun train/test split')
	parser.add_argument('--accumulate',   action = 'store_false', default = True,    help = 'Not doing gradient accumulation or not')
	parser.add_argument('--bayes_opt',    action = 'store_true',  default = False,   help = 'Do Bayesian optimization in finding hyper-parameters')
	parser.add_argument('-l','--load_mod',action = 'store_true',  default = False,   help = 'Load a pre-trained model')
	parser.add_argument('-t','--test_run',action = 'store_true',  default = False,   help = 'Run the script quickly to check all functions')
	parser.add_argument('--VAT',          action = 'store_true',  default = False,   help = 'Add VAT loss in the loss functions')
	parser.add_argument('--bo_fineTune',  action = 'store_true',  default = False,   help = 'Fine-Tuning BO result')

	parser.add_argument('--decoder',     type = str,  default = 'cbam_con', help = 'Structure in the Unet decoder')
	parser.add_argument('--normalize',   type = int,  default = 1,          help = 'Normalize the images or not')
	parser.add_argument('--wlovasz',     type = float,default = 0.2,        help = 'The weight used in Lovasz loss')
	parser.add_argument('--augment',     type = int,  default = 0,          help = 'The type of train augmentations: 0 vanilla, 1 add contrast, 2 add  ')
	parser.add_argument('--loss',        type = int,  default = 2,          help = 'The loss: 0 BCE vanilla; 1 wbce+dice; 2 wbce+lovasz.')
	parser.add_argument('--sch',         type = int,  default = 2,          help = 'The schedule of the learning rate: 0 step; 1 cosine annealing; 2 cosine annealing with warmup.')	
	parser.add_argument('-m', '--model', type = str,  default = 'resnet34', help = 'The backbone network of the neural network.')
	parser.add_argument('-e', '--epoch', type = int,  default = 5,          help = 'The number of epochs in the training')
	parser.add_argument('--height',      type = int,  default = 256,        help = 'The height of the image')
	parser.add_argument('--width',       type = int,  default = 1600,       help = 'The width of the image')
	parser.add_argument('--category',    type = int,  default = 4,          help = 'The category of the problem')
	parser.add_argument('-b', '--batch', type = int,  default = 8,          help = 'The batch size of the training')
	parser.add_argument('-s','--swa',    type = int,  default = 4,          help = 'The number of epochs for stochastic weight averaging')
	parser.add_argument('-o','--output', type = int,  default = 2,          help = 'The type of the network, 0 vanilla, 1 add regression, 2 add classification.')
	parser.add_argument('--seed',        type = int,  default = 1234,       help = 'The random seed of the algorithm.')
	parser.add_argument('--eva_method',  type = int,  default = 1,          help = 'The evaluation method in postprocessing: 0 thres/size; 1 thres/size/classify; 2 thres/size/classify/after')
	parser.add_argument('--vat_xi',      type = float,default = 0.01,       help = 'How much we modify each pixel in VAT')
	args = parser.parse_args()

	######################################################
	seed_everything(seed = args.seed)
	print('===========================')
	for key, val in vars(args).items():
		print('{}: {}'.format(key, val))
	print('===========================\n')

	# input folder paths
	TRAIN_PATH  = '../input/severstal-steel-defect-detection/train_images/'
	TEST_PATH   = '../input/severstal-steel-defect-detection/test_images/'
	TRAIN_MASKS = '../input/severstal-steel-defect-detection/train.csv'
	
	# ouput folder paths
	dicSpec = {'dec_':args.decoder, 'm_':args.model, 'e_':args.epoch, 'norm_':args.normalize, 'sch_':args.sch, 'loss_':args.loss, 'out_':args.output, 'seed_':args.seed}
	strSpec = '_'.join(key+str(val) for key,val in dicSpec.items())
	
	VALID_ID_FILE = '../output/validID_{:s}.csv'.format(strSpec)
	MODEL_FILE    = '../output/model_{:s}.pth'.format(strSpec)
	MODEL_SWA_FILE= '../output/model_swa_{:s}.pth'.format(strSpec)
	HISTORY_FILE  = '../output/history_{:s}.csv'.format(strSpec)
	LOG_FILE      = '../output/log_{:s}.txt'.format(strSpec)
	# rewrite the file if not load mod
	if not args.load_mod:
		with open(LOG_FILE, 'w') as fopen:
			fopen.write(strSpec+'\n')
	
	# find all files in the directory
	TRAIN_FILES_ALL = sorted(glob.glob(TRAIN_PATH+'*.jpg'))
	TEST_FILES      = sorted(glob.glob(TEST_PATH+'*.jpg'))
	
	# read in the masks
	mask_df = pd.read_csv(TRAIN_MASKS).set_index(['ImageId_ClassId']).fillna('-1')
	print(mask_df.head())
	print('===========================\n')
	########################################################################
	# if test run a small version
	if args.test_run:
		rows = 100
		TEST_FILES = TEST_FILES[:100]
	else:
		rows = len(TRAIN_FILES_ALL)
		
	# Re run train test split
	if not args.load_split:
		# get train and test for statistics
		steel_ds       = SteelDataset(TRAIN_FILES_ALL, args,  mask_df = mask_df)
		steel_ds_test  = SteelDataset(TEST_FILES, args)

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
		X_train, X_valid, _, _ = train_test_split(np.arange(stat_df.shape[0]), labels, test_size = 0.16, random_state = 1234)
		valid_df = pd.DataFrame({'Valid':X_valid})
		valid_df.to_csv(VALID_ID_FILE)
		stat_df_valid = stat_df.iloc[X_valid,:]

		# print statistics
		sout =  '\n========   Train Stat ==========\n' + analyze_labels(stat_df.iloc[X_train,:]) +\
				'======== Validation Stat ==========\n' + analyze_labels(stat_df_valid)+'\n'
		print2file(sout, LOG_FILE)

		# plot the distributions
		fig, axs = plt.subplots(1,2, figsize=(16,5))
		sns.distplot(stat_df['mean'], ax=axs[0], kde_kws={"label": "Train"}); axs[0].set_title('Distribution of mean');
		sns.distplot(stat_df['std'] , ax=axs[1], kde_kws={"label": "Train"}); axs[1].set_title('Distribution of std');
		sns.distplot(stat_df_test['mean'], ax=axs[0], kde_kws={"label": "Test"}); 
		sns.distplot(stat_df_test['std'] , ax=axs[1], kde_kws={"label": "Test"}); 
		plt.savefig('../output/Distribution.png')

		# get the train and valid files
		TRAIN_FILES = [TRAIN_FILES_ALL[i] for i in X_train]
		VALID_FILES = [TRAIN_FILES_ALL[i] for i in X_valid]

	else: # load previous result
		train_mean, train_std = 0.3438812517320016, 0.056746666005067205
		test_mean, test_std = 0.25951299299868136, 0.051800296725619116
		# TO DO: change this part
		# load validation id 
		if args.bo_fineTune:
			valid_data_df = pd.read_csv('validID.csv')
		else:
			valid_data_df = pd.read_csv('../input/validID.csv')
		print(valid_data_df.head())
		X_valid = list(valid_data_df['Valid'])[:rows]
		X_train = list(set(np.arange(len(TRAIN_FILES_ALL))) - set(X_valid))[:rows]

		# get the train and valid files
		TRAIN_FILES = [TRAIN_FILES_ALL[i] for i in X_train]
		VALID_FILES = [TRAIN_FILES_ALL[i] for i in X_valid]

		steel_ds_valid = SteelDataset(VALID_FILES, args, mask_df = mask_df)
		stat_df_valid = steel_ds_valid.stat_images(rows)

		# print statistics
		sout =  '======== Validation Stat ==========\n' + analyze_labels(stat_df_valid)+'\n'
		print2file(sout, LOG_FILE)

	# not using sophisticated normalize
	#if args.normalize == 0:
	norm_mean, norm_std = (0,0,0), (1,1,1)

	sout = 'Train/Test {:d}/{:d}\n'.format(len(TRAIN_FILES_ALL), len(TEST_FILES)) + \
			'Train mean/std {:.3f}/{:.3f}\n'.format(train_mean, train_std) + \
			'Test mean/std {:.3f}/{:.3f}\n'.format(test_mean, test_std) +\
			'Train num/sample {:d}'.format(len(TRAIN_FILES)) + ' '.join(TRAIN_FILES[:2]) + \
			'\nValid num/sample {:d}'.format(len(VALID_FILES)) + ' '.join(VALID_FILES[:2])+'\n'
	print2file(sout, LOG_FILE)

	########################################################################
	# Augmentations
	augment_train = Compose([
		Flip(p=0.5),          # Flip vertically or horizontally or both
		RandomBrightnessContrast(p = 0.3),
		ShiftScaleRotate(rotate_limit = 10, p = 0.3),
		Normalize(mean = norm_mean, std  = norm_std),
		ToFloat(max_value=1.) # Divide pixel values by max_value to get a float32 output
	],p=1)

	# validation
	augment_valid = Compose([
		Normalize(mean = norm_mean, std = norm_std),
		ToFloat(max_value = 1.)],p = 1)

	# normal prediction
	augment_test = Compose([
		Normalize(mean = norm_mean, std = norm_std),
		ToFloat(max_value = 1.)],p = 1)


	########################################################################
	# do some simple checking
	if True:
		# check rle2mask and mask2rle
		# mask_df = pd.read_csv(TRAIN_MASKS).set_index(['ImageId_ClassId']).fillna('-1')
		#for i, pixel in enumerate(mask_df['EncodedPixels']):
		#	if pixel != '-1':
		#		rle_pass = mask2rle(rle2mask(pixel, 1600, 256))
		#		if rle_pass != pixel:
		#			print(i)
					
		# check dataloader			
		steel_ds = SteelDataset(TRAIN_FILES, args, mask_df = mask_df)
		steel_ds_train = SteelDataset(TRAIN_FILES, args, mask_df = mask_df, augment = augment_train)
		steel_ds_valid = SteelDataset(VALID_FILES, args, mask_df = mask_df, augment = augment_valid)
		res = steel_ds_train[1]
		image, mask = res[0], res[1]
		print(image.shape, image.min(), image.max())
		print(mask.shape, mask.min(), mask.max())
		res = steel_ds_valid[1]
		image, mask = res[0], res[1]
		print(image.shape, image.min(), image.max())
		print(mask.shape, mask.min(), mask.max())

		# check on the images
		nplot = 4
		fig, axs = plt.subplots(nplot, 2, figsize=(16,nplot*2))
		for i in range(nplot):
			ax = axs[divmod(i, 2)]
			ax.axis('off')
			plot_mask(*steel_ds[i][:2], ax)

			ax = axs[divmod(i + nplot, 2)]
			plot_mask(*steel_ds_train[i][:2], ax)
			ax.axis('off')
		plt.savefig('../output/Dataset_augment.png')


	########################################################################
	# Prepare dataset -> dataloader
	# creat the data set
	steel_ds_train = SteelDataset(TRAIN_FILES, args, mask_df = mask_df, augment = augment_train)
	steel_ds_valid = SteelDataset(VALID_FILES, args, mask_df = mask_df, augment = augment_valid)	
	steel_ds_test = SteelDataset(TEST_FILES, args, augment = augment_test)    
		
	# create the dataloader
	trainloader = torch.utils.data.DataLoader(steel_ds_train, batch_size = args.batch, shuffle = True, num_workers = 4)
	validloader = torch.utils.data.DataLoader(steel_ds_valid, batch_size = args.batch, shuffle = False, num_workers = 4)
	vatloader = torch.utils.data.DataLoader(steel_ds_test, batch_size = args.batch, sampler = InfiniteSampler(len(steel_ds_test)), num_workers = 4)
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
	if args.model == 'resnet34' or args.model == 'se_resnet50':
		net = Unet(args.model, encoder_weights="imagenet", classes = 4, activation = None, args = args).to(device)  # pass model specification to the resnet32
		if args.load_mod:
			# load the Unet model
			net.load_state_dict(torch.load(MODEL_SWA_FILE))
			# generate the same form of the input net
			nets = [net]
		
	elif args.model == 'ensemble':
		if args.load_mod:
			nets = []
			model_types = ['resnet34', 'se_resnet50']
			for model_type in model_types:
				# concatenate the file name
				model_file = concatenate_model_file(args, model_type)
				print(model_type, model_file)
				net = Unet(model_type, encoder_weights = None, classes = 4, activation = None, args = args).to(device)  # pass model specification to the resnet32
				net.load_state_dict(torch.load(model_file))
				nets.append(net)
		else:
			raise ValueError('Can not train the model in loading models for ensemble.')
	else:
		raise NotImplementedError
	

	########################################################################
	# Loss function
	if args.loss == 0:
		criterion = [nn.BCEWithLogitsLoss(), None]
	elif args.loss == 1:
		if args.output == 0:	
			criterion = [criterion_wbce_dice, None]
		elif args.output == 1:
			criterion = [criterion_wbce_dice, criterion_wmse]
		elif args.output == 2:
			criterion = [criterion_wbce_dice, criterion_wbce]
		else:
			raise NotImplementedError
	elif args.loss == 2:
		if args.output == 0:	
			criterion = [criterion_wbce_lovasz, None]
		elif args.output == 1:
			criterion = [criterion_wbce_lovasz, criterion_wmse]
		elif args.output == 2:
			criterion = [criterion_wbce_lovasz, criterion_wbce]
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError
	

	########################################################################
	# optimizer
	optimizer = optim.Adam(net.parameters(), lr = 0.001)
	
	########################################################################
	# Train the network
	seed_everything(seed = args.seed)
	if not args.load_mod:
		net_swa, history = train_net(net, criterion, optimizer, device, args, LOG_FILE, MODEL_FILE)
		torch.save(net_swa, MODEL_SWA_FILE)
		
		# save the final result
		print('Finished Training')
		history_df = pd.DataFrame(history)
		history_df.to_csv(HISTORY_FILE)
		# torch.save(net.state_dict(),MODEL_FILE)
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
	if args.load_mod and not args.bo_fineTune:
		
		# load swa model
		eva = Evaluate(nets, device, validloader, args, isTest = False)
		eva.search_parameter()
		dice, dicPred, dicSubmit, _ = eva.predict_dataloader(gen_pseudo = False)
		eva.plot_sampled_predict()
	
		# evaluate the prediction
		sout = '\n\nFinal SWA Dice {:.3f}\n'.format(dice) +\
			'==============SWA Predict===============\n' + \
                        analyze_labels(pd.DataFrame(dicPred)) + \
                        '==============True===============\n' + \
                        analyze_labels(stat_df_valid)
		
		print(sout)
		print2file(sout, LOG_FILE)
		print2file(','.join('"'+str(key)+'":'+str(val) for key,val in eva.dicPara.items()), LOG_FILE)
		
	
	# fine tune the result in ensemble
	if args.bo_fineTune and args.model == 'ensemble':
		dicPara = {'thres_seg1': 0.4533134987534817, 'size_seg1': 613.1016620220344, 'thres_oth1': 0.6794573692116359, 'size_oth1': 2067.734842164869,\
			   'thres_seg2': 0.4660851963381183, 'size_seg2': 1337.604976031266, 'thres_oth2': 0.1724603105855378, 'size_oth2': 3637.458108598964,\
			   'thres_seg3': 0.5082858551894693, 'size_seg3': 655.6238707406021, 'thres_oth3': 0.4241747771586643, 'size_oth3': 2178.811772705825,\
			   'thres_seg4': 0.4199609444062345, 'size_seg4': 1235.609034701938, 'thres_oth4': 0.5516533322432883, 'size_oth4': 2704.011274575476}
		
		eva = Evaluate(nets, device, validloader, args, isTest = False)
		eva.search_parameter_fine(dicPara)
		dice, dicPred, dicSubmit, _ = eva.predict_dataloader(gen_pseudo = False)

                # evaluate the prediction
		sout = '\n\nFinal SWA Dice {:.3f}\n'.format(dice) +\
			'==============SWA Predict===============\n' + \
			analyze_labels(pd.DataFrame(dicPred)) + \
			'==============True===============\n' + \
			analyze_labels(stat_df_valid)
		
		print(sout)
		print2file(sout, LOG_FILE)
		print2file(','.join('"'+str(key)+'":'+str(val) for key,val in eva.dicPara.items()), LOG_FILE)
		
		# generate the pseudo labeling data
		testloader = torch.utils.data.DataLoader(steel_ds_test, batch_size = args.batch, shuffle = False, num_workers = 4)
			
		eva_test = Evaluate(nets, device, testloader, args, dicPara = eva.dicPara, isTest = True)
		dice, dicPred, dicSubmit, dicPseudo = eva_test.predict_dataloader(gen_pseudo = True, to_rle = True, fnames = TEST_FILES)
		print(analyze_labels(pd.DataFrame(dicPred)))
		
		# output the pseudo label
		pseudo_df = pd.DataFrame(dicPseudo)
		print(pseudo_df.head())
		pseudo_df.to_csv('../output/Pseudo_{:s}.csv'.format(strSpec), index=False)
