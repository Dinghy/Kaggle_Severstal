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
from albumentations import (
    Compose, Flip, HorizontalFlip, Normalize, 
    RandomBrightnessContrast, RandomBrightness, RandomContrast, RandomGamma, OneOf, ToFloat, 
    RandomSizedCrop, ShiftScaleRotate)
import copy
from bayes_opt import BayesianOptimization

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, sampler
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
# sys.path.insert(0,'../input/severstal-model/')
from dataset import BalanceClassSampler, SteelDataset, SteelOneDataset
from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask, analyze_labels, seed_everything, print2file
from evaluate import Evaluate, EvaluateOneCategory
from unet import Unet


def evaluate_batch(data, outputs, args, threshold = 0.5):
    if args.output == 0:
        masks   = data[1].detach().cpu().numpy()
        pred_masks  = (torch.sigmoid(outputs).detach().cpu().numpy() > threshold).astype(int)
        # print(masks.shape, pred_masks.shape)
        return dice_metric(masks, pred_masks), 0.0
    elif args.output == 1:
        masks   = data[1].detach().cpu().numpy()
        labels  = data[2].detach().cpu().numpy()
        pred_masks  = (torch.sigmoid(outputs[0]).detach().cpu().numpy() > threshold).astype(int)
        pred_labels = outputs[1].detach().cpu().numpy()
        return dice_metric(masks, pred_masks), np.sum(np.sqrt((pred_labels-labels)**2))
    elif args.output == 2:  # classification
        masks   = data[1].detach().cpu().numpy()
        labels  = data[2].detach().cpu().numpy()
        pred_masks  = (torch.sigmoid(outputs[0]).detach().cpu().numpy() > threshold).astype(int)
        pred_labels = (torch.sigmoid(outputs[1]).detach().cpu().numpy() > threshold).astype(int)
        return dice_metric(masks, pred_masks), np.sum((pred_labels == labels).astype(int)) 


def evaluate_loader(net, device, criterion, dataloader, args):
    loss, dice, other = 0.0, 0.0, 0.0
    with torch.no_grad():
        for data in dataloader:
            images, masks = data[0].to(device), data[1].to(device)
            images = images.permute(0, 3, 1, 2)
            masks = masks.permute(0, 3, 1, 2)
            outputs = net(images)

            if args.output == 2:
                loss += 0.2*criterion[1](outputs[1], data[2].to(device)).item()
                loss += criterion[0](outputs[0], masks, weight = args.wlovasz).item()
            elif args.output == 0:
                loss += criterion[0](outputs, masks).item()

            res = evaluate_batch(data, outputs, args)
            dice, other = dice+res[0], other + res[1]
    return loss, dice, other


# argsparse
class args:
    test_run  = False      # do a test run
    eda       = True      # run eda from start
    epoch     = 3         # the number of epochs
    augment   = 2         # whether using augmentations 
    normalize = False     # whether do normalization on the data
    height    = 256
    width     = 1600
    batch     = 8
    model     = 'resnet34'
    accumulate= True
    bayes_opt = False
    load_mod  = True
    output    = 0
    sch       = 2
    swa       = 3 
    spec_cat  = 2
    wlovasz   = 0.2
    category  = 1
    seed      = 1234 
    loss      = 2
    sampler   = True

# folder paths
TRAIN_PATH  = '../input/severstal-steel-defect-detection/train_images/'
TEST_PATH   = '../input/severstal-steel-defect-detection/test_images/'
TRAIN_MASKS = '../input/severstal-steel-defect-detection/train.csv'

# find all files in the directory
TRAIN_FILES_ALL = sorted(glob.glob(TRAIN_PATH+'*.jpg'))
TEST_FILES  = sorted(glob.glob(TEST_PATH+'*.jpg'))

mask_df = pd.read_csv(TRAIN_MASKS).set_index(['ImageId_ClassId']).fillna('-1')                

# ouput folder paths
dicSpec = {'m_':args.model, 'e_':args.epoch, 'wl_':int(100*args.wlovasz), 'sch_':args.sch, 'loss_':args.loss, 'out_':args.output, 'seed_':args.seed}
strSpec = '_'.join(key+str(val) for key,val in dicSpec.items())
	
VALID_ID_FILE = '../output/validID_{:s}.csv'.format(strSpec)
MODEL_FILE    = '../output/model_{:s}.pth'.format(strSpec)
MODEL_SWA_FILE= '../output/model_swa_{:s}.pth'.format(strSpec)
HISTORY_FILE  = '../output/history_{:s}.csv'.format(strSpec)
LOG_FILE      = '../output/log_{:s}.txt'.format(strSpec)

# not using sophisticated normalize
if not args.normalize:
    train_mean, train_std = 0, 1
    test_mean, test_std = 0, 1
    
# get the train and valid files

nTrain = int(len(TRAIN_FILES_ALL)*0.8)
TRAIN_FILES = TRAIN_FILES_ALL[:nTrain]
VALID_FILES = TRAIN_FILES_ALL[nTrain:]
if args.test_run:
    TRAIN_FILES = TRAIN_FILES[:100]
    VALID_FILES = VALID_FILES[:100]

# train
augment_train = Compose([
    Flip(p=0.5),   # Flip vertically or horizontally or both
    RandomBrightnessContrast(p = 0.5),
    Normalize(mean = (train_mean, train_mean, train_mean), std  = (train_std, train_std, train_std)),
    ToFloat(max_value=1.)
],p=1)

# validation
augment_valid = Compose([
    Normalize(mean=(train_mean, train_mean, train_mean), std=(train_std, train_std, train_std)),
    ToFloat(max_value=1.)],p=1)

# test
augment_test = Compose([
    Normalize(mean=(test_mean, test_mean, test_mean), std=(test_std, test_std, test_std)),
    ToFloat(max_value=1.)],p=1)


# optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Unet("resnet34", encoder_weights="imagenet", classes = 1, activation = None, args = args).to(device)

optimizer = optim.Adam(net.parameters(), lr = 0.001)
criterion = [nn.BCEWithLogitsLoss(), None]

train_dataset = SteelOneDataset(TRAIN_FILES, args = args, mask_df = mask_df, augment = augment_train, spec_cat = args.spec_cat)
valid_dataset = SteelOneDataset(VALID_FILES, args = args, mask_df = mask_df, augment = augment_valid, spec_cat = args.spec_cat)

# create the dataloader
if args.sampler:
    train_sampler = BalanceClassSampler(train_dataset, len(train_dataset))
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size = args.batch, num_workers = 4,
                                               sampler = train_sampler, drop_last = True,)
else:
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, shuffle = True, num_workers = 4)

validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch, shuffle = False, num_workers = 4)

# output regression information
history = {'Train_loss':[], 'Train_dice':[], 'Train_other':[],  'Valid_loss':[], 'Valid_dice':[], 'Valid_other':[]}	

# scheduler
if args.sch == 1:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epoch//2, args.epoch*3//4], gamma = 0.4)
elif args.sch == 2:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, 1.6e-4)

# criterion
criterion_seg, criterion_other = criterion[0], criterion[1]

print('Training begin')
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

        # forward + backward + optimize
        outputs = net(images)
        # do not accumulate the gradient
        if not args.accumulate:
            # different ways of handling the outputs
            if args.output == 0:
                loss = criterion_seg(outputs, masks)
            elif args.output == 1 or args.output == 2:
                loss = criterion_seg(outputs[0], masks, weight = args.wlovasz) + 0.2 * criterion_other(outputs[1], labels)
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
            loss.backward()
            if (i+1)%acc_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_loss = loss.item() * acc_step
        
        # print(outputs.shape, outputs.min(), outputs.max())
        
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
    history['Train_dice'].append(running_dice   / len(TRAIN_FILES)/args.category) # four categories
    history['Valid_dice'].append(val_dice       / len(VALID_FILES)/args.category) 
    history['Train_other'].append(running_other / len(TRAIN_FILES)/args.category)
    history['Valid_other'].append(val_other     / len(VALID_FILES)/args.category) 
    sout = '\nEpoch {:d} :'.format(epoch)+' '.join(key+':{:.3f}'.format(val[-1]) for key,val in history.items())
    print(sout)


# evaluate
eva = EvaluateOneCategory(net, device, validloader, args, isTest=False)
eva.search_parameter()
dice = eva.predict_dataloader()
print('Final Dice', dice)

