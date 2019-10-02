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
from loss import loss_BCE, loss_dice, loss_lovasz, loss_BCE_dice, loss_BCE_lovasz


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


def compute_loss(args, outputs, data, acc_step = 1):
    'compute the loss function'
    # loss function, loss_BCE, loss_dice, loss_lovasz, loss_BCE_dice, loss_BCE_lovasz
    if args.loss == 0:
        criterion = nn.BCEWithLogitsLoss()

    # obtain the mask
    masks = data[1].to(device)
    if args.output == 1 or args.output == 2:
        labels = data[2].to(device)
    masks  = masks.permute(0, 3, 1, 2)

    # obtain the loss
    # different ways of handling the outputs
    if args.output == 0:
        if args.loss == 0:
            loss = criterion(outputs, masks) / acc_step
        elif args.loss == 1:
            loss = loss_lovasz(outputs, masks,\
                                weight = args.weight_other,\
                                symmetric = args.symmetric) / acc_step
        elif args.loss == 2:
            loss = loss_BCE_dice(outputs, masks, \
                                    weight_bce = args.weight_bce, \
                                    weight_dice = args.weight_other, \
                                    weight_mix = args.weight_mix) / acc_step
        elif args.loss == 3:
            loss = loss_BCE_lovasz(outputs, masks,\
                                    weight_bce = args.weight_bce, \
                                    weight_lovasz = args.weight_other, \
                                    weight_mix = args.weight_mix, \
                                    symmetric = args.symmetric) / acc_step
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return loss


if __name__ == '__main__':
    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_split',   action = 'store_false', default = True,    help = 'Rerun train/test split')
    parser.add_argument('--normalize',    action = 'store_true',  default = False,   help = 'Normalize the images or not')
    parser.add_argument('--accumulate',   action = 'store_false', default = True,    help = 'Not doing gradient accumulation or not')
    parser.add_argument('--bayes_opt',    action = 'store_true',  default = False,   help = 'Do Bayesian optimization in finding hyper-parameters')
    parser.add_argument('-l','--load_mod',action = 'store_true',  default = False,   help = 'Load a pre-trained model')
    parser.add_argument('-t','--test_run',action = 'store_true',  default = False,   help = 'Run the script quickly to check all functions')
    parser.add_argument('--sampler',      action = 'store_true',  default = False,   help = 'Using sampling mechanism in dataloader.')
    parser.add_argument('--symmetric',    action = 'store_true',  default = False,   help = 'Using symmetric loss in the Lovasz loss function.')
    
    parser.add_argument('--wlovasz',     type = float,default = 0.2,        help = 'The weight used in Lovasz loss')
    parser.add_argument('--augment',     type = int,  default = 0,          help = 'The type of train augmentations: 0 vanilla, 1 add contrast, 2 add  ')
    parser.add_argument('--loss',        type = int,  default = 0,          help = 'The loss: 0 BCE vanilla; 1 wbce+dice; 2 wbce+lovasz.')
    parser.add_argument('--sch',         type = int,  default = 0,          help = 'The schedule of the learning rate: 0 step; 1 cosine annealing; 2 cosine annealing with warmup.')    
    parser.add_argument('-m', '--model', type = str,  default = 'resnet34', help = 'The backbone network of the neural network.')
    parser.add_argument('-e', '--epoch', type = int,  default = 5,          help = 'The number of epochs in the training')
    parser.add_argument('--height',      type = int,  default = 256,        help = 'The height of the image')
    parser.add_argument('--width',       type = int,  default = 1600,       help = 'The width of the image')
    parser.add_argument('--category',    type = int,  default = 1,          help = 'The category of the problem')
    parser.add_argument('-b', '--batch', type = int,  default = 8,          help = 'The batch size of the training')
    parser.add_argument('-s','--swa',    type = int,  default = 4,          help = 'The number of epochs for stochastic weight averaging')
    parser.add_argument('-o','--output', type = int,  default = 0,          help = 'The type of the network, 0 vanilla, 1 add regression, 2 add classification.')
    parser.add_argument('--seed',        type = int,  default = 1234,       help = 'The random seed of the algorithm.')
    parser.add_argument('--spec_cat',    type = int,  default = 2,          help = 'The category of the mask.')
    parser.add_argument('--weight_mix',  type = float,  default = 1,        help = 'The mix ratio of two loss function.')
    parser.add_argument('--weight_bce',  type = float,  default = 0.5,        help = 'The mix ratio of two loss function.')
    parser.add_argument('--weight_other',type = float,  default = 0.5,        help = 'The mix ratio of two loss function.')
    args = parser.parse_args()

    ################################################################################################
    # weight_mix = 1, symmetric = False
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

    ################################################################################################
    # not using sophisticated normalize
    if not args.normalize:
        train_mean, train_std = 0, 1
        test_mean, test_std = 0, 1

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

    ################################################################################################
    # prepare the dataset
    # get the train and valid files
    nTrain = int(len(TRAIN_FILES_ALL)*0.8)
    TRAIN_FILES = TRAIN_FILES_ALL[:nTrain]
    VALID_FILES = TRAIN_FILES_ALL[nTrain:]
    if args.test_run:
        TRAIN_FILES = TRAIN_FILES[:32]
        VALID_FILES = VALID_FILES[:32]
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

    ################################################################################################
    # output regression information
    history = {'Train_loss':[], 'Train_dice':[], 'Train_other':[],  'Valid_loss':[], 'Valid_dice':[], 'Valid_other':[]}	
    # optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net 
    net = Unet("resnet34", encoder_weights="imagenet", classes = 1, activation = None, args = args).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    # scheduler
    if args.sch == 1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epoch//2, args.epoch*3//4], gamma = 0.4)
    elif args.sch == 2:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, 1.6e-4)

    # criterion
    criterion_seg, criterion_other = criterion[0], criterion[1]

    print('Training begin!!!!!')
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
            images = data[0].to(device)

            # forward + backward + optimize
            outputs = net(images)
            # do not accumulate the gradient
            if not args.accumulate:
                # different ways of handling the outputs
                loss = compute_loss(args, outputs, data, acc_step = 1)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = loss.item()
            # do accumulation
            else:
                acc_step = 64//args.batch
                loss = compute_loss(args, outputs, data, acc_step = acc_step)
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
        history['Train_dice'].append(running_dice   / len(TRAIN_FILES)/args.category) # four categories
        history['Valid_dice'].append(val_dice       / len(VALID_FILES)/args.category) 
        history['Train_other'].append(running_other / len(TRAIN_FILES)/args.category)
        history['Valid_other'].append(val_other     / len(VALID_FILES)/args.category) 
        sout = '\nEpoch {:d} :'.format(epoch)+' '.join(key+':{:.3f}'.format(val[-1]) for key,val in history.items())
        print(sout)

    print('Training finished!!!!')

    ################################################################################################
    # evaluate
    eva = EvaluateOneCategory(net, device, validloader, args, isTest=False)
    eva.search_parameter()
    dice = eva.predict_dataloader()

