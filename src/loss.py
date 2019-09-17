import torch
import torch.nn.functional as F
import numpy as np
# position of the height, width dimension

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def criterion_weightedBCE(logit, truth, use_weight = True):
	'Weighted BCE loss'
	h, w = 2, 3
	# use weighted function
	if use_weight:
		pos_weight = np.array([1./0.4, 1./0.2, 1., 1./0.4]).astype(np.float32)
		neg_weight = np.array([1./1.6, 1./1.8, 1., 1./1.6]).astype(np.float32) 
	# unweighted version
	else:
		pos_weight = np.array([1., 1., 1., 1.]).astype(np.float32) 
		neg_weight = np.array([1., 1., 1., 1.]).astype(np.float32) 
		
	weight = (truth.sum([h, w]) > 0).float() * torch.from_numpy(pos_weight).to(device)+\
			 (truth.sum([h, w]) == 0).float() * torch.from_numpy(neg_weight).to(device)
	res = F.binary_cross_entropy_with_logits(logit, truth, reduction='none').mean([h, w])
	return (weight*res).mean()


def criterion_wbce(pred, truth):
	'Weighted BCE loss'
	# use weighted function
	pos_weight = np.array([1./0.4, 1./0.2, 1., 1./0.4]).astype(np.float32)
	neg_weight = np.array([1./1.6, 1./1.8, 1., 1./1.6]).astype(np.float32)
	weight = (truth == 1).float() * torch.from_numpy(pos_weight).to(device)+\
		 (truth == 0).float() * torch.from_numpy(neg_weight).to(device)
	res = F.binary_cross_entropy_with_logits(pred, truth, reduction='none')
	return (weight*res).mean()


def criterion_wmse(pred, truth):
	'Weighted mse loss'
	pos_weight = np.array([1./0.4, 1./0.2, 1., 1./0.4]).astype(np.float32)
	neg_weight = np.array([1./1.6, 1./1.8, 1., 1./1.6]).astype(np.float32)
	
	weight = (truth == 1).float() * torch.from_numpy(pos_weight).to(device)+\
		 (truth == 0).float() * torch.from_numpy(neg_weight).to(device)
	res = (pred-truth) ** 2
	return (weight * res).mean()


def criterion_dice(logit, truth):
	'Dice loss'
	smooth = 1
	prob  = torch.sigmoid(logit)
	intersection = (prob * truth).sum()
	union =  (prob * prob).sum() + (truth * truth).sum()
	dice  = 1 - 2*(intersection + smooth)/(union + smooth)
	return dice


def criterion_wbce_dice(logit, truth, use_weight = True):
	return criterion_weightedBCE(logit, truth, use_weight = True) + criterion_dice(logit, truth)

