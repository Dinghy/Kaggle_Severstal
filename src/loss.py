import torch
import torch.nn.functional as F
import numpy as np


from loss_lovasz import lovasz_hinge, lovasz_softmax, flatten_binary_scores, lovasz_hinge_flat, mean
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def criterion_weightedBCE(logit, truth, weight = 0, use_weight = True):
	'Weighted BCE loss'
	h, w = 2, 3
	# use weighted function
	if use_weight:
		pos_weight = np.array([1./0.4, 1./0.1, 3./0.6, 1./0.4]).astype(np.float32)
		neg_weight = np.array([1./1.6, 1./1.9, 3./1.4, 1./1.6]).astype(np.float32) 
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
	pos_weight = np.array([1./0.4, 1./0.1, 1./0.6, 1./0.4]).astype(np.float32)
	neg_weight = np.array([1./1.6, 1./1.9, 1./1.4, 1./1.6]).astype(np.float32)
	weight = (truth == 1).float() * torch.from_numpy(pos_weight).to(device)+\
		 (truth == 0).float() * torch.from_numpy(neg_weight).to(device)
	res = F.binary_cross_entropy_with_logits(pred, truth, reduction='none')
	return (weight*res).mean()


def criterion_wmse(pred, truth):
	'Weighted mse loss'
	pos_weight = np.array([1./0.4, 1./0.1, 1./0.6, 1./0.4]).astype(np.float32)
	neg_weight = np.array([1./1.6, 1./1.9, 1./1.4, 1./1.6]).astype(np.float32)
	
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


def criterion_lovasz_hinge(logit, truth, weight = 0.2):
    'Lovasz loss for four channels: Need to change the shape of images if it is for other tasks'
    logit = logit.contiguous().view(-1, 256, 1600)
    truth = truth.contiguous().view(-1, 256, 1600)
    
    loss = lovasz_hinge(logit, truth, weight)
    return loss


def criterion_wlovasz_hinge(logit, truth, weight = 0.2):
    '''Lovasz loss for four channels: Need to change the shape of images if it is for other tasks
       weighted version
    '''
        # Category 1: Mean 0.9688, True Area[Neg,0.0000; Pos,0.0127], Pred Dice[Neg,0.9942; Pos,0.5755], Dice Diff[Neg,11.000; Pos,51.791]
	# Category 2: Mean 0.9913, True Area[Neg,0.0000; Pos,0.0093], Pred Dice[Neg,0.9995; Pos,0.5980], Dice Diff[Neg,1.000; Pos,16.483]
	# Category 3: Mean 0.8523, True Area[Neg,0.0000; Pos,0.0634], Pred Dice[Neg,0.9596; Pos,0.7055], Dice Diff[Neg,47.000; Pos,250.050]
	# Category 4: Mean 0.9797, True Area[Neg,0.0000; Pos,0.0870], Pred Dice[Neg,0.9973; Pos,0.7267], Dice Diff[Neg,5.000; Pos,35.806]
    bn, ch = logit.shape[:2]
    h, w = 2, 3
    
    pos_weight = np.array([1, 1, 1, 1]).astype(np.float32)
    neg_weight = np.array([0.2, 0.2, 0.2, 0.2]).astype(np.float32)
    weight = (truth.sum([h, w]) > 0).float() * torch.from_numpy(pos_weight).to(device)+\
             (truth.sum([h, w]) == 0).float() * torch.from_numpy(neg_weight).to(device)
    
    loss = mean(weight[i,j]*lovasz_hinge_flat(*flatten_binary_scores(logit[i,j].unsqueeze(0), truth[i,j].unsqueeze(0))) 
                   for j in range(ch) for i in range(bn))
    return loss


def criterion_wbce_dice(logit, truth, weight = 0):
	return criterion_weightedBCE(logit, truth, use_weight = True) + criterion_dice(logit, truth)


def criterion_wbce_lovasz(logit, truth, weight):
	return criterion_weightedBCE(logit, truth, use_weight = True) + criterion_lovasz_hinge(logit, truth, weight)


