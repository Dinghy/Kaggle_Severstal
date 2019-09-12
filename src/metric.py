import numpy as np


def dice_metric(A, B):
	# Numpy version   
	if A.shape[-1] == 4:
		A = np.moveaxis(A, -1, 0)
	if B.shape[-1] == 4:
		B = np.moveaxis(B, -1, 0)	
	if len(A.shape) == 3:
		A = A[np.newaxis,:]
	if len(B.shape) == 3:
		B = B[np.newaxis,:]
	batch_size, channel_num = A.shape[0], A.shape[1]
	metric = 0.0
	
	for batch in range(batch_size):
		for ch in range(channel_num):
			t, p = A[batch, ch, :, :], B[batch, ch, :, :]
			true = np.sum(t)
			pred = np.sum(p)

			# deal with empty mask first
			if true == 0:
				metric += (pred == 0)
				continue

			# non empty mask case.  Union is never empty 
			# hence it is safe to divide by its number of pixels
			intersection = np.sum(t * p)
			dice = (2*intersection + 1e-8)/(true + pred + 1e-8)
			metric += dice
	return metric