import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler

from tqdm import tqdm
from utils import rle2mask


def random_crop_shift_pad_old(image, mask, p = 0.3):
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
    else:             # image in the back
        beg, end = pos[-1]+1, width-1

    if end-beg+1 >= 800:
        dx   = np.random.randint(200, end-beg+1)  # width of the crop
        image_new, mask_new = np.random.uniform(size = image.shape)*0.01, np.zeros(mask.shape)
        if np.random.uniform() >= 0.5:
            image_new[:, :dx] = image[:,beg:beg+dx]
            mask_new[:, :dx] = mask[:,beg:beg+dx]
        else:
            image_new[:, -dx:] = image[:,beg:beg+dx]
            mask_new[:, -dx:] = mask[:,beg:beg+dx]
        return image_new, mask_new
    else:
        return image, mask


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
    else:             # image in the back
        beg, end = pos[-1]+1, width-1
    
    if end-beg+1 >= 800:
        nlimit = min(end-beg+1, width-300)
        dx   = np.random.randint(200, nlimit)     # width of the crop
        dbeg = np.random.randint(200, width-dx)   # dx <= width-300, width-dx >= 300
        image_new, mask_new = np.random.uniform(size = image.shape)*0.01, np.zeros(mask.shape)
        image_new[:, dbeg:dbeg+dx] = image[:,beg:beg+dx]
        mask_new[:, dbeg:dbeg+dx] = mask[:,beg:beg+dx]
        return image_new, mask_new
    else:
        return image, mask


class SteelDataset(Dataset):
    def __init__(self, fpaths ,args,
                     mask_df  = None,
                     height   = 256,
                     width    = 1600,
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
    	# get prepared for balanced labels
        if self.mask_df is not None:
            test1 = np.array([filepath.split('/')[-1]+'_{:d}'.format(1) for filepath in fpaths])
            test2 = np.array([filepath.split('/')[-1]+'_{:d}'.format(2) for filepath in fpaths])
            test3 = np.array([filepath.split('/')[-1]+'_{:d}'.format(3) for filepath in fpaths])
            test4 = np.array([filepath.split('/')[-1]+'_{:d}'.format(4) for filepath in fpaths])
            pos_flag1 =  np.array(self.mask_df.loc[test1,'EncodedPixels']!='-1')[:,np.newaxis]
            pos_flag2 =  np.array(self.mask_df.loc[test2,'EncodedPixels']!='-1')[:,np.newaxis]
            pos_flag3 =  np.array(self.mask_df.loc[test3,'EncodedPixels']!='-1')[:,np.newaxis]
            pos_flag4 =  np.array(self.mask_df.loc[test4,'EncodedPixels']!='-1')[:,np.newaxis] 

            arrtags = np.concatenate([pos_flag1, pos_flag2, pos_flag3, pos_flag4], axis = 1).astype(int)
            self.arrtags = np.concatenate([arrtags, (arrtags.sum(axis=1)==0)[:,np.newaxis].astype(int)], axis = 1)


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
                if not self.args.conservative:
                    image, mask = random_crop_shift_pad(image, mask)
                else:
                    image, mask = random_crop_shift_pad_old(image, mask)

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



class SteelOneDataset(Dataset):
    'Only extract the results for a specific category'
    def __init__(self, fpaths ,args,
                     mask_df  = None,
                     height   = 256,
                     width    = 1600,
                     channel  = 3,
                     category = 4,
                     augment  = None,
                     spec_cat = 2):
        
        self.mask_df = mask_df
        self.fpaths  = fpaths
        # basic parameters
        self.category = category
        self.spec_cat = spec_cat
        
        self.height, self.width, self.channel = height, width, channel
        # augmentations
        self.augment = augment
        self.args = args
        self.augTag = self.augment is not None and len([item for item in self.augment]) > 2 and self.args.augment == 2    
        # obtain the positive flag
        if mask_df is not None:
            test = np.array([filepath.split('/')[-1]+'_{:d}'.format(self.spec_cat+1) for filepath in fpaths])
            self.pos_flag = self.mask_df.loc[test,'EncodedPixels']!='-1'
        else:
            self.pos_flag = None


    def __getitem__(self, idx):
        'get one image along with its masks on four categories'
        fpath = self.fpaths[idx]
        fname = fpath.split('/')[-1]
        # get the image
        image = cv2.imread(fpath)
        mask  = np.zeros((self.height, self.width, 1))
        # if this is for training
        if self.mask_df is not None:
            # get the masks
            classid = self.spec_cat
            rle = self.mask_df.loc[fname + '_{:d}'.format(classid + 1), 'EncodedPixels']
            mask[:,:,0] = rle2mask(rle, self.width, self.height)
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
    
    
    @classmethod
    def plot_mask(cls, image, mask, ax=None):
        '''
        plot the image and mask for one id(Int)
        Class: yellow 1; cyan 2; purple 3; red 4
        '''
        image, mask = np.uint8(image*255), np.uint8(mask*255)
        # plot one sample image
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize = (16,3))
        palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
        
        mask_ = mask[:,:,0]
        contours, _ = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            cv2.polylines(image, contour, True, palet[self.spec_cat], 2)
        ax.imshow(image)
        return
    

    def stat_images(self, rows = float('inf')):
        'generate a file storing all inforamtion'
        # record the ratio
        dicStat = {'ImageId':[], 'mean':[], 'std':[], 'Class '+str(self.spec_cat+1):[]} # ratios for the images
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
            classid = self.spec_cat
            dicStat['Class '+str(classid+1)].append(self.stat_mask(mask[:,:,0]))
        
        self.stat_df = pd.DataFrame(dicStat)
        return self.stat_df



# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97456
class BalanceClassSamplerMultilabel(Sampler):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        if length is None:
            length = len(self.dataset)
        self.length = int(length)


    def __iter__(self):
        p1, p2, p3, p4, p5 = 0.123, 0.028, 0.89, 0.08, 0.80
        p = self.dataset.arrtags[:,0]/p1 + self.dataset.arrtags[:,1]/p2 +\
			self.dataset.arrtags[:,2]/p3 + self.dataset.arrtags[:,3]/p4 + self.dataset.arrtags[:,4]/p5
        p /= p.sum()
        l = np.random.choice(self.dataset.arrtags.shape[0], self.length, replace=True, p = p)
        l = l.reshape(-1)
        np.random.shuffle(l)
        l = l[:self.length]
        return iter(l)


    def __len__(self):
        return self.length



# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97456
class BalanceClassSampler(Sampler):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        if length is None:
            length = len(self.dataset)
        self.length = int(length)

        half = self.length // 2 + 1
        self.pos_length = half
        self.neg_length = half
        print('pos num: %s, neg num: %s' % (self.pos_length, self.neg_length))

    def __iter__(self):
        pos_index = np.where(self.dataset.pos_flag)[0]
        neg_index = np.where(~self.dataset.pos_flag)[0]

        pos = np.random.choice(pos_index, self.pos_length, replace=True)
        neg = np.random.choice(neg_index, self.neg_length, replace=True)

        l = np.hstack([pos, neg]).T
        l = l.reshape(-1)
        np.random.shuffle(l)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length
