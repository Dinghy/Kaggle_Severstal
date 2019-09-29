import numpy as np
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import matplotlib.pyplot as plt 

from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask

def post_process_single(pred, other, thres_seg = 0.5, size_seg = 100, thres_oth = -float('inf'), size_oth = 0):
    pred = (pred > thres_seg).astype(int)
    # TTA: combining classification and size thresholding
    nsize = pred.sum() 
    if nsize < size_seg or (nsize < size_oth+size_seg and other < thres_oth):
        pred *= 0
    return pred


def post_process(pred, other, dicPara):
    # TTA: thresholding
    assert(pred.shape[2] == 4)
    for category in range(4):
        if 'thres_oth{:d}'.format(category+1) in dicPara:
            paras = [dicPara[item] for item in ('thres_seg{:d}'.format(category+1), 'size_seg{:d}'.format(category+1), \
                        'thres_oth{:d}'.format(category+1), 'size_oth{:d}'.format(category+1))]
        else:
            paras = [dicPara[item] for item in ('thres_seg{:d}'.format(category+1), 'size_seg{:d}'.format(category+1))]
        pred[:,:,category] = post_process_single(pred[:,:,category], other[category], *paras)

    return pred


class Evaluate:

    def __init__(self, net, device, dataloader, args, dicPara = None, isTest = True):
        self.args = args
        self.net = net
        self.device = device
        self.dataloader = dataloader
        self.dicPara = dicPara
        self.isTest = isTest


    def search_parameter(self):
        'Bayes opt to determine the threshold for each label'
        self.eval_net()

        # store the parameters
        self.dicPara = {}

        def cal_dice(thres_seg, size_seg, thres_oth=-float('inf'), size_oth=0):
            ipos = 0
            dice = 0.0
            for pred, other, true_rle in zip(preds, others, trues):
                # post process
                true = rle2mask(true_rle, self.args.width, self.args.height)
                pred = post_process_single(pred, other, thres_seg, size_seg, thres_oth, size_oth)
                ipos += 1
                dice += dice_metric(true, pred)
            return dice/len(preds)

        preds, trues, others = [], [], []
        for category in range(4):
            ipos = 0
            # get the prediction and save it
            with torch.no_grad():
                for data in tqdm(self.dataloader):
                    images, labels = data[0], data[1]
                    for image_raw, label_raw in zip(images, labels):
                        # flip and predict
                        output_merge, output_other = self.predict_flip(image_raw)
                        true_mask = label_raw[:,:,category].detach().numpy().astype(int)
                        if category == 0:
                            trues.append(mask2rle(true_mask))
                            preds.append(output_merge[:,:,category])
                            others.append(output_other[category])
                        else:
                            trues[ipos] = mask2rle(true_mask)
                            preds[ipos] = output_merge[:,:,category]
                            others[ipos] = output_other[category]
                        ipos += 1
            
            # using bayes optimize to determine the threshold
            if self.args.output == 0:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000)}
            elif self.args.output == 1:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000), 'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            elif self.args.output == 2:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000), 'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            optimizer = BayesianOptimization(f = cal_dice, pbounds = pbounds, random_state = 1)   
            # adjust the bayes opt stage
            if self.args.test_run or self.args.epoch < 5:
                optimizer.maximize(init_points = 5, n_iter = 1)
            else:
                optimizer.maximize(init_points = 50, n_iter = 250)

            self.dicPara['thres_seg{:d}'.format(category+1)] = optimizer.max['params']['thres_seg']
            self.dicPara['size_seg{:d}'.format(category+1)]  = optimizer.max['params']['size_seg']
            if self.args.output > 0:
                self.dicPara['thres_oth{:d}'.format(category+1)] = optimizer.max['params']['thres_oth']
                self.dicPara['size_oth{:d}'.format(category+1)]  = optimizer.max['params']['size_oth']
        return


    def predict_flip(self, image_raw):
        'predict the mask for one simple image'
        output_merge = np.zeros((self.args.height, self.args.width, self.args.category))
        output_other = np.zeros((1, self.args.category))
        
        for i in range(4):
            lr, ud = divmod(i, 2)
            image = image_raw.detach().numpy()
            if lr == 1: # flip left to right
                image = np.fliplr(image)
            if ud == 1: # flip up to down
                image = np.flipud(image)

            image_flip = torch.from_numpy(image.copy()).unsqueeze(0).to(self.device)

            # obtain the prediction
            if isinstance(self.net, list):
                preds = None
                for net in self.net:
                    if preds is None:
                        preds = net(image_flip.permute(0, 3, 1, 2))
                    else:
                        preds += net(image_flip.permute(0, 3, 1, 2))
                preds /= len(self.net)
            else:
                preds = self.net(image_flip.permute(0, 3, 1, 2))
            
            if self.args.output == 0:   # vanilla
                outputs = torch.sigmoid(preds).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
            elif self.args.output == 1: # regression
                outputs = torch.sigmoid(preds[0]).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                output_other += preds[1].detach().cpu().numpy()/4
            elif self.args.output == 2: # classification
                outputs = torch.sigmoid(preds[0]).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                output_other += torch.sigmoid(preds[1]).detach().cpu().numpy()/4

            # flip the predicted results
            if lr == 1: # flip the prediction from right to left
                outputs = np.fliplr(outputs)
            if ud == 1:
                outputs = np.flipud(outputs)
            # merge the result
            output_merge += outputs/4
        return output_merge, output_other[0]

    
    def eval_net(self):
        if isinstance(self.net, list):
            for net in self.net:
                net.eval()
        else:
            self.net.eval()
        return


    def predict_dataloader(self, to_rle = False, fnames = None):
        if self.dicPara is None:
            self.search_parameter()

        if to_rle and fnames is None:
            raise ValueError('File names are not given.')
        # evaluate the net
        self.eval_net()
        
        dicPred = dict()
        for classid in range(self.args.category):
            dicPred['Class '+str(classid+1)] = []
            dicPred['Dice '+str(classid+1)] = []
            dicPred['True '+str(classid+1)] = []
        dicSubmit = {'ImageId_ClassId':[], 'EncodedPixels':[]}
        dice, preds = 0.0, []
        ipos = 0
        def area_ratio(mask):
            return mask.sum()/self.args.height/self.args.width
            
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                images, labels = data[0], data[1]
                for image_raw, label_raw in zip(images, labels):
                    # flip and predicted
                    output_merge, output_other = self.predict_flip(image_raw)
        
                    # using simple threshold and output the result
                    output_thres = post_process(output_merge, output_other, self.dicPara)
                    # transfer into the rles
                    # record the predicted labels
                    for category in range(self.args.category):
                        # to rle if required
                        if to_rle:
                            fname = fnames[ipos]
                            fname_short = fname.split('/')[-1]+'_{:d}'.format(category+1)
                            dicSubmit['ImageId_ClassId'].append(fname_short)
                            rle = mask2rle(output_thres[:,:,category])
                            dicSubmit['EncodedPixels'].append(rle)
                        dicPred['Class {:d}'.format(category+1)].append(area_ratio(output_thres[:,:,category]))
                        
                        if not self.isTest:
                            dice_cat = dice_metric(label_raw[:,:,category].detach().numpy(), output_thres[:,:,category])
                            dicPred['Dice {:d}'.format(category+1)].append(dice_cat)
                            dicPred['True {:d}'.format(category+1)].append(area_ratio(label_raw[:,:,category].detach().numpy()))
                            
                    ipos += 1
                    # calculate the dice if it is not a test dataloader
                    if not self.isTest:
                        # print(label_raw.shape, label_raw.min(), label_raw.max())
                        # print(output_thres.shape, output_thres.min(), output_thres.max())
                        dice += dice_metric(label_raw.detach().numpy(), output_thres)
                        # print(ipos, dice)
        
        keys = [key for key in dicPred.keys()]
        for key in keys:
            if len(dicPred[key]) == 0:
                dicPred.pop(key, None)
        return dice, dicPred, dicSubmit


    def plot_sampled_predict(self):
        'plot the sampled results'
        if self.dicPara is None:
            self.search_parameter()
        
        # evaluate the net
        self.eval_net()
        iplot = 0
        fig, axs = plt.subplots(self.args.batch, 2, figsize=(16,16))
        
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                images, labels = data[0], data[1]
                for image_raw, label_raw in zip(images, labels):
                    # flip and predict
                    output_merge, output_other = self.predict_flip(image_raw)
            
                    # using simple threshold and output the result
                    output_thres = post_process(output_merge, output_other, self.dicPara)
            
                    # plot
                    if iplot < self.args.batch:
                        ax = axs[iplot, 0]
                        plot_mask(image_raw.detach().numpy(), output_thres, ax)
                        ax.axis('off')

                        ax = axs[iplot, 1]
                        plot_mask(image_raw.detach().numpy(), label_raw.detach().numpy(), ax)
                        ax.axis('off')
                        iplot += 1

                if iplot == self.args.batch:
                    break
        
        if self.isTest:
            plt.savefig('evaluate_image_test.png')
        else:
            plt.savefig('../output/evaluate_image.png')
        return
