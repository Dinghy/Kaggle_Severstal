import numpy as np
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import matplotlib.pyplot as plt 

from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask

def post_process_single(pred, other, thres_seg = 0.5, size_seg = 100, thres_after = 0.4, thres_oth = -float('inf'),  size_oth = 0):
    pred_greed = (pred > thres_seg).astype(int)
    # TTA: combining classification and size thresholding
    nsize = pred_greed.sum() 
    if nsize < size_seg or (nsize < size_oth+size_seg and other < thres_oth):
        return pred * 0
    return (pred > thres_after).astype(int)


def post_process(pred, other, dicPara):
    # TTA: thresholding
    assert(pred.shape[2] == 4)
    for category in range(4):
        if 'thres_oth{:d}'.format(category+1) in dicPara:
            paras = [dicPara[item] for item in ('thres_seg{:d}'.format(category+1), 'size_seg{:d}'.format(category+1), \
                                                'thres_after{:d}'.format(category+1), 'thres_oth{:d}'.format(category+1), \
                                                'size_oth{:d}'.format(category+1))]
        else:
            paras = [dicPara[item] for item in ('thres_seg{:d}'.format(category+1), 'size_seg{:d}'.format(category+1),\
                                                'thres_after{:d}'.format(category+1))]
        pred[:,:,category] = post_process_single(pred[:,:,category], other[category], *paras)

    return pred


def post_process_segment(pred, thres_seg = 0.5, size_seg = 600, thres_after = 0.3):
    'From the first position in the last competition'
    pred_greed = (pred > thres_seg).astype(int)
    # TTA: combining classification and size thresholding
    nsize = pred_greed.sum() 
    if nsize < size_seg:
        return pred_greed*0
    return (pred > thres_after).astype(int)


class Meter:
    def __init__(self):
        self.value = 0
        self.num = 0

    def add(self, val):
        self.value += val
        self.num += 1
    
    def avg(self):
        if self.num > 0:
            return self.value/self.num
        else:
            return 0


class EvaluateOneCategory:
    def __init__(self, net, device, dataloader, args, dicPara = None, isTest = True):
        self.args = args
        if not isinstance(net, list):
            self.net = [net]
        else:
            self.net = net

        self.device = device
        self.dataloader = dataloader
        self.dicPara = dicPara
        self.isTest = isTest
        

    def search_parameter(self):
        'Using Bayes optimization to determine the threshold for each label'
        # evaluate the network
        self.eval_net()

        # store the parameters
        self.dicPara = {}

        def cal_dice(thres_seg, size_seg, thres_after = 0.2):
            ipos = 0
            dice = 0.0
            for pred, true_rle in zip(preds, trues):
                # post process
                true = rle2mask(true_rle, self.args.width, self.args.height)
                pred = post_process_segment(pred, thres_seg, size_seg, thres_after)
                ipos += 1
                dice += dice_metric(true, pred)
            return dice/len(preds)

        preds, trues, others = [], [], []
        category = self.args.spec_cat

        # get the prediction and save it
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                images, labels = data[0], data[1]
                images = images.permute(0,3,1,2).to(self.device)

		# flip and predict
                output_masks, output_labels = self.predict_flip_batch(images)
                # store the prediction
                for output_mask, label_true in zip(output_masks, labels):

                    true_mask = label_true[:,:,0].detach().numpy().astype(int)
                    trues.append(mask2rle(true_mask))
                    preds.append(output_mask[:,:,0])

            
            # using bayes optimize to determine the threshold
            if self.args.output == 0:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg': (500, 6000), 'thres_after': (0.1, 0.7)}
            elif self.args.output == 1:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg': (500, 6000), 'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            elif self.args.output == 2:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg': (500, 6000), 'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            optimizer = BayesianOptimization(f = cal_dice, pbounds = pbounds, random_state = 1)   
            # adjust the bayes opt stage
            if self.args.test_run or self.args.epoch < 5:
                optimizer.maximize(init_points = 5, n_iter = 1)
            else:
                optimizer.maximize(init_points = 100, n_iter = 10)

            self.dicPara['thres_seg'] = optimizer.max['params']['thres_seg']
            self.dicPara['size_seg']  = optimizer.max['params']['size_seg']
            self.dicPara['thres_after'] = optimizer.max['params']['thres_after']
        return


    def predict_flip_batch(self, images):
        'Same as predict flip but deal with a batch of images'
        # clean the data
        self.output_mask = np.zeros((images.shape[0], self.args.height, self.args.width, 1))
        self.output_label =  np.zeros((images.shape[0], 1))

        # obtain the prediction
        for net in self.net:
            for i in range(4):
                lr, ud = divmod(i, 2)
                images_batch = images.clone()
                # torch.Size([8, 3, 256, 1600])
                if lr == 1: # flip left to right
                    images_batch = torch.flip(images_batch, [3])
                if ud == 1: # flip up to down
                    images_batch = torch.flip(images_batch, [2])
                
                # predict
                preds = net(images_batch)

                # rectify the output
                if self.args.output == 0:   # vanilla
                    masks = torch.sigmoid(preds)
                elif self.args.output == 1: # regression
                    masks = torch.sigmoid(preds[0])
                    labels = preds[1]
                elif self.args.output == 2: # classification
                    masks = torch.sigmoid(preds[0])
                    labels = torch.sigmoid(preds[1])
  
                # flip the predicted results
                if lr == 1: # flip the prediction from right to left
                    masks = torch.flip(masks, [3])  # (256, 1600, 4)
                if ud == 1:
                    masks = torch.flip(masks, [2])
                    
                # merge the result
                self.output_mask += masks.permute(0, 2, 3, 1).detach().cpu().numpy()
                if self.args.output > 0:
                    self.output_label += labels.detach().cpu().numpy()

        return self.output_mask/4/len(self.net), self.output_label/4/len(self.net)

    
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
        
        dice_total, dice_pos, dice_neg = Meter(), Meter(), Meter()
 
        dicSubmit = {'ImageId_ClassId':[], 'EncodedPixels':[]}
        preds = []
        ipos = 0
        def area_ratio(mask):
            return mask.sum()/self.args.height/self.args.width
            
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                # load the data
                images, labels = data[0].to(self.device), data[1].to(self.device)
                images = images.permute(0, 3, 1, 2)

                output_masks, output_labels = self.predict_flip_batch(images)
                
                for output_mask, output_label, label_raw in zip(output_masks, output_labels, labels):
                    # using simple threshold and output the result
                    output_thres = post_process_segment(output_mask, thres_seg = self.dicPara['thres_seg'], \
                                                             size_seg = self.dicPara['size_seg'], \
                                                             thres_after = self.dicPara['thres_after'])

                    # calculate the dice if it is not a test dataloader
                    if not self.isTest:
                        dice = dice_metric(label_raw.detach().cpu().numpy(), output_thres)
                        dice_total.add(dice)
                        if label_raw.sum() > 0:
                            dice_pos.add(dice)
                        else:
                            dice_neg.add(dice)
        
        # print information
        print('Parameters: ',','.join(['"{:s}":{:.4f}'.format(key, val) for key,val in self.dicPara.items()]))
        print('Dice total {:.3f}'.format(dice_total.avg()))
        print('Positive Data {:d}, {:.3f}'.format(dice_pos.num, dice_pos.avg()))
        print('Negative Data {:d}, {:.3f}'.format(dice_neg.num, dice_neg.avg())) 
        return dice_total.avg()


class Evaluate:

    def __init__(self, net, device, dataloader, args, dicPara = None, isTest = True):
        self.args = args
        if not isinstance(net, list):
            self.net = [net]
        else:
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

        def cal_dice(thres_seg, size_seg, thres_after, thres_oth=-float('inf'), size_oth=0):
            ipos = 0
            dice = 0.0
            for pred, other, true_rle in zip(preds, others, trues):
                # post process
                true = rle2mask(true_rle, self.args.width, self.args.height)
                pred = post_process_single(pred, other, thres_seg, size_seg, thres_after, thres_oth, size_oth)
                ipos += 1
                dice += dice_metric(true, pred)
            return dice/len(preds)

        preds, trues, others = [], [], []
        bfirst = True
        categories = [0,1,2,3]

        for category in categories:
            ipos = 0
            # get the prediction and save it
            with torch.no_grad():
                for data in tqdm(self.dataloader):
                    images, labels = data[0], data[1]
                    for image_raw, label_raw in zip(images, labels):
                        # flip and predict
                        output_merge, output_other = self.predict_flip(image_raw, category)
                        true_mask = label_raw[:,:,category].detach().numpy().astype(int)
                        if bfirst:
                            trues.append(mask2rle(true_mask))
                            preds.append(output_merge)
                            others.append(output_other)
                        else:
                            trues[ipos] = mask2rle(true_mask)
                            preds[ipos] = output_merge
                            others[ipos] = output_other
                        ipos += 1
            bfirst = False
            
            # using bayes optimize to determine the threshold
            if self.args.output == 0:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000)}
            elif self.args.output == 1:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000), 'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            elif self.args.output == 2:
                pbounds = {'thres_seg': (0.1, 0.7), 'size_seg' : (500, 6000), 'thres_after': (0.1, 0.5),\
                            'thres_oth':(0.1, 0.7), 'size_oth':(500, 6000)}
            optimizer = BayesianOptimization(f = cal_dice, pbounds = pbounds, random_state = 1)   
            # adjust the bayes opt stage
            if self.args.test_run or self.args.epoch < 5:
                optimizer.maximize(init_points = 5, n_iter = 1)
            else:
                optimizer.maximize(init_points = 300, n_iter = 50)

            self.dicPara['thres_seg{:d}'.format(category+1)] = optimizer.max['params']['thres_seg']
            self.dicPara['size_seg{:d}'.format(category+1)]  = optimizer.max['params']['size_seg']
            self.dicPara['thres_after{:d}'.format(category+1)] = optimizer.max['params']['thres_after']
            if self.args.output > 0:
                self.dicPara['thres_oth{:d}'.format(category+1)] = optimizer.max['params']['thres_oth']
                self.dicPara['size_oth{:d}'.format(category+1)]  = optimizer.max['params']['size_oth']
        return


    def predict_flip_batch(self, images):
        'Same as predict flip but deal with a batch of images'
        # clean the data
        self.output_mask = np.zeros((images.shape[0], self.args.height, self.args.width, self.args.category))
        self.output_label =  np.zeros((images.shape[0], self.args.category))

        # obtain the prediction
        for net in self.net:
            for i in range(4):
                lr, ud = divmod(i, 2)
                images_batch = images.clone()
                # torch.Size([8, 3, 256, 1600])
                if lr == 1: # flip left to right
                    images_batch = torch.flip(images_batch, [3])
                if ud == 1: # flip up to down
                    images_batch = torch.flip(images_batch, [2])
                
                # predict
                preds = net(images_batch)

                # rectify the output
                if self.args.output == 0:   # vanilla
                    masks = torch.sigmoid(preds)
                elif self.args.output == 1: # regression
                    masks = torch.sigmoid(preds[0])
                    labels = preds[1]
                elif self.args.output == 2: # classification
                    masks = torch.sigmoid(preds[0])
                    labels = torch.sigmoid(preds[1])
  
                # flip the predicted results
                if lr == 1: # flip the prediction from right to left
                    masks = torch.flip(masks, [3])  # (256, 1600, 4)
                if ud == 1:
                    masks = torch.flip(masks, [2])
                    
                # merge the result
                self.output_mask += masks.permute(0, 2, 3, 1).detach().cpu().numpy()
                self.output_label += labels.detach().cpu().numpy()

        return self.output_mask/4/len(self.net), self.output_label/4/len(self.net)


    def predict_flip(self, image_raw, category):
        'predict the mask for one simple image'
        
        # clean the data
        output_mask = np.zeros((self.args.height, self.args.width))
        output_label = np.zeros((1,))

        # obtain the prediction
        for net in self.net:
            for i in range(4):
                lr, ud = divmod(i, 2)
                image = image_raw.detach().numpy()   # torch.Size([256, 1600, 3])
                if lr == 1: # flip left to right
                    image = np.fliplr(image)
                if ud == 1: # flip up to down
                    image = np.flipud(image)
                # flip the image
                image_flip = torch.from_numpy(image.copy()).unsqueeze(0).to(self.device)  # torch.Size([1, 256, 1600, 3])
                # predict
                preds = net(image_flip.permute(0, 3, 1, 2))

                if self.args.output == 0:   # vanilla
                    outputs = torch.sigmoid(preds).permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,category]
                elif self.args.output == 1: # regression
                    outputs = torch.sigmoid(preds[0]).permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,category]
                    output_label += preds[1].detach().cpu().numpy()[0,category]
                elif self.args.output == 2: # classification
                    outputs = torch.sigmoid(preds[0]).permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,category]
                    output_label += torch.sigmoid(preds[1]).detach().cpu().numpy()[0,category]

                # flip the predicted results
                if lr == 1: # flip the prediction from right to left
                    outputs = np.fliplr(outputs)  # (256, 1600, 4)
                if ud == 1:
                    outputs = np.flipud(outputs)

                # merge the result
                output_mask += outputs

        return output_mask/4/len(self.net), output_label/4/len(self.net)

    
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
                # load the data
                images, labels = data[0].to(self.device), data[1].to(self.device)
                images = images.permute(0, 3, 1, 2)

                output_masks, output_labels = self.predict_flip_batch(images)
                
                for output_mask, output_label, label_raw in zip(output_masks, output_labels, labels):
                    # using simple threshold and output the result
                    output_thres = post_process(output_mask, output_label, self.dicPara)
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
                            dice_cat = dice_metric(label_raw[:,:,category].detach().cpu().numpy(), output_thres[:,:,category])
                            dicPred['Dice {:d}'.format(category+1)].append(dice_cat)
                            dicPred['True {:d}'.format(category+1)].append(area_ratio(label_raw[:,:,category].detach().cpu().numpy()))
                            
                    ipos += 1
                    # calculate the dice if it is not a test dataloader
                    if not self.isTest:
                        dice += dice_metric(label_raw.detach().cpu().numpy(), output_thres)
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
        
        try:
            plt.savefig('../output/evaluate_image.png')
        except:
            plt.savefig('evaluate_image.png')
