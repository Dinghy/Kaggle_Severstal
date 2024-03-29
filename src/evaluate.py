import numpy as np
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import matplotlib.pyplot as plt 

from metric import dice_metric
from utils import mask2rle, rle2mask, plot_mask


def evaluate_batch(data, outputs, args, threshold = 0.5):
    # evaluate a mini-batch without too complicated thresholding
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
    # evaluate the dataloader without too complicated thresholding
    loss, dice, other = 0.0, 0.0, 0.0
    with torch.no_grad():
        for data in dataloader:
            images, masks = data[0].to(device), data[1].to(device)
            images = images.permute(0, 3, 1, 2)
            masks = masks.permute(0, 3, 1, 2)
            outputs = net(images)

            if criterion[1] is not None:
                loss += 0.2*criterion[1](outputs[1], data[2].to(device)).item()
                loss += criterion[0](outputs[0], masks, weight = args.wlovasz).item()
            else:
                loss += criterion[0](outputs, masks, weight = args.wlovasz).item()
                           
            res = evaluate_batch(data, outputs, args)
            dice, other = dice+res[0], other + res[1]
    return loss, dice, other


def post_process_single(pred, other, thres_seg = 0.5, size_seg = 100, thres_oth = -float('inf'), size_oth = 0, thres_after = -float('inf')):
    pred = (pred > thres_seg).astype(int)
    nsize = pred.sum()
    # TTA: combining classification and size thresholding 
    if nsize < size_seg or (nsize < size_oth + size_seg and other < thres_oth):
        return pred * 0
    # additional postprocessing
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

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Evaluate:
    # evaluate the models or a list of models in a systematical way
    def __init__(self, net, device, dataloader, args, dicPara = None, isTest = True):
        self.args = args
        if not isinstance(net, list):
            self.net = [net]
        else:
            self.net    = net
        self.device     = device
        self.dataloader = dataloader
        self.dicPara    = dicPara
        self.isTest     = isTest
        self.flip_num   = 4

    def eval_net(self):
        if isinstance(self.net, list):
            for net in self.net:
                net.eval()
        else:
            self.net.eval()
        return


    def search_parameter_fine(self, dicPara = None):
        'Use the Bayesian Optimization again to fine tune the parameters'
        self.eval_net()
        if dicPara is not None:
            self.dicPara = dicPara

        if self.dicPara is None:
            self.search_parameter()

        def cal_dice(thres_seg, size_seg, thres_oth=-float('inf'), size_oth=0, thres_after = -float('inf')):
            ipos = 0
            dice = 0.0
            for pred, other, true_rle in zip(preds, others, trues):
                # post process
                true = rle2mask(true_rle, self.args.width, self.args.height)
                pred = post_process_single(pred, other, thres_seg, size_seg, thres_oth, size_oth, thres_after)
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
                        output_merge, output_other = self.predict_flip(image_raw, category)
                        true_mask = label_raw[:,:,category].detach().numpy().astype(int)
                        if category == 0:
                            trues.append(mask2rle(true_mask))
                            preds.append(output_merge)
                            others.append(output_other)
                        else:
                            trues[ipos] = mask2rle(true_mask)
                            preds[ipos] = output_merge
                            others[ipos] = output_other
                        ipos += 1
            
            # using bayes optimize to determine the threshold
            thres_seg, size_seg, thres_oth, size_oth = self.dicPara['thres_seg{:d}'.format(category+1)],\
                                                       self.dicPara['size_seg{:d}'.format(category+1)],\
                                                       self.dicPara['thres_oth{:d}'.format(category+1)],\
                                                       self.dicPara['size_oth{:d}'.format(category+1)]

            pbounds = {'thres_seg'  : (thres_seg*0.95 , thres_seg*1.05), \
                       'size_seg'   : (size_seg *0.95  , size_seg*1.05), \
                       'thres_oth'  : (thres_oth*0.95 , thres_oth*1.05), \
                       'size_oth'   : (size_oth *0.95  , size_oth*1.05)}
                       
            optimizer = BayesianOptimization(f = cal_dice, pbounds = pbounds, random_state = 1)   
            
            # adjust the bayes opt stage
            # test
            if self.args.test_run or self.args.epoch < 5:
                optimizer.maximize(init_points = 10, n_iter = 1)
            else:
                optimizer.maximize(init_points = 200, n_iter = 150)
            
            # store the parameters
            for spara in ('thres_seg', 'size_seg', 'thres_oth', 'size_oth'):
                self.dicPara['{:s}{:d}'.format(spara, category+1)] = optimizer.max['params'][spara]

        
        print(self.dicPara)
        return


    def search_parameter(self):
        'Use the Bayesian optimization to determine the threshold for each label'
        self.eval_net()
        # store the parameters
        self.dicPara = {}

        def cal_dice(thres_seg, size_seg, thres_oth=-float('inf'), size_oth=0, thres_after = -float('inf')):
            ipos = 0
            dice = 0.0
            for pred, other, true_rle in zip(preds, others, trues):
                # post process
                true = rle2mask(true_rle, self.args.width, self.args.height)
                pred = post_process_single(pred, other, thres_seg, size_seg, thres_oth, size_oth, thres_after)
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
                        output_merge, output_other = self.predict_flip(image_raw, category)
                        true_mask = label_raw[:,:,category].detach().numpy().astype(int)
                        if category == 0:
                            trues.append(mask2rle(true_mask))
                            preds.append(output_merge)
                            others.append(output_other)
                        else:
                            trues[ipos] = mask2rle(true_mask)
                            preds[ipos] = output_merge
                            others[ipos] = output_other
                        ipos += 1
            
            # using bayes optimize to determine the threshold
            if self.args.eva_method == 0:     # basic operation
                pbounds = {'thres_seg': (0.4, 0.6), 'size_seg' : (500, 2000)}
            elif self.args.eva_method == 1 and self.args.output == 2:   # currently used version with classification
                if category >= 2:
                    pbounds = {'thres_seg': (0.4, 0.6), 'size_seg' : (1000, 2000), 'thres_oth':(0.4, 0.7), 'size_oth':(1000, 4000)}
                else:
                    pbounds = {'thres_seg': (0.4, 0.6), 'size_seg' : (500, 1500), 'thres_oth':(0.4, 0.7), 'size_oth':(1000, 4000)}
            elif self.args.eva_method == 2 and self.args.output == 2:   # adding another thresholding after the conditions with zeros
                pbounds = {'thres_seg': (0.4, 0.6), 'size_seg' : (500, 2000), 'thres_oth':(0.25, 0.7), 'size_oth':(1000, 4000), 'thres_after':(0.3, 0.5)}
            optimizer = BayesianOptimization(f = cal_dice, pbounds = pbounds, random_state = 1)   
            
            # adjust the bayes opt stage
            # test
            if self.args.test_run or self.args.epoch < 5:
                optimizer.maximize(init_points = 10, n_iter = 1)
            else:
                optimizer.maximize(init_points = 200, n_iter = 150)
            
            # store the parameters
            self.dicPara['thres_seg{:d}'.format(category+1)] = optimizer.max['params']['thres_seg']
            self.dicPara['size_seg{:d}'.format(category+1)]  = optimizer.max['params']['size_seg']
            if self.args.eva_method == 1 and self.args.output == 2:
                self.dicPara['thres_oth{:d}'.format(category+1)] = optimizer.max['params']['thres_oth']
                self.dicPara['size_oth{:d}'.format(category+1)]  = optimizer.max['params']['size_oth']
            elif self.args.eva_method == 2 and self.args.output == 2:
                self.dicPara['thres_oth{:d}'.format(category+1)] = optimizer.max['params']['thres_oth']
                self.dicPara['size_oth{:d}'.format(category+1)]  = optimizer.max['params']['size_oth']
                self.dicPara['thres_after{:d}'.format(category+1)] = optimizer.max['params']['thres_after']
        
        print(self.dicPara)
        return


    def predict_flip(self, image_raw, category):
        'predict the mask for one simple image'
        
        # clean the data
        output_mask = np.zeros((self.args.height, self.args.width))
        output_label = np.zeros((1,))

        # obtain the prediction
        for net in self.net:
            for i in range(self.flip_num):
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
                    if self.args.avg_test:
                        outputs = preds[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,category]
                        output_label += preds[1].detach().cpu().numpy()[0,category]
                    else:
                        outputs = torch.sigmoid(preds[0]).permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,category]
                        output_label += torch.sigmoid(preds[1]).detach().cpu().numpy()[0,category]

                # flip the predicted results
                if lr == 1: # flip the prediction from right to left
                    outputs = np.fliplr(outputs)  # (256, 1600, 4)
                if ud == 1:
                    outputs = np.flipud(outputs)

                # merge the result
                output_mask += outputs
        if self.args.avg_test:
            return sigmoid(output_mask/self.flip_num/len(self.net)), sigmoid(output_label/self.flip_num/len(self.net))
        else:
            return output_mask/self.flip_num/len(self.net), output_label/self.flip_num/len(self.net)

    def predict_flip_batch(self, images):
        'Same as predict flip but deal with a batch of images'
        # clean the data
        self.output_mask = np.zeros((images.shape[0], self.args.height, self.args.width, self.args.category))
        self.output_label =  np.zeros((images.shape[0], self.args.category))

        # obtain the prediction
        for net in self.net:
            for i in range(self.flip_num):
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
                    if self.args.avg_test:
                        masks = preds[0]
                        labels = preds[1]
                    else:
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
        if self.args.avg_test:
            return sigmoid(self.output_mask/self.flip_num/len(self.net)), sigmoid(self.output_label/self.flip_num/len(self.net))
        else:
            return self.output_mask/self.flip_num/len(self.net), self.output_label/self.flip_num/len(self.net)

    
    def predict_dataloader(self, to_rle = False, fnames = None, gen_pseudo = True):
        if self.dicPara is None:
            self.search_parameter()

        if to_rle and fnames is None:
            raise ValueError('File names are not given.')

        self.eval_net()
        dicPred = dict()
        for classid in range(self.args.category):
            dicPred['Class '+str(classid+1)] = []
            dicPred['Dice '+str(classid+1)] = []
            dicPred['True '+str(classid+1)] = []
        
        # initialize the dict for submission
        dicSubmit = {'ImageId_ClassId':[], 'EncodedPixels':[]}
        # initialize the dict for pseudo labeling
        dicPseudo = {'ImageId_ClassId':[], 'EncodedPixels':[], 'Trust':[]}

        # store the dice overall
        dice, preds = 0.0, []
        ipos = 0
        def area_ratio(mask):
            return mask.sum()/self.args.height/self.args.width
            
        with torch.no_grad():
            for data in tqdm(self.dataloader):
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
                            rle = mask2rle(output_thres[:,:,category])
                            # submission
                            dicSubmit['ImageId_ClassId'].append(fname_short)
                            dicSubmit['EncodedPixels'].append(rle)

                            # Pseudo labeling
                            if gen_pseudo:
                                dicPseudo['ImageId_ClassId'].append(fname_short)
                                dicPseudo['EncodedPixels'].append(rle)
                                dicPseudo['Trust'].append(output_label[category])

                        # record the statistics
                        dicPred['Class {:d}'.format(category+1)].append(area_ratio(output_thres[:,:,category]))
                        # add to the dice 
                        if not self.isTest:
                            dice_cat = dice_metric(label_raw[:,:,category].detach().cpu().numpy(), output_thres[:,:,category])
                            dicPred['Dice {:d}'.format(category+1)].append(dice_cat)
                            dicPred['True {:d}'.format(category+1)].append(area_ratio(label_raw[:,:,category].detach().cpu().numpy()))
                            dice  += dice_cat
                    ipos += 1
        
        keys = [key for key in dicPred.keys()]

        for key in keys:
            if len(dicPred[key]) == 0:
                dicPred.pop(key, None)
        # print the dictionary of parameters
        # print(self.dicPara)
        return dice/len(self.dataloader.dataset)/self.args.category, dicPred, dicSubmit, dicPseudo


    def plot_sampled_predict(self):
        'plot the sampled results'
        if self.dicPara is None:
            self.search_parameter()

        self.eval_net()
        iplot = 0
        fig, axs = plt.subplots(self.args.batch, 2, figsize=(16,16))
        
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                images, labels = data[0].to(self.device).permute(0,3,1,2), data[1].to(self.device)
                
                output_masks, output_labels = self.predict_flip_batch(images)

                for output_mask, output_label, image_raw, label_raw in zip(output_masks, output_labels, images, labels):
                    # using simple threshold and output the result
                    output_thres = post_process(output_mask, output_label, self.dicPara)
                    image_raw = image_raw.permute(1,2,0)
                    # plot
                    if iplot < self.args.batch:
                        ax = axs[iplot, 0]
                        plot_mask(image_raw.detach().cpu().numpy(), output_thres, ax)
                        ax.axis('off')

                        ax = axs[iplot, 1]
                        plot_mask(image_raw.detach().cpu().numpy(), label_raw.detach().cpu().numpy(), ax)
                        ax.axis('off')
                        iplot += 1

                if iplot == self.args.batch:
                    break
        
        if self.isTest:
            plt.savefig('evaluate_image_test.png')
        else:
            plt.savefig('../output/evaluate_image.png')
        return
