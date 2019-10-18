"""
UNet with Resnet34 as encoder
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
"""
import torch
from torch import nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from utils import seed_everything

import argparse
############################################
# SENet Encoder
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class SENetEncoder(SENet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        
        del self.last_linear
        del self.avg_pool

    def forward(self, x):
        for module in self.layer0[:-1]:
            x = module(x)

        x0 = x
        x = self.layer0[-1](x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


############################################
# Resnet Encoder

class ResNetEncoder(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)

#############################################
# Settings of the pretrained models

pretrained_settings = {}
pretrained_settings['resnet34'] = {
        'imagenet': {
            'path': '../input/weights/resnet34-333f7ec4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
}

pretrained_settings['se_resnet50'] = {
        'imagenet': {
            'path': '../input/weights/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
}


def get_encoder(name, encoder_weights = None):
    encoders = {
        'resnet34': {
            'encoder': ResNetEncoder,
            'pretrained_settings': pretrained_settings['resnet34'],
            'out_shapes': (512, 256, 128, 64, 64),
            'params': {
                'block': BasicBlock,
                'layers': [3, 4, 6, 3],
            },
        },
        'se_resnet50': {
            'encoder': SENetEncoder,
            'pretrained_settings': pretrained_settings['se_resnet50'],
            'out_shapes': (2048, 1024, 512, 256, 64),
            'params': {
                'block': SEResNetBottleneck,
                'layers': [3, 4, 6, 3],
                'downsample_kernel_size': 1,
                'downsample_padding': 0,
                'dropout_p': None,
                'groups': 1,
                'inplanes': 64,
                'input_3x3': False,
                'num_classes': 1000,
                'reduction': 16
        },
    },
    }
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(torch.load(settings['path']))
    return encoder

###########################################################
# Unet decoder
class CBAM_Module(nn.Module):
    # https://github.com/bestfitting/kaggle/blob/master/siim_acr/src/layers/layer_util.py
    def __init__(self, channels, reduction, attention_kernel_size=3):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = True, use_attention = True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )
        self.use_attention = use_attention
        if self.use_attention:
            # attention
            self.channel_gate = CBAM_Module(out_channels, reduction = 16, attention_kernel_size = 3)
            # resnet link
            # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0))
            # self.bn   = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #if self.use_attention:
        #    shortcut = self.bn(self.conv(x))

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        if self.use_attention:
            x = self.channel_gate(x)
            # x = F.relu(x+shortcut)            
        return x


class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels = (256, 128, 64, 32, 16),
            final_channels = 1,
            use_batchnorm = True,
            center = False,
            concat_output = True,
            use_attention = True,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm = use_batchnorm)
        else:
            self.center = None

        # concatenate the outputs in the decoder 
        self.concat_output = concat_output

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm = use_batchnorm, use_attention = use_attention)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm = use_batchnorm, use_attention = use_attention)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm = use_batchnorm, use_attention = use_attention)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm = use_batchnorm, use_attention = use_attention)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm = use_batchnorm, use_attention = use_attention)
        
        if self.concat_output:
            self.cbr2 = Conv2dReLU(out_channels[3], 16, kernel_size=1, use_batchnorm=use_batchnorm)
            self.cbr3 = Conv2dReLU(out_channels[2], 16, kernel_size=1, use_batchnorm=use_batchnorm)
            self.cbr4 = Conv2dReLU(out_channels[1], 16, kernel_size=1, use_batchnorm=use_batchnorm)
            self.cbr5 = Conv2dReLU(out_channels[0], 16, kernel_size=1, use_batchnorm=use_batchnorm)
            self.final_conv = nn.Conv2d(16*5, final_channels, kernel_size=(1, 1))
        else:
            self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        
        
        d5 = self.layer1([encoder_head, skips[0]]) # torch.Size([1, 256, 16, 100])
        d4 = self.layer2([d5, skips[1]])           # torch.Size([1, 128, 32, 200]) 
        d3 = self.layer3([d4, skips[2]])           # torch.Size([1, 64, 64, 400]) 
        d2 = self.layer4([d3, skips[3]])           # torch.Size([1, 32, 128, 800]) 
        d1 = self.layer5([d2, None])               # torch.Size([1, 16, 256, 1600]) 

        if self.concat_output:
            dconcat = torch.cat((d1,
                                 F.interpolate(self.cbr2(d2), scale_factor=2, mode='bilinear',align_corners=False),
                                 F.interpolate(self.cbr3(d3), scale_factor=4, mode='bilinear', align_corners=False),
                                 F.interpolate(self.cbr4(d4), scale_factor=8, mode='bilinear', align_corners=False),
                                 F.interpolate(self.cbr5(d5), scale_factor=16, mode='bilinear', align_corners=False),
                        ), 1)
            x = self.final_conv(dconcat)
        else:
            x = self.final_conv(d1)

        return x
    
    def initialize(self):
        seed_everything()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

###########################################################
# Unet structure

class Unet(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
    Returns:
        ``torch.nn.Module``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    def __init__(
            self,
            encoder_name = 'resnet34',
            encoder_weights = 'imagenet',
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32, 16),
            classes = 1,
            activation = 'sigmoid',
            center = False, args = None):  # usefull for VGG models

        super().__init__()
    
        self.args = args

        self.encoder = get_encoder(
            encoder_name,
            encoder_weights = encoder_weights
        )

        # extract information from the result
        use_attention = self.args.decoder.find('cbam') != -1
        concat_output = self.args.decoder.find('con') != -1

        self.decoder = UnetDecoder(
            encoder_channels = self.encoder.out_shapes,
            decoder_channels = decoder_channels,
            final_channels = classes,
            use_batchnorm = decoder_use_batchnorm,
            center = center,
            use_attention = use_attention,
            concat_output = concat_output,
        )

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')
        self.name = 'u-{}'.format(encoder_name)
        
        # add a regression/regression part
        if self.args.output == 1 or self.args.output == 2: 
            self.ap0 = nn.AvgPool2d((3,3))
            self.nsize = self.get_flat_shape()
            self.lin0 = nn.Linear(self.nsize, self.args.category)
            # self.lin1 = nn.Linear(100, self.args.category)
    

    def get_flat_shape(self, input_shape = None):
        '''get the shape of the bottom by a forward passing'''
        if input_shape is None:
            input_shape = (3, self.args.height, self.args.width)
        x = self.encoder(torch.randn(1, *input_shape))
        x = self.ap0(x[0])
        return int(np.prod(x.size()[1:]))

    
    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        # vanilla version
        if self.args.output == 0:
            x = self.decoder(x)
            return x
        # regression
        elif self.args.output == 1:
            xreg = self.ap0(x[0])
            xreg = self.lin0(xreg.view(-1, self.nsize))
            # xreg = F.relu(self.lin0(xreg.view(-1, self.nsize)))
            # xreg = self.lin1(xreg)
            x = self.decoder(x)
            return x, xreg        
        # classification
        elif self.args.output == 2:
            xcla = self.ap0(x[0])
            xcla = self.lin0(xcla.view(-1, self.nsize))
            # xcla = F.relu(self.lin0(xcla.view(-1, self.nsize)))
            # xcla = self.lin1(xcla)
            x = self.decoder(x)
            return x, xcla
    

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
        return x



if __name__ == '__main__':
    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_split',   action = 'store_false', default = True,    help = 'Rerun train/test split')
    parser.add_argument('--accumulate',   action = 'store_false', default = True,    help = 'Not doing gradient accumulation or not')
    parser.add_argument('--bayes_opt',    action = 'store_true',  default = False,   help = 'Do Bayesian optimization in finding hyper-parameters')
    parser.add_argument('-l','--load_mod',action = 'store_true',  default = False,   help = 'Load a pre-trained model')
    parser.add_argument('-t','--test_run',action = 'store_true',  default = False,   help = 'Run the script quickly to check all functions')
    
    parser.add_argument('--decoder',     type = str,  default = 'cbam_con', help = 'Structure in the Unet decoder')
    parser.add_argument('--normalize',   type = int,  default = 0,          help = 'Normalize the images or not')
    parser.add_argument('--wlovasz',     type = float,default = 0.2,        help = 'The weight used in Lovasz loss')
    parser.add_argument('--augment',     type = int,  default = 0,          help = 'The type of train augmentations: 0 vanilla, 1 add contrast, 2 add  ')
    parser.add_argument('--loss',        type = int,  default = 0,          help = 'The loss: 0 BCE vanilla; 1 wbce+dice; 2 wbce+lovasz.')
    parser.add_argument('--sch',         type = int,  default = 0,          help = 'The schedule of the learning rate: 0 step; 1 cosine annealing; 2 cosine annealing with warmup.')    
    parser.add_argument('-m', '--model', type = str,  default = 'resnet34', help = 'The backbone network of the neural network.')
    parser.add_argument('-e', '--epoch', type = int,  default = 5,          help = 'The number of epochs in the training')
    parser.add_argument('--height',      type = int,  default = 256,        help = 'The height of the image')
    parser.add_argument('--width',       type = int,  default = 1600,       help = 'The width of the image')
    parser.add_argument('--category',    type = int,  default = 4,          help = 'The category of the problem')
    parser.add_argument('-b', '--batch', type = int,  default = 8,          help = 'The batch size of the training')
    parser.add_argument('-s','--swa',    type = int,  default = 4,          help = 'The number of epochs for stochastic weight averaging')
    parser.add_argument('-o','--output', type = int,  default = 0,          help = 'The type of the network, 0 vanilla, 1 add regression, 2 add classification.')
    parser.add_argument('--seed',        type = int,  default = 1234,       help = 'The random seed of the algorithm.')
    parser.add_argument('--eva_method',  type = int,  default = 1,          help = 'The evaluation method in postprocessing: 0 thres/size; 1 thres/size/classify; 2 thres/size/classify/after')
    args = parser.parse_args()

    # test
    net = Unet('resnet34', encoder_weights = None, classes = 4, activation = None, args = args)
    image = torch.zeros(1, 3, 256, 1600)
    #test = F.interpolate(image, scale_factor=2, mode='nearest')
    #print(test.shape)
    output = net(image)
    print(output.shape)
