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

from utils import seed_everything

############################################
# Encoder

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


pretrained_settings = {}
pretrained_settings['resnet34'] = {
		'imagenet': {
			'url': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
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
	}
	Encoder = encoders[name]['encoder']
	encoder = Encoder(**encoders[name]['params'])
	encoder.out_shapes = encoders[name]['out_shapes']

	if encoder_weights is not None:
		settings = encoders[name]['pretrained_settings'][encoder_weights]
		encoder.load_state_dict(torch.load('weights/resnet34-333f7ec4.pth'))
	return encoder

###########################################################
# decoder

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
	def __init__(self, in_channels, out_channels, use_batchnorm=True):
		super().__init__()
		self.block = nn.Sequential(
			Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
			Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
		)

	def forward(self, x):
		x, skip = x
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		if skip is not None:
			x = torch.cat([x, skip], dim=1)
		x = self.block(x)
		return x


class CenterBlock(DecoderBlock):
	def forward(self, x):
		return self.block(x)


class UnetDecoder(nn.Module):
	def __init__(
			self,
			encoder_channels,
			decoder_channels = (256, 128, 64, 32, 16),
			final_channels=1,
			use_batchnorm=True,
			center=False,
	):
		super().__init__()

		if center:
			channels = encoder_channels[0]
			self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
		else:
			self.center = None

		in_channels = self.compute_channels(encoder_channels, decoder_channels)
		out_channels = decoder_channels

		self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
		self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
		self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
		self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
		self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
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

		x = self.layer1([encoder_head, skips[0]])
		x = self.layer2([x, skips[1]])
		x = self.layer3([x, skips[2]])
		x = self.layer4([x, skips[3]])
		x = self.layer5([x, None])
		x = self.final_conv(x)

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

		self.decoder = UnetDecoder(
			encoder_channels = self.encoder.out_shapes,
			decoder_channels = decoder_channels,
			final_channels = classes,
			use_batchnorm = decoder_use_batchnorm,
			center = center,
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
