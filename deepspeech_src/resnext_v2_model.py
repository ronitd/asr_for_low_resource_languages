import math 
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import copy


def C_BN_ACT(c_in, c_out, activation, transpose=False, dropout=None, bn=True):
	layers = []
	if transpose:
		layers.append(nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))
	else:
		layers.append(nn.Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1))
	if dropout:
		layers.append(nn.Dropout1d(dropout))
	if bn:
		layers.append(nn.BatchNorm1d(c_out))
	layers.append(activation)
	return nn.Sequential(*layers)

class Encoder(nn.Module):
	'''
    Input: (batch_size, num_channels, H, W)
    Output: (batch_size, 512, H / 2**7, W / 2**7)
	'''
	def __init__(self):
		super(Encoder, self).__init__()
		k_list = [80, 16, 32, 64, 128, 256, 512]
		activation = nn.LeakyReLU(0.2)
		layers = []
		for i in range(1, len(k_list)):
			print(k_list[i - 1], k_list[i])
			c_in, c_out = k_list[i - 1], k_list[i]
			bn = False if i == len(k_list) - 1 else True
			layers.append(C_BN_ACT(c_in, c_out, activation, bn=bn))
		self.convs = nn.Sequential(*layers)
    
	def forward(self, x):
		#print("Shape before Encoder: ", x.shape)  
		Ec = self.convs(x)
		#print("Shape After Encoder: ", Ec.shape)		
		return Ec

	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package


class Decoder(nn.Module):
	'''
    Input: (batch_size, 512, H, W), (batch_size, attr_dim)
    Output: (batch_size, 3, H * 2**7, W * 2**7)
	'''
	def __init__(self):
		super(Decoder, self).__init__()
		k_list = [80, 16, 32, 64, 128, 256, 512]	
		activation = nn.ReLU()
		'''
        layers = []
        for i in range(len(k_list) - 1, 0, -1):
            print(k_list[i], k_list[i - 1])
            c_in, c_out = k_list[i], k_list[i - 1]
            layers.append(C_BN_ACT(c_in, c_out, activation, transpose=True))
        self.deconvs = nn.Sequential(*layers)
		'''
		#self.deconv1 = C_BN_ACT(k_list[7] + attr_dim, k_list[6], activation, transpose=True)
		self.deconv2 = C_BN_ACT(k_list[6], k_list[5], activation, transpose=True)
		self.deconv3 = C_BN_ACT(k_list[5], k_list[4], activation, transpose=True)		
		self.deconv4 = C_BN_ACT(k_list[4], k_list[3], activation, transpose=True)
		self.deconv5 = C_BN_ACT(k_list[3], k_list[2], activation, transpose=True)
		self.deconv6 = C_BN_ACT(k_list[2], k_list[1], activation, transpose=True)
		self.deconv7 = C_BN_ACT(k_list[1], k_list[0], nn.Tanh(), transpose=True, bn=False)	
                
	def forward(self, Ec):
		#print("Before Decoder: ", Ec.shape)
		Ec = self.deconv2(Ec)
		Ec = self.deconv3(Ec)
		Ec = self.deconv4(Ec)
		Ec = self.deconv5(Ec)
		Ec = self.deconv6(Ec)
		Ec = self.deconv7(Ec)
		#print("After Decoder: ", Ec.shape) 
		return Ec
	
	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package


# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return silu(input) # simply apply already implemented SiLU

def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_clones_blocks(in_c, start, bottleneck_depth, width, jump, group_norm = False, weight_standardization = False, activation = nn.ReLU):
	n = int(width)
	return nn.ModuleList([copy.deepcopy(BottleneckBlock(in_c, start + jump*i, bottleneck_depth, group_norm, weight_standardization, activation = activation, power = i)) for i in range(n)])

def get_clones_filter(bottleneck_depth, out_channel, kernel_size = 3, padding = 0, repeat = 1, conv = nn.Conv1d, norm = nn.BatchNorm1d, activation = nn.ReLU):		
	return nn.ModuleList([copy.deepcopy(ConvBN(kernel_size, bottleneck_depth, out_channel, padding = 0, conv = conv, norm = norm, activation = activation)) for i in range(int(repeat))])

class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=16, **kargs):
        super().__init__(num_groups, num_channels, **kargs)

class Conv1d(nn.Conv1d):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
		super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

	def forward(self, x):
		weight = self.weight
		#print("Weight shape: ", self.weight.shape)
		weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
		weight = weight - weight_mean
		std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
		weight = weight / std.expand_as(weight)		
		return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 0):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1)
		
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvBN(nn.Module):
	def __init__(self, kernel_size, in_channel, out_channel, padding = 0, conv = nn.Conv1d, norm = nn.BatchNorm1d, activation = nn.ReLU, groups=1):
		super().__init__()
		self.conv = conv(in_channel, out_channel, kernel_size=kernel_size, padding=((kernel_size-1)//2), groups=groups)
		self.bn = norm(out_channel)
		self.activation = activation

	def forward(self, x):
		
		x = self.bn(self.activation(self.conv(x)))
		return x 


class BottleneckBlock(nn.Module): 
	def __init__(self, in_c, kernel_size, bottleneck_depth, group_norm=False, weight_standardization=False, activation=nn.ReLU, repeat_filter = 1, power = 4):
		super().__init__()
		conv = Conv1d if weight_standardization else nn.Conv1d #DepthwiseSeparableConv 
		norm = GroupNorm32 if group_norm else nn.BatchNorm1d
		self.repeat = 1 #kernel_size//2
		out_channel = int(math.pow(2, 4))  #bottleneck_depth 
		self.conv1 = conv(in_c, bottleneck_depth, kernel_size=1)
		self.conv2 = get_clones_filter(bottleneck_depth, out_channel, kernel_size=kernel_size, padding=((kernel_size-1)//2), repeat = self.repeat, conv = conv, norm = norm, activation = activation)
		self.conv3 = conv(out_channel, in_c, kernel_size=1)

		self.bn1 = norm(bottleneck_depth)
		self.bn3 = norm(in_c)
		self.activation = activation

	def forward(self, x): 
		x = self.bn1(self.activation(self.conv1(x)))
		for i in range(self.repeat):
			x = self.conv2[i](x)
		x = self.bn3(self.activation(self.conv3(x)))
		return x 


class ResNeXTBlock(nn.Module):
	def __init__(self, in_c, bottleneck_depth, width, jump, group_norm = False, weight_standardization = False, activation = nn.ReLU, dropout = 0.5):
		super().__init__()
		self.width = width
		self.blocks = get_clones_blocks(in_c, 3, bottleneck_depth, width, jump, group_norm, weight_standardization, activation)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		combo_x = 0
		for i in range(self.width):
			#print(self.blocks[i](x).shape)
			combo_x += (self.blocks[i](x))

		#combo_x = torch.cat(combo_x, 1)			
		x = self.dropout(x + combo_x)
		return x 

class ResNextASR_v2(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=128, args = []):
		super(ResNextASR_v2, self).__init__() 
		print("Using ResNexT like architecture")
		self.depth = args.depth
		self.depth_residual = args.depth_residual
		conv = Conv1d if args.weight_standardization else nn.Conv1d #DepthwiseSeparableConv
		norm = GroupNorm32 if args.group_norm else nn.BatchNorm1d
		activation = nn.ReLU #SiLU
		'''
		self.gamma_spectogram = nn.Parameter(torch.ones(1))
		self.gamma_mfcc = nn.Parameter(torch.ones(1))
		self.gamma_fbank = nn.Parameter(torch.ones(1))
		'''
		self.start_layers = nn.Sequential(
			conv(num_features, dense_dim, kernel_size=13),
			norm(dense_dim), 
			activation(),
			nn.Dropout(0.25),

			conv(dense_dim, dense_dim, kernel_size=5), 
			norm(dense_dim), 
			activation(), 
			nn.Dropout(0.25),
		)

		'''
		self.start_layers = nn.Sequential(
			conv(num_features, 480, kernel_size=13, groups=16),
			norm(480), 
			nn.ReLU(),
			nn.Dropout(0.25),
			
			conv(480, 384, kernel_size=17, groups = 96), 
			norm(384), 
			nn.ReLU(), 
			nn.Dropout(0.25),

			conv(384, 480, kernel_size=21, groups = 96), 
			norm(480), 
			nn.ReLU(), 
			nn.Dropout(0.25),

			conv(480, dense_dim, kernel_size=25, groups = 32), 
			norm(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		'''
		

		self.resnext_blocks = get_clones(ResNeXTBlock(dense_dim, bottleneck_depth, args.width, args.width_jump, args.group_norm, args.weight_standardization, nn.ReLU()), args.depth)
		self.end_layers = nn.Sequential(
			conv(dense_dim, dense_dim, kernel_size=1),
			norm(dense_dim), 
			activation(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)

		self.start_layers.apply(self.init_weights)
		self.resnext_blocks.apply(self.init_weights)
		self.end_layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		'''
		batch[:,:128, :] = self.gamma_spectogram*batch[:,:128, :].clone()
		batch[:,129:167, :] = self.gamma_mfcc*batch[:,129:167, :].clone()
		batch[:,167:, :] = self.gamma_fbank*batch[:,167:, :].clone()
		'''
		#batch = spec_augment_pytorch.spec_augment(batch, time_warping_para=40, frequency_masking_para=27, time_masking_para=70, frequency_mask_num=2, time_mask_num=2)		
		batch = batch.transpose(1,2)
		batch = self.start_layers(batch)

		for i in range(self.depth):
			if self.depth_residual:
				residual = batch
			batch = self.resnext_blocks[i](batch)
			if self.depth_residual:
				batch = batch + residual

		batch = self.end_layers(batch)		
		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
		#log_probs = F.softmax(y_pred, dim=1)
		return log_probs

	@staticmethod 
	def get_param_size(model): 
		params = 0 
		for p in model.parameters(): 
			tmp = 1 
			for x in p.size(): 
				tmp *= x 
			params += tmp 
		return params

	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package

