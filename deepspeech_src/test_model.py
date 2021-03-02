import math 
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import copy

def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_clones_blocks(in_c, start, bottleneck_depth, width, jump, group_norm = False, weight_standardization = False):
	return nn.ModuleList([copy.deepcopy(BottleneckBlock(in_c, start + jump*i, bottleneck_depth, group_norm, weight_standardization)) for i in range(int(width))])

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

class BottleneckBlock(nn.Module): 
	def __init__(self, in_c, kernel_size, bottleneck_depth, group_norm=False, weight_standardization=False):
		super().__init__()
		conv = Conv2d if weight_standardization else nn.Conv2d
		norm = GroupNorm32 if group_norm else nn.BatchNorm2d

		self.conv1 = conv(in_c, bottleneck_depth, kernel_size=1)
		self.conv2 = conv(bottleneck_depth, bottleneck_depth, kernel_size=kernel_size, padding=((kernel_size-1)//2))
		self.conv3 = conv(bottleneck_depth, in_c, kernel_size=1)

		self.bn1 = norm(bottleneck_depth)
		self.bn2 = norm(bottleneck_depth)
		self.bn3 = norm(in_c)

	def forward(self, x): 
		x = self.bn1(F.relu(self.conv1(x)))
		x = self.bn2(F.relu(self.conv2(x)))
		x = self.bn3(F.relu(self.conv3(x)))
		return x

class ResNeXTBlock(nn.Module):
	def __init__(self, in_c, bottleneck_depth, width, jump, group_norm = False, weight_standardization = False, dropout = 0.5):
		super().__init__()
		self.width = width
		self.blocks = get_clones_blocks(in_c, 3, bottleneck_depth, width, jump, group_norm, weight_standardization)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		combo_x = 0
		for i in range(self.width):
			combo_x += self.blocks[i](x)
		x = self.dropout(x + combo_x)
		return x 

class ResNextASR_v2(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16, args = []):
		super(ResNextASR_v2, self).__init__() 
		print("Using ResNexT like architecture")
		self.depth = args.depth
		conv = Conv1d if args.weight_standardization else nn.Conv2d
		norm = GroupNorm32 if args.group_norm else nn.BatchNorm2d
		self.conv2 = nn.Conv2d(1, dense_dim, kernel_size=13)
		self.start_layers = nn.Sequential(
			conv(1, dense_dim, kernel_size=13),
			norm(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			conv(dense_dim, dense_dim, kernel_size=5), 
			norm(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.resnext_blocks = get_clones(ResNeXTBlock(dense_dim, bottleneck_depth, args.width, args.width_jump, args.group_norm, args.weight_standardization), args.depth)
		self.end_layers = nn.Sequential(
			conv(dense_dim, dense_dim, kernel_size=1),
			norm(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv2d(dense_dim, num_classes, kernel_size=3)

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
		batch = torch.unsqueeze(batch, dim=1)
		#print("Batch Shape: ", batch.shape)
		#batch = self.start_layers(batch)
		#print("After Convlution: ", batch.shape)
		
		batch = self.start_layers(batch)
		for i in range(self.depth):
			batch = self.resnext_blocks[i](batch)
			
		batch = self.end_layers(batch)
					
		y_pred = self.classifier(batch)
		#print("After End Batch: ", y_pred.shape)
		y_pred = torch.sum(y_pred, dim=2)
		#print("After Sum: ", y_pred.shape)
		#exit() 	
		log_probs = F.log_softmax(y_pred, dim=1)
		#print("Output shape: ", log_probs.shape)
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

