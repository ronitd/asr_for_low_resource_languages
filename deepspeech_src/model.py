import math 
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter

supported_rnns = {
	'lstm': nn.LSTM, 
	'rnn': nn.RNN, 
	'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

class FaderNetEncoder(nn.Module):
	def __ini__(self):
		super(FaderNetEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			)
	
	def forward(self, encoding):
		return self.encoder(encoding)

class FaderNetDecoder(nn.Module):
	def __ini__(self):
		super(FaderNetDecoder, self).__init__()
		self.decoder = nn.Sequential(
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 
			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)
	
	def forward(self, decoding):
		decoder = self.decoder(decoding)
		y_pred = self.classifier(decoder)
		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs 
				

class FaderNetDiscriminater(nn.Module):
	def __init__(self, no_of_languages, features=512):
		super(FaderNetDiscriminater, self).__init__()
		self.no_of_languages = no_of_languages
		self.features = features
		self.conv1 = nn.Conv1d(self.features, self.features, kernel_size = 4, stride = 2, padding = 1)
 
		self.seq = nn.Sequential(
			nn.Conv1d(self.features, self.features, kernel_size = 4, stride = 2, padding = 1),
			)	
		self.discriminater = nn.Sequential(
			nn.Linear(self.features, self.features),
			nn.BatchNorm1d(self.features),
			nn.ReLU(), 
			nn.Dropout(0.25),
			nn.Linear(self.features, self.no_of_languages),
			nn.BatchNorm1d(self.no_of_languages),
		 	nn.Softmax(),
			)
		
		
	def forward(self, x):
		x = self.conv1(x)
		batch_size = x.shape[0]
		x = x.view(x.shape[0]*x.shape[2], x.shape[1])
		#print(x.shape)
		x = self.discriminater(x)
		#x = x.view(batch_size, self.features, -1)
		#print("Discriminater: ", x.shape)
		#x = torch.argmax(x, dim=0)
		#print("Argmax x: ", x.shape)
		x = x.view(batch_size, -1, self.no_of_languages)
			
		#print("Batch Wise: ", x.shape)
		x = torch.sum(x, dim=1)
		#print("Sum: ", x.shape)	
		#exit()
		#max_voting = []
		#for i in range(batch_size):
			#unique, count = torch.unique(x[i][:], return_counts=True, dim=0)
			#print(unique)
			#print(count)
			#print(count.double() / torch.sum(count).double())
			#print(unique[torch.argmax(count)])
			#max_voting.append(count.double() / torch.sum(count).double())
			#torch.cat(max_voting, unique[torch.argmax(count)])
		#print(max_voting)
		#print(torch.stack(max_voting))
		#max_voting = torch.stack(max_voting)
		#max_voting.requires_grad = True
		#print("Max Voting: ", max_voting)
		#print("Unique: ", unique.shape)	
		#x = x.view(batch_size, -1)
		#print("X :", x.shape)
		return x

class FaderWireNetworkASR(nn.Module):
	def __init__(self, num_features, num_classes, num_languages, dense_dim=256, bottleneck_depth=16):
		super(FaderWireNetworkASR, self).__init__()
		self.num_features = num_features
		self.num_classes = num_classes
		self.num_languages = num_languages
		self.dense_dim = dense_dim
		self.bottleneck_depth = bottleneck_depth
		self.conv_ver_1 = nn.Conv2d(self.num_features, self.dense_dim, kernel_size=(80, 1))
		self.conv_ver_3 = nn.Conv2d(self.num_features, self.dense_dim, kernel_size=(80, 3))
		self.conv_ver_5 = nn.Conv2d(self.num_features, self.dense_dim, kernel_size=(80, 5))
		self.batchnorm_ver = nn.BatchNorm1d(self.dense_dim)
		self.conv2d = nn.Conv2d(self.num_features, 1, kernel_size=3)	
		self.encode = nn.Sequential(
			nn.Conv1d(self.dense_dim, self.dense_dim, kernel_size=3),
			nn.BatchNorm1d(self.dense_dim),
			nn.ReLU(), 
			nn.Dropout(0.25),
			FaderResNeXTBlock(self.dense_dim, self.bottleneck_depth),
			nn.Dropout(0.25),
			FaderResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25),
			FaderResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25),
			)
		self.resNeXTBlock = FaderResNeXTBlock(self.dense_dim, self.bottleneck_depth, self.num_languages)
		self.dropout = nn.Dropout(0.25)
		self.relu_activation = nn.ReLU()
		self.conv1d = nn.Conv1d(self.dense_dim + self.num_languages, self.dense_dim + self.num_languages, kernel_size=1)
		self.batchnorm = nn.BatchNorm1d(self.dense_dim + self.num_languages)
		self.classifier = nn.Conv1d(self.dense_dim + self.num_languages, self.num_classes, kernel_size=1)

	def forward(self, x, y):
		#print("Input x in forward", x.shape)
		x = x.transpose(2,3)
		x1 = self.conv_ver_1(x)
		x1 = torch.squeeze(x1, 2)
		#print(x1.shape)
		x3 = self.conv_ver_3(x)
		x3 = torch.squeeze(x3, 2)
		#print(x2.shape)
		x5 = self.conv_ver_5(x)
		x5 = torch.squeeze(x5, 2)

		x = torch.cat((x1, x3, x5), 2)
		x = self.relu_activation(self.batchnorm_ver(x))
		#print(x3.shape)
		#exit()
		#x = self.conv2d(x)
		#print(x.shape)
		#print(torch.squeeze(x, 1).shape)
		#exit()
		#x = torch.squeeze(x, 1).transpose(1,2)
		
		enc_outputs = self.encode(x)
		#print("Encoder Shape: ", enc_outputs.shape)
		decoder_input = self.concatenate_attribute(enc_outputs, y)
		decoder_input = self.dropout(self.resNeXTBlock(decoder_input, y))
		
		#print("REsNext Block1 Output: ", decoder_input.shape)	
		decoder_input = self.dropout(self.resNeXTBlock(decoder_input, y))
		#print("REsNext Block2 Output: ", decoder_input.shape)
		decoder_input = self.dropout(self.batchnorm(self.relu_activation(self.conv1d(decoder_input))))
		#print("REsNext Block3 Output: ", decoder_input.shape) 
		#decoder_output = self.classifier(decoder_input)
		y_pred = self.classifier(decoder_input)
		log_probs = F.log_softmax(y_pred, dim=1)
		return enc_outputs, log_probs
	
	def concatenate_attribute(self, output, y):	
		#print("Y shape: ", y.shape)
		y = y.view(y.shape[0], y.shape[1], -1).repeat(1, 1, output.shape[2])
		#print("Y after Reshape: ", y.shape)
		return torch.cat((output, y), 1)

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

class FaderBottleneckBlock(nn.Module): 
	def __init__(self, in_c, kernel_size, bottleneck_depth, num_languages=0):
		super().__init__()
		self.num_languages = num_languages
		self.conv1 = nn.Conv1d(in_c + self.num_languages, bottleneck_depth, kernel_size=1)
		self.conv2 = nn.Conv1d(bottleneck_depth + self.num_languages, bottleneck_depth, kernel_size=kernel_size, padding=((kernel_size-1)//2))
		self.conv3 = nn.Conv1d(bottleneck_depth + self.num_languages, in_c, kernel_size=1)

		self.bn1 = nn.BatchNorm1d(bottleneck_depth)
		self.bn2 = nn.BatchNorm1d(bottleneck_depth)
		self.bn3 = nn.BatchNorm1d(in_c)

	def forward(self, x, y): 
		x = self.bn1(F.relu(self.conv1(x)))
		if self.num_languages != 0:
			x = self.concatenate_attribute(x, y)
		x = self.bn2(F.relu(self.conv2(x)))
		if self.num_languages != 0:
			x = self.concatenate_attribute(x, y)
		x = self.bn3(F.relu(self.conv3(x)))
		return x


	def concatenate_attribute(self, output, y):	
		#print("Y shape: ", y.shape)
		y = y.view(y.shape[0], y.shape[1], -1).repeat(1, 1, output.shape[2])
		#print("Y after Reshape: ", y.shape)
		return torch.cat((output, y), 1)


class FaderResNeXTBlock(nn.Module):
	def __init__(self, in_c, bottleneck_depth, num_languages=0):
		super().__init__()
		self.num_languages = num_languages
		self.b_block_1 = FaderBottleneckBlock(in_c, 3, bottleneck_depth, num_languages) 
		self.b_block_1b = FaderBottleneckBlock(in_c, 5, bottleneck_depth, num_languages) #Not in original
		self.b_block_2 = FaderBottleneckBlock(in_c, 7, bottleneck_depth, num_languages)
		self.b_block_2b = FaderBottleneckBlock(in_c, 9, bottleneck_depth, num_languages) #Not in original
		self.b_block_3 = FaderBottleneckBlock(in_c, 11, bottleneck_depth, num_languages)
		self.b_block_3b = FaderBottleneckBlock(in_c, 13, bottleneck_depth, num_languages) #Not in original
		self.b_block_4 = FaderBottleneckBlock(in_c, 15, bottleneck_depth, num_languages)
		self.b_block_4b = FaderBottleneckBlock(in_c, 17, bottleneck_depth, num_languages) #Not in original
		self.b_block_5 = FaderBottleneckBlock(in_c, 19, bottleneck_depth, num_languages)

	def forward(self, x, y = []): 
		x1 = self.b_block_1(x, y)
		x2 = self.b_block_2(x, y)
		x3 = self.b_block_3(x, y)
		x4 = self.b_block_4(x, y)
		x5 = self.b_block_5(x, y)
		x1b = self.b_block_1b(x, y)
		x2b = self.b_block_2b(x, y)
		x3b = self.b_block_3b(x, y)
		x4b = self.b_block_4b(x, y)
		combo_x = x1 + x2 + x3 + x4 + x5 + x1b + x2b + x3b + x4b
		#print("Input x", x.shape)
		#print("Combo", (x + combo_x).shape)
		if self.num_languages != 0:
			combo_x = self.concatenate_attribute(combo_x, y)
		return x + combo_x

	def concatenate_attribute(self, output, y):	
		#print("Y shape: ", y.shape)
		y = y.view(y.shape[0], y.shape[1], -1).repeat(1, 1, output.shape[2])
		#print("Y after Reshape: ", y.shape)
		return torch.cat((output, y), 1)
 	
	'''
	def encode(self, x):
		return nn.Sequential(
			nn.Conv1d(self.num_features, self.dense_dim, kernel_size=13),
			nn.BatchNorm1d(self.dense_dim),
			nn.ReLU(), 
			nn.Dropout(0.25),
			ResNeXTBlock(self.dense_dim, self.bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25),
			)
		
	def decode(self, enc_outputs, y):
		
		
		return nn.Sequential(
			ResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(self.dense_dim, self.bottleneck_depth), 
			nn.Dropout(0.25), 
			nn.Conv1d(self.dense_dim, self.dense_dim, kernel_size=1),
			nn.BatchNorm1d(self.dense_dim),
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
	'''	
	

		
'''		
class FaderNetworkASR(nn.Module):
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(FaderNetworkASR, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv1d(num_features, 16, kernel_size=15),
			nn.BatchNorm1d(16), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(16, 32, kernel_size=3),
			nn.BatchNorm1d(32), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(32, 64, kernel_size=3),
			nn.BatchNorm1d(64), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(64, 128, kernel_size=3),
			nn.BatchNorm1d(128), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(128, 256, kernel_size=3),
			nn.BatchNorm1d(256), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(256, 256, kernel_size=3),
			nn.BatchNorm1d(256), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
		)
		self.decoder = nn.Sequential(
			nn.Conv1d(256, 256, kernel_size=3),
			nn.BatchNorm1d(256), 
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Conv1d(256, 128, kernel_size=3),
			nn.BatchNorm1d(128), 
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Conv1d(128, 64, kernel_size=3),
			nn.BatchNorm1d(64), 
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Conv1d(64, 32, kernel_size=3),
			nn.BatchNorm1d(32), 
			nn.LeakyReLU(0.2),
			nn.Dropout(0.25),
			nn.Conv1d(32, 16, kernel_size=3),
			nn.BatchNorm1d(16), 
			nn.ReLU(0.2),
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(16, num_classes, kernel_size=1)
		

            

	def forward(self, batch):
		#print(batch.shape) 
		batch = self.encoder(batch)
		#print("Encoder: ", batch.shape)
		batch = self.decoder(batch)
		#print("Decoder: ", batch.shape)
		y_pred = self.classifier(batch)
		#print("Pred: ", y_pred.shape)
		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs
		batch = batch.transpose(1,2)
		print("After Transpose: ", batch.shape)
		batch = batch.contiguous().view(batch.shape[0]*batch.shape[1], -1)
		print(batch.shape)
		exit()
'''	
	
class InferenceBatchSoftmax(nn.Module): 
	def forward(self, input_): 
		"""Expect last dimension to contain features"""
		if not self.training:
			return F.softmax(input_, dim=-1) 
		else: 
			#return input_
			return F.log_softmax(input_, dim=-1)

class SequenceWise(nn.Module): 
	def __init__(self, module):
		"""
		Collapses input of dim T*N*H to (T*N) * H, and applies to a module 
		Allows handling of variable sequence lengths and minibatch sizes
		So: turn a <seq_len x batch x height/depth> into <seq_len * batch x height/depth>
		"""
		super(SequenceWise, self).__init__() 
		self.module = module
	
	def forward(self, x): 
		t, n = x.size(0), x.size(1)
		x = x.view(t * n, -1) 
		x = self.module(x) 
		x = x.view(t, n, -1) 
		return x 

class MaskConv(nn.Module): 
	def __init__(self, seq_module): 
		"""
		Adds padding to the output of the module based on the given lengths. 
		This is to ensure the results of the model do not change when 
		batch sizes change during inference
		Input needs to be in the shape: <batch x C x D x seq> 
		"""
		super(MaskConv, self).__init__() 
		self.seq_module = seq_module 

	def forward(self, x, lengths): 
		""" 
		x: input of size <batch x C x D x seq>
		lengths: supposed length of the output from each layer of seq_module
		return: masked output from the module
		""" 
		for module in self.seq_module:
			x = module(x) 
			#Create a mask, initially filled with 0
			mask = torch.ByteTensor(x.size()).fill_(0)
			if x.is_cuda:
				#Put the thing in the CUDA 
				mask = mask.cuda() 
			for i, length in enumerate(lengths): 
				#Grab the true sequence length
				length = length.item() 
				if (mask[i].size(-1) - length) > 0: 
					#If the mask seq_len is longer than true length, 
					#fill the seq_length of the mask from length to the end with 1
					mask[i].narrow(-1, length, mask[i].size(-1) - length).fill_(1)
			#Any where the mask has a 1, fill x with 0 at that spot
			x = x.masked_fill(mask, 0)
		return x, lengths 

class BatchRNN(nn.Module): 
	def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True): 
		super(BatchRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional 

		self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None

		self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, 
			bidirectional = bidirectional, bias=True)
		self.num_directions = 2 if bidirectional else 1 

	def flatten_parameters(self): 
		self.rnn.flatten_parameters() 

	def forward(self, x, output_lengths): 
		if self.batch_norm is not None: 
			x = self.batch_norm(x)
		# print("output lengths: {}".format(output_lengths))
		# print("x shape pre pack: {}".format(x.shape))
		x = nn.utils.rnn.pack_padded_sequence(x, output_lengths) 
		# print("x shape post pack: {}".format(x.data.shape))
		x, h = self.rnn(x) 
		# print("x shape post rnn: {}".format(x.data.shape))
		x, _ = nn.utils.rnn.pad_packed_sequence(x) 
		if self.bidirectional:
			x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
		return x 

class DeepSpeech1D(nn.Module): 
	def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, 
		bidirectional=True, context=20):
		super(DeepSpeech1D, self).__init__() 

		self.version = '0.1.1'
		self.hidden_size = rnn_hidden_size 
		self.hidden_layers = nb_layers
		self.rnn_type = rnn_type 
		self.labels = labels 
		self.bidirectional = bidirectional 

		num_classes = len(self.labels) 

		#Input should have: <batch x channel x seq>
		#3 layer, 512, 512, 512 -- filter size: 5, 5, 5 -- stride: 2, 2, 1
		self.conv = MaskConv(nn.Sequential(
			nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True), 
			nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True), 
			nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True),
			nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1), 
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True),
			nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1), 
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True)
		))

		rnn_input_size = 512

		rnns = [] 
		rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
			bidirectional=bidirectional, batch_norm=False)
		rnns.append(('0', rnn)) 
		for x in range(nb_layers - 1):
			rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
				bidirectional=bidirectional, batch_norm=False)
			rnns.append(('{}'.format(x+1), rnn))
		self.rnns = nn.Sequential(OrderedDict(rnns)) 

		fully_connected = nn.Sequential(
			nn.BatchNorm1d(rnn_hidden_size), 
			nn.Linear(rnn_hidden_size, num_classes, bias=False)
		)
		self.fc = nn.Sequential(
			SequenceWise(fully_connected)
		)
		self.inference_softmax = InferenceBatchSoftmax()

	def forward(self, x, lengths):
		"""
		x shape: <batch x channel x seq>
		"""
		lengths = lengths.cpu().int() 
		#print("Input shape pre-conv {}".format(x.shape))
		output_lengths = self.get_seq_lens(lengths)
		#print("Output lenghts: {}".format(output_lengths))
		x, _ = self.conv(x, output_lengths)

		#right here, it would be: <batch x channel x seq> 
		sizes = x.size() 
		#so here, it would be <batch x seq x channel>, then <seq x batch x channel> 
		x = x.transpose(1, 2)
		x = x.transpose(0, 1).contiguous()

		#For RNN, we need: <seq x batch x feat>
		for rnn in self.rnns: 
			x = rnn(x, output_lengths) 

		x = self.fc(x) 

		x = self.inference_softmax(x) 
		return x, output_lengths

	def get_seq_lens(self, input_length): 
		seq_len = input_length
		seq_len = ((seq_len + 2 * 3 - 9 - 1) / 5) + 1 #First layer of conv 
		seq_len = ((seq_len + 2 * 2 - 7 - 1) / 4) + 1 #Second layer of conv 
		seq_len = ((seq_len + 2 * 1 - 3 - 1) / 2) + 1
		seq_len = ((seq_len + 2 * 1 - 3 - 1) / 2) + 1
		seq_len = ((seq_len + 2 * 1 - 3 - 1) / 2) + 1
		#seq_len = int((seq_len + 1) / 1) #Last layer of conv
		return seq_len.int() 

		# for m in self.conv.modules(): 
		# 	if type(m) == nn.modules.conv.Conv1d:
		# 		seq_len = ((seq_len + 2 * m.padding - m.dilation * (m.kernel_size - 1) - 1) /m.stride + 1)
		# 	return seq_len.int()

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
			'version': model.version, 
			'hidden_size': model.hidden_size, 
			'hidden_layers': model.hidden_layers, 
			'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()), 
			'labels': model.labels, 
			'state_dict': model.state_dict(), 
			'bidirectional': model.bidirectional, 
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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

	@classmethod
	def load_model(cls, path):
		package = torch.load(path, map_location=lambda storage, loc: storage)
		model = cls(rnn_hidden_size=package['hidden_size'],
					nb_layers=package['hidden_layers'],
					labels=package['labels'],
					rnn_type=supported_rnns[package['rnn_type']],
					bidirectional=package.get('bidirectional', True),
		)
		model.load_state_dict(package['state_dict'])
		for x in model.rnns:
			x.flatten_parameters()
		return model

	@classmethod
	def load_model_package(cls, package):
		model = cls(rnn_hidden_size=package['hidden_size'],
					nb_layers=package['hidden_layers'],
					labels=package['labels'],
					rnn_type=supported_rnns[package['rnn_type']],
					bidirectional=package.get('bidirectional', True),
				)
		model.load_state_dict(package['state_dict'])
		return model


class DeepSpeechCPC(nn.Module): 
	def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, 
		bidirectional=True, context=20):
		super(DeepSpeechCPC, self).__init__() 

		self.version = '0.1.1'
		self.hidden_size = rnn_hidden_size 
		self.hidden_layers = nb_layers
		self.rnn_type = rnn_type 
		self.labels = labels 
		self.bidirectional = bidirectional 

		num_classes = len(self.labels) 

		#Input should have: <batch x channel x seq>
		#3 layer, 512, 512, 512 -- filter size: 5, 5, 5 -- stride: 2, 2, 1
		self.conv = MaskConv(nn.Sequential(
			nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True), 
			nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True), 
			nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm1d(512), 
			nn.Hardtanh(0, 20, inplace=True)
		))

		rnn_input_size = 512

		rnns = [] 
		rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
			bidirectional=bidirectional, batch_norm=False)
		rnns.append(('0', rnn)) 
		for x in range(nb_layers - 1):
			rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
				bidirectional=bidirectional, batch_norm=False)
			rnns.append(('{}'.format(x+1), rnn))
		self.rnns = nn.Sequential(OrderedDict(rnns)) 

		fully_connected = nn.Sequential(
			nn.BatchNorm1d(rnn_hidden_size), 
			nn.Linear(rnn_hidden_size, num_classes, bias=False)
		)
		self.fc = nn.Sequential(
			SequenceWise(fully_connected)
		)
		self.inference_softmax = InferenceBatchSoftmax()

	def forward(self, x, lengths):
		"""
		x shape: <batch x channel x seq>
		"""
		lengths = lengths.cpu().int() 
		#print("Input shape pre-conv {}".format(x.shape))
		output_lengths = self.get_seq_lens(lengths)
		#print("Output lenghts: {}".format(output_lengths))
		x, _ = self.conv(x, output_lengths)

		#right here, it would be: <batch x channel x seq> 
		sizes = x.size() 
		#so here, it would be <batch x seq x channel>, then <seq x batch x channel> 
		x = x.transpose(1, 2)
		x = x.transpose(0, 1).contiguous()

		#For RNN, we need: <seq x batch x feat>
		for rnn in self.rnns: 
			x = rnn(x, output_lengths) 

		x = self.fc(x) 

		x = self.inference_softmax(x) 
		return x, output_lengths

	def get_seq_lens(self, input_length): 
		seq_len = input_length
		for m in self.conv.modules(): 
			if type(m) == nn.modules.conv.Conv1d:
				seq_len = ((seq_len + 2 * m.padding - m.dilation * (m.kernel_size - 1) - 1) /m.stride + 1)
			return seq_len.int()

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
			'version': model.version, 
			'hidden_size': model.hidden_size, 
			'hidden_layers': model.hidden_layers, 
			'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()), 
			'labels': model.labels, 
			'state_dict': model.state_dict(), 
			'bidirectional': model.bidirectional, 
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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

	@classmethod
	def load_model(cls, path):
		package = torch.load(path, map_location=lambda storage, loc: storage)
		model = cls(rnn_hidden_size=package['hidden_size'],
					nb_layers=package['hidden_layers'],
					labels=package['labels'],
					rnn_type=supported_rnns[package['rnn_type']],
					bidirectional=package.get('bidirectional', True),
		)
		model.load_state_dict(package['state_dict'])
		for x in model.rnns:
			x.flatten_parameters()
		return model

	@classmethod
	def load_model_package(cls, package):
		model = cls(rnn_hidden_size=package['hidden_size'],
					nb_layers=package['hidden_layers'],
					labels=package['labels'],
					rnn_type=supported_rnns[package['rnn_type']],
					bidirectional=package.get('bidirectional', True),
				)
		model.load_state_dict(package['state_dict'])
		return model


class Wav2Letter(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		if not (self.raw_audio):
			print("Use MFCC network")
			#Something else already take care of reducing temporal dimension
			#Could be MFCC, could be AudioCPC
			self.layers = nn.Sequential(
				# nn.Conv1d(num_features, 250, 36, 2),
				# nn.ReLU(), 
				# nn.BatchNorm1d(250), 
				nn.Conv1d(num_features, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 2000, 32), 
				nn.ReLU(), 
				nn.BatchNorm1d(2000), 
				nn.Conv1d(2000, 2000, 1), 
				nn.ReLU(),
				nn.BatchNorm1d(2000),  
				nn.Conv1d(2000, num_classes, 1), 
			)
		else: 
			print("Use raw audio network")
			self.layers = nn.Sequential(
				nn.Conv1d(num_features, 250, 250, 40), 
				nn.ReLU(),
				nn.BatchNorm1d(250),  
				nn.Conv1d(250, 250, 48, 2),
				nn.ReLU(), 
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 2000, 32), 
				nn.ReLU(), 
				nn.BatchNorm1d(2000), 
				nn.Conv1d(2000, 2000, 1), 
				nn.ReLU(), 
				nn.BatchNorm1d(2000), 
				nn.Conv1d(2000, num_classes, 1)
			)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		y_pred = self.layers(batch) 
		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	def predict(self, sample): 
		"""
		Sample shape: num features x frame_len
		"""
		_input = sample.reshape(1, sample.shape[0], sample.shape[1])
		log_prob = self.forward(_input) 
		return log_prob

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package


class Wav2Letter_Inception(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_Inception, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		if not (self.raw_audio):
			print("Use MFCC network")
			#Something else already take care of reducing temporal dimension
			#Could be MFCC, could be AudioCPC
			self.layers = nn.Sequential(
				# nn.Conv1d(num_features, 250, 36, 2),
				# nn.ReLU(), 
				# nn.BatchNorm1d(250), 
				nn.Conv1d(num_features, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 250, 7), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				nn.Conv1d(250, 2000, 32), 
				nn.ReLU(), 
				nn.BatchNorm1d(2000), 
				nn.Conv1d(2000, 2000, 1), 
				nn.ReLU(),
				nn.BatchNorm1d(2000),  
				nn.Conv1d(2000, num_classes, 1), 
			)
		else: 
			print("Use raw audio network")
			self.layers = nn.Sequential(
				nn.Conv1d(num_features, 250, 250, 40), 
				nn.ReLU(),
				nn.BatchNorm1d(250), 
				#Bottleneck
				nn.Conv1d(250, 64, 1),
				nn.ReLU(),
				#There. That's your bottleneck 
				nn.Conv1d(64, 250, 32, 2),	nn.ReLU(), nn.BatchNorm1d(250), 
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 250, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(250),
				#Bottleneck
				nn.Conv1d(250, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 2000, 32), nn.ReLU(), nn.BatchNorm1d(2000),

				nn.Conv1d(2000, 64, 1), nn.ReLU(),

				nn.Conv1d(64, 2000, 1), nn.ReLU(), nn.BatchNorm1d(2000), 
				nn.Conv1d(2000, num_classes, 1)
			)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		y_pred = self.layers(batch) 
		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	def predict(self, sample): 
		"""
		Sample shape: num features x frame_len
		"""
		_input = sample.reshape(1, sample.shape[0], sample.shape[1])
		log_prob = self.forward(_input) 
		return log_prob

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class Wav2Letter_ResNet(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_ResNet, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use raw audio network")
		self.conv1 = nn.Conv1d(num_features, 250, 250, 40)
		self.relu1 = nn.ReLU() 
		self.bn1 = nn.BatchNorm1d(250)

		self.conv2 = nn.Conv1d(250, 64, 1) 
		self.relu2 = nn.ReLU() 
		self.bn2 = nn.BatchNorm1d(64)

		self.conv3 = nn.Conv1d(64, 250, 32, 2) 
		self.relu3 = nn.ReLU() 
		self.bn3 = nn.BatchNorm1d(250)
			#ResnetBlock
		self.resnet_1_conv1 = nn.Conv1d(250, 64, 1) 
		self.resnet_1_relu1 = nn.ReLU()
		self.resnet_1_bn1 = nn.BatchNorm1d(64)
		self.resnet_1_conv2 = nn.Conv1d(64, 250, 7, padding=3) 
		self.resnet_1_relu2 = nn.ReLU() 
		self.resnet_1_bn2 = nn.BatchNorm1d(250)
		self.resnet_1_conv3 = nn.Conv1d(250, 250, 7, padding=3) 

		self.relu4 = nn.ReLU() 
		self.bn4 = nn.BatchNorm1d(250)

		self.resnet_2_conv1 = nn.Conv1d(250, 64, 1) 
		self.resnet_2_relu1 = nn.ReLU()
		self.resnet_2_bn1 = nn.BatchNorm1d(64)
		self.resnet_2_conv2 = nn.Conv1d(64, 250, 7, padding=3) 
		self.resnet_2_relu2 = nn.ReLU()
		self.resnet_2_bn2 = nn.BatchNorm1d(250) 
		self.resnet_2_conv3 = nn.Conv1d(250, 250, 7, padding=3) 

		self.relu5 = nn.ReLU() 
		self.bn5 = nn.BatchNorm1d(250)

		self.resnet_3_conv1 = nn.Conv1d(250, 64, 1) 
		self.resnet_3_relu1 = nn.ReLU()
		self.resnet_3_bn1 = nn.BatchNorm1d(64)
		self.resnet_3_conv2 = nn.Conv1d(64, 250, 7, padding=3) 
		self.resnet_3_relu2 = nn.ReLU()
		self.resnet_3_bn2 = nn.BatchNorm1d(250) 
		self.resnet_3_conv3 = nn.Conv1d(250, 250, 7, padding=3) 

		self.relu6 = nn.ReLU() 
		self.bn6 = nn.BatchNorm1d(250)

		self.resnet_4_conv1 = nn.Conv1d(250, 64, 1) 
		self.resnet_4_relu1 = nn.ReLU()
		self.resnet_4_bn1 = nn.BatchNorm1d(64)
		self.resnet_4_conv2 = nn.Conv1d(64, 250, 7, padding=3) 
		self.resnet_4_relu2 = nn.ReLU()
		self.resnet_4_bn2 = nn.BatchNorm1d(250) 
		self.resnet_4_conv3 = nn.Conv1d(250, 250, 7, padding=3) 

		self.relu7 = nn.ReLU() 
		self.bn7 = nn.BatchNorm1d(250)

		self.conv8 = nn.Conv1d(250, 64, 1) 
		self.relu8 = nn.ReLU() 
		self.bn8 = nn.BatchNorm1d(64)
		self.conv9 = nn.Conv1d(64, 250, 32) 
		self.relu9 = nn.ReLU() 
		self.bn9 = nn.BatchNorm1d(250)

		self.conv10 = nn.Conv1d(250, 64, 1) 
		self.relu10 = nn.ReLU() 
		self.bn10 = nn.BatchNorm1d(64)
		self.conv11 = nn.Conv1d(64, 1000, 1) 
		self.relu11 = nn.ReLU() 
		self.bn11 = nn.BatchNorm1d(1000)

		self.conv12 = nn.Conv1d(1000, 1000, 1) 
		self.relu12 = nn.ReLU() 
		self.bn12 = nn.BatchNorm1d(1000) 

		self.classifier = nn.Conv1d(1000, num_classes, 1)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.bn1(self.relu1(self.conv1(batch)))
		batch = self.bn2(self.relu2(self.conv2(batch)))

		batch = self.bn3(self.relu3(self.conv3(batch)))

		resid = batch
		batch = self.resnet_1_bn1(self.resnet_1_relu1(self.resnet_1_conv1(batch)))
		batch = self.resnet_1_bn2(self.resnet_1_relu2(self.resnet_1_conv2(batch)))
		batch = self.resnet_1_conv3(batch)

		batch += resid 
		batch = self.bn4(self.relu4(batch)) 

		resid = batch
		batch = self.resnet_2_bn1(self.resnet_2_relu1(self.resnet_2_conv1(batch)))
		batch = self.resnet_2_bn2(self.resnet_2_relu2(self.resnet_2_conv2(batch)))
		batch = self.resnet_2_conv3(batch)

		batch += resid 
		batch = self.bn5(self.relu5(batch)) 

		resid = batch
		batch = self.resnet_3_bn1(self.resnet_3_relu1(self.resnet_3_conv1(batch)))
		batch = self.resnet_3_bn2(self.resnet_3_relu2(self.resnet_3_conv2(batch)))
		batch = self.resnet_3_conv3(batch)

		batch += resid 
		batch = self.bn6(self.relu6(batch)) 

		resid = batch
		batch = self.resnet_4_bn1(self.resnet_4_relu1(self.resnet_4_conv1(batch)))
		batch = self.resnet_4_bn2(self.resnet_4_relu2(self.resnet_4_conv2(batch)))
		batch = self.resnet_4_conv3(batch)

		batch += resid 
		batch = self.bn7(self.relu7(batch)) 

		batch = self.bn8(self.relu8(self.conv8(batch)))
		batch = self.bn9(self.relu9(self.conv9(batch)))
		batch = self.bn10(self.relu10(self.conv10(batch)))
		batch = self.bn11(self.relu11(self.conv11(batch)))
		batch = self.bn12(self.relu12(self.conv12(batch)))

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package


class Wav2Letter_ResNet45(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_ResNet45, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use raw audio network with 45 layers-ResNet style")
		self.conv1 = nn.Conv1d(num_features, 256, 250, 40)
		self.relu1 = nn.ReLU() 
		self.bn1 = nn.BatchNorm1d(256)

		self.conv2 = nn.Conv1d(256, 64, 1) 
		self.relu2 = nn.ReLU() 
		self.bn2 = nn.BatchNorm1d(64)

		self.conv3 = nn.Conv1d(64, 256, 32, 2) 
		self.relu3 = nn.ReLU() 
		self.bn3 = nn.BatchNorm1d(256)
		
		#ResnetBlock -- 7 x 1
		self.resnet_block_1 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_1_relu = nn.ReLU() 
		self.resnet_1_bn = nn.BatchNorm1d(256)

		self.resnet_block_2 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_2_relu = nn.ReLU() 
		self.resnet_2_bn = nn.BatchNorm1d(256)

		self.resnet_block_3 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_3_relu = nn.ReLU() 
		self.resnet_3_bn = nn.BatchNorm1d(256)

		#ResnetBlock -- 5 x 1
		self.resnet_block_4 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_4_relu = nn.ReLU() 
		self.resnet_4_bn = nn.BatchNorm1d(256)

		self.resnet_block_5 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_5_relu = nn.ReLU() 
		self.resnet_5_bn = nn.BatchNorm1d(256)

		self.resnet_block_6 = nn.Sequential(
			nn.Conv1d(256, 64, 1), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), 
			nn.Conv1d(64, 256, 1)
		)
		self.resnet_6_relu = nn.ReLU() 
		self.resnet_6_bn = nn.BatchNorm1d(256)

		self.resnet_block_7 = nn.Sequential(
			nn.Conv1d(256, 128, 1), nn.ReLU(), nn.BatchNorm1d(128), 
			nn.Conv1d(128, 128, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(128), 
			nn.Conv1d(128, 128, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(128), 
			nn.Conv1d(128, 256, 1)
		)
		self.resnet_7_relu = nn.ReLU() 
		self.resnet_7_bn = nn.BatchNorm1d(256)

		self.conv13 = nn.Conv1d(256, 64, 1) 
		self.relu13 = nn.ReLU() 
		self.bn13 = nn.BatchNorm1d(64)
		self.conv14 = nn.Conv1d(64, 256, 16) 
		self.relu14 = nn.ReLU() 
		self.bn14 = nn.BatchNorm1d(256)

		self.conv15 = nn.Conv1d(256, 64, 1) 
		self.relu15 = nn.ReLU() 
		self.bn15 = nn.BatchNorm1d(64)
		self.conv16 = nn.Conv1d(64, 1000, 1) 
		self.relu16 = nn.ReLU() 
		self.bn16 = nn.BatchNorm1d(1000)

		self.conv17 = nn.Conv1d(1000, 1000, 1) 
		self.relu17 = nn.ReLU() 
		self.bn17 = nn.BatchNorm1d(1000) 

		self.classifier = nn.Conv1d(1000, num_classes, 1)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.bn1(self.relu1(self.conv1(batch)))
		batch = self.bn2(self.relu2(self.conv2(batch)))

		batch = self.bn3(self.relu3(self.conv3(batch)))

		resid = batch
		batch = self.resnet_block_1(batch)
		batch += resid
		batch = self.resnet_1_bn(self.resnet_1_relu(batch))

		resid = batch
		batch = self.resnet_block_2(batch)
		batch += resid
		batch = self.resnet_2_bn(self.resnet_2_relu(batch))

		resid = batch
		batch = self.resnet_block_3(batch)
		batch += resid
		batch = self.resnet_3_bn(self.resnet_3_relu(batch))

		resid = batch
		batch = self.resnet_block_4(batch)
		batch += resid
		batch = self.resnet_4_bn(self.resnet_4_relu(batch))

		resid = batch
		batch = self.resnet_block_5(batch)
		batch += resid
		batch = self.resnet_5_bn(self.resnet_5_relu(batch))

		resid = batch
		batch = self.resnet_block_6(batch)
		batch += resid
		batch = self.resnet_6_bn(self.resnet_6_relu(batch))

		resid = batch
		batch = self.resnet_block_7(batch)
		batch += resid
		batch = self.resnet_7_bn(self.resnet_7_relu(batch))

		batch = self.bn13(self.relu13(self.conv13(batch)))
		batch = self.bn14(self.relu14(self.conv14(batch)))
		batch = self.bn15(self.relu15(self.conv15(batch)))
		batch = self.bn16(self.relu16(self.conv16(batch)))
		batch = self.bn17(self.relu17(self.conv17(batch)))

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class Wav2Letter_pp(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_pp, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use mfcc-based Wav2Letter++ architecture for WSJ")
		self.layers = nn.Sequential(
			nn.utils.weight_norm(nn.Conv1d(num_features, 200, 13, 1, padding=6), dim=0),
			nn.GLU(dim=1), 
			nn.Dropout(0.25),
			
			nn.utils.weight_norm(nn.Conv1d(100, 200, 3, 1, padding=1), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(100, 200, 4, 1, padding=2), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(100, 250, 5, 1, padding=2), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(125, 300, 6, 1, padding=3), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(150, 350, 7, 1, padding=3), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(175, 400, 8, 1, padding=4), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(200, 450, 9, 1, padding=4), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(225, 500, 10, 1, padding=5), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 500, 11, 1, padding=5), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 500, 12, 1, padding=6), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 600, 13, 1, padding=6), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(300, 600, 14, 1, padding=7), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(300, 600, 15, 1, padding=7), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(175, 750, 21, 1, padding=10), dim=0), 
			nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(375, 1000, 1, 1), dim=0),
			nn.GLU(dim=1),
			nn.Dropout(0.25), 
		)
		

		self.classifier = nn.utils.weight_norm(nn.Conv1d(500, num_classes, 1), dim=0)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.layers(batch)

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			m.bias.data.fill_(0.01) 

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class GLUblock(nn.Module): 
	def __init__(self, k, in_c, out_c, bot_c=50, skip_connection=True, use_bottleneck=False): 
		super().__init__()
		self.skip_connection = skip_connection
		self.use_bottleneck = use_bottleneck
		if self.skip_connection:
			if in_c == out_c: 
				self.use_proj = 0
			else: 
				self.use_proj = 1
			self.convresid = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=1), dim=0)
		if self.use_bottleneck:
			self.convx1a = nn.utils.weight_norm(nn.Conv1d(in_c, bot_c, kernel_size=1), dim=0)
			self.convx2a = nn.utils.weight_norm(nn.Conv1d(in_c, bot_c, kernel_size=1), dim=0)

			self.convx1b = nn.utils.weight_norm(nn.Conv1d(bot_c, bot_c, kernel_size=k, padding=((k-1)//2)), dim=0)
			self.convx2b = nn.utils.weight_norm(nn.Conv1d(bot_c, bot_c, kernel_size=k, padding=((k-1)//2)), dim=0)

			self.convx1c = nn.utils.weight_norm(nn.Conv1d(bot_c, out_c, kernel_size=1), dim=0)
			self.convx2c = nn.utils.weight_norm(nn.Conv1d(bot_c, out_c, kernel_size=1), dim=0)
		else: 
			self.convx1 = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k, padding=((k-1)//2)), dim=0)
			self.convx2 = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k, padding=((k-1)//2)), dim=0)

	def forward(self, x): 
		if self.skip_connection:
			resid = x 
			if self.use_proj == 1: 
				resid = self.convresid(resid)
		if self.use_bottleneck:
			x1 = self.convx1c(self.convx1b(self.convx1a(x))) 
			x2 = self.convx2c(self.convx2b(self.convx2a(x))) 
		else: 
			x1 = self.convx1(x)
			x2 = self.convx2(x)
		x2 = torch.sigmoid(x2) 
		x = torch.mul(x1,x2) 
		if self.skip_connection: 
			return x + resid[:, :, :x.shape[2]] 
		return x

class Wav2Letter_pp_resid(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_pp_resid, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use mfcc-based Wav2Letter++ architecture for WSJ")
		self.layers = nn.Sequential(
			GLUblock(k=13, in_c=num_features, out_c=100, bot_c=num_features),
			nn.Dropout(0.25), 
			GLUblock(k=3, in_c=100, out_c=100, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=4, in_c=100, out_c=100, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=5, in_c=100, out_c=125, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=6, in_c=125, out_c=150, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=7, in_c=150, out_c=175, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=8, in_c=175, out_c=200, bot_c=50),
			nn.Dropout(0.25), 
			GLUblock(k=9, in_c=200, out_c=225, bot_c=100),
			nn.Dropout(0.25), 
			GLUblock(k=10, in_c=225, out_c=250, bot_c=100),
			nn.Dropout(0.25), 
			GLUblock(k=11, in_c=250, out_c=250, bot_c=100),
			nn.Dropout(0.25), 
			GLUblock(k=12, in_c=250, out_c=250, bot_c=100),
			nn.Dropout(0.25), 
			GLUblock(k=13, in_c=250, out_c=250, bot_c=100),
			nn.Dropout(0.25), 
			GLUblock(k=21, in_c=250, out_c=375, bot_c=100),
			nn.Dropout(0.25), 
			nn.utils.weight_norm(nn.Conv1d(375, 1000, 1, 1), dim=0),
			nn.GLU(dim=1),
			nn.Dropout(0.25), 
		)
		

		self.classifier = nn.utils.weight_norm(nn.Conv1d(500, num_classes, 1), dim=0)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.layers(batch)

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			m.bias.data.fill_(0.01) 

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class GCNN_WSJ(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True, skip_connection=False):
		super(GCNN_WSJ, self).__init__() 
		self.skip_connection = skip_connection
		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use mfcc-based GCNN architecture for WSJ")
		self.layers = nn.Sequential(
			GLUblock(k=13, in_c=num_features, out_c=100, bot_c=num_features, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=3, in_c=100, out_c=100, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=4, in_c=100, out_c=100, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=5, in_c=100, out_c=125, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=6, in_c=125, out_c=150, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=7, in_c=150, out_c=175, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=8, in_c=175, out_c=200, bot_c=50, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=9, in_c=200, out_c=225, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=10, in_c=225, out_c=250, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=11, in_c=250, out_c=250, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=12, in_c=250, out_c=250, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=13, in_c=250, out_c=250, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=14, in_c=250, out_c=300, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=15, in_c=300, out_c=300, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=21, in_c=300, out_c=375, bot_c=100, skip_connection=skip_connection),
			nn.Dropout(0.25), 
			GLUblock(k=1, in_c=375, out_c=500, bot_c=100, skip_connection=skip_connection),
			#nn.utils.weight_norm(nn.Conv1d(375, 1000, 1, 1), dim=0),
			#nn.GLU(dim=1),
			nn.Dropout(0.25), 
		)
		

		self.classifier = nn.utils.weight_norm(nn.Conv1d(500, num_classes, 1), dim=0)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.layers(batch)

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			m.bias.data.fill_(0.01) 

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class Wav2Letter_pp_conv(nn.Module): 
	def __init__(self, num_features, num_classes, raw_audio=True):
		super(Wav2Letter_pp_conv, self).__init__() 

		self.raw_audio = raw_audio
		#Conv1D: <in x out x kernel size x stride>
		print("Use mfcc-based Wav2Letter no GLU architecture for WSJ")
		self.layers = nn.Sequential(
			nn.utils.weight_norm(nn.Conv1d(num_features, 200, 13, 1, padding=6), dim=0),
			#nn.GLU(dim=1), 
			nn.Dropout(0.25),
			
			nn.utils.weight_norm(nn.Conv1d(200, 200, 3, 1, padding=1), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(200, 200, 4, 1, padding=2), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(200, 250, 5, 1, padding=2), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(250, 300, 6, 1, padding=3), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(300, 350, 7, 1, padding=3), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(175, 400, 8, 1, padding=4), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(200, 450, 9, 1, padding=4), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(225, 500, 10, 1, padding=5), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 500, 11, 1, padding=5), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 500, 12, 1, padding=6), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(250, 600, 13, 1, padding=6), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(300, 600, 14, 1, padding=7), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			# nn.utils.weight_norm(nn.Conv1d(300, 600, 15, 1, padding=7), dim=0), 
			# nn.GLU(dim=1), 
			# nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(350, 750, 21, 1, padding=10), dim=0), 
			#nn.GLU(dim=1), 
			nn.Dropout(0.25), 

			nn.utils.weight_norm(nn.Conv1d(750, 1000, 1, 1), dim=0),
			#nn.GLU(dim=1),
			nn.Dropout(0.25), 
		)
		

		self.classifier = nn.utils.weight_norm(nn.Conv1d(1000, num_classes, 1), dim=0)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def forward(self, batch): 
		"""
		Batch shape: (batch, num_features, frame_len)
		"""
		#y_pred = self.layers(batch) 
		batch = self.layers(batch)

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		return log_probs

	# def predict(self, sample): 
	# 	"""
	# 	Sample shape: num features x frame_len
	# 	"""
	# 	_input = sample.reshape(1, sample.shape[0], sample.shape[1])
	# 	log_prob = self.forward(_input) 
	# 	return log_prob

	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			m.bias.data.fill_(0.01) 

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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class DenseBlock(nn.Module):
	def __init__(self, in_c, kernel_size, growth_rate=12):
		super().__init__()
		self.glu_1 = GLUblock(kernel_size, in_c, growth_rate, skip_connection=True)
		self.glu_2 = GLUblock(kernel_size, in_c + growth_rate * 1, growth_rate, skip_connection=True)
		self.glu_3 = GLUblock(kernel_size, in_c + growth_rate * 2, growth_rate, skip_connection=True)
		self.glu_4 = GLUblock(kernel_size, in_c + growth_rate * 3, growth_rate, skip_connection=True)
		self.transition = GLUblock(1, in_c + growth_rate * 4, in_c, skip_connection=True)

	def forward(self, x): 
		x1 = self.glu_1(x)
		concat_1 = torch.cat((x, x1), 1) 

		x2 = self.glu_2(concat_1) 
		concat_2 = torch.cat((concat_1, x2), 1) 

		x3 = self.glu_3(concat_2)
		concat_3 = torch.cat((concat_2, x3), 1)

		x4 = self.glu_4(concat_3)
		concat_4 = torch.cat((concat_3, x4), 1)

		return self.transition(concat_4)

class GCNN_DenseNet(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=64, growth_rate=24):
		super(GCNN_DenseNet, self).__init__()

		print("Use mfcc-based GCNN architecture with DenseNet for WSJ")
		self.layers = nn.Sequential(
			GLUblock(k=13, in_c=num_features, out_c=dense_dim, skip_connection=True),
			DenseBlock(in_c=dense_dim, kernel_size=3, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=5, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=7, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=9, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=11, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=13, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=15, growth_rate=growth_rate),
			DenseBlock(in_c=dense_dim, kernel_size=21, growth_rate=growth_rate),
			nn.utils.weight_norm(nn.Conv1d(dense_dim, 500, kernel_size=1), dim=0),
		)
		self.classifier = nn.utils.weight_norm(nn.Conv1d(500, num_classes, kernel_size=1), dim=0)
		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)
		# self.glu_3 = GLUblock(k=13, in_c=num_features, out_c=growth_rate, skip_connection=False)
		# self.glu_4 = GLUblock(k=3, in_c=num_features + growth_rate*1, out_c=growth_rate, skip_connection=False)
		# self.glu_5 = GLUblock(k=5, in_c=num_features + growth_rate*2, out_c=growth_rate, skip_connection=False)
		# self.glu_6 = GLUblock(k=5, in_c=num_features + growth_rate*3, out_c=growth_rate, skip_connection=False)
		# self.glu_7 = GLUblock(k=7, in_c=num_features + growth_rate*4, out_c=growth_rate, skip_connection=False)
		# self.glu_8 = GLUblock(k=7, in_c=num_features + growth_rate*5, out_c=growth_rate, skip_connection=False)
		# self.glu_9 = GLUblock(k=9, in_c=num_features + growth_rate*6, out_c=growth_rate, skip_connection=False)
		# self.glu_10 = GLUblock(k=9, in_c=num_features + growth_rate*7, out_c=growth_rate, skip_connection=False)
		# self.glu_11 = GLUblock(k=11, in_c=num_features + growth_rate*8, out_c=growth_rate, skip_connection=False)
		# self.glu_12 = GLUblock(k=11, in_c=num_features + growth_rate*9, out_c=growth_rate, skip_connection=False)
		# self.glu_13 = GLUblock(k=13, in_c=num_features + growth_rate*10, out_c=growth_rate, skip_connection=False)
		# self.glu_14 = GLUblock(k=13, in_c=num_features + growth_rate*11, out_c=growth_rate, skip_connection=False)
		# self.glu_15 = GLUblock(k=15, in_c=num_features + growth_rate*12, out_c=growth_rate, skip_connection=False)
		# self.glu_16 = GLUblock(k=15, in_c=num_features + growth_rate*13, out_c=growth_rate, skip_connection=False)
		# self.glu_17 = GLUblock(k=17, in_c=num_features + growth_rate*14, out_c=growth_rate, skip_connection=False)
		# self.glu_18 = GLUblock(k=17, in_c=num_features + growth_rate*15, out_c=growth_rate, skip_connection=False)
		# self.glu_19 = GLUblock(k=19, in_c=num_features + growth_rate*16, out_c=growth_rate, skip_connection=False)
		# self.glu_20 = GLUblock(k=19, in_c=num_features + growth_rate*17, out_c=growth_rate, skip_connection=False)
		# self.glu_21 = GLUblock(k=21, in_c=num_features + growth_rate*14, out_c=growth_rate, skip_connection=False)

		# self.glu_last = GLUblock(k=1, in_c=num_features + growth_rate*15, out_c=1000, skip_connection=False) 
		# self.conv_1 = nn.utils.weight_norm(nn.Conv1d(num_features, growth_rate, kernel_size=3, padding=1), dim=0)
		# self.conv_2 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*1, growth_rate, kernel_size=5, padding=2), dim=0)
		# self.conv_3 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*2, growth_rate, kernel_size=7, padding=3), dim=0)
		# self.conv_4 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*3, growth_rate, kernel_size=9, padding=4), dim=0)
		# self.conv_5 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*4, growth_rate, kernel_size=11, padding=5), dim=0)
		# self.conv_6 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*5, growth_rate, kernel_size=13, padding=6), dim=0)
		# self.conv_7 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*6, growth_rate, kernel_size=15, padding=7), dim=0)
		# self.conv_8 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*7, growth_rate, kernel_size=17, padding=8), dim=0)
		# self.conv_9 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*8, growth_rate, kernel_size=19, padding=9), dim=0)
		# self.conv_10 = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate*9, growth_rate, kernel_size=21, padding=10), dim=0)

		# self.preclassifier = nn.utils.weight_norm(nn.Conv1d(num_features + growth_rate * 10, 1000, 1), dim=0)
		# self.classifier = nn.utils.weight_norm(nn.Conv1d(1000, num_classes, 1), dim=0)

		# self.bn_3 = nn.BatchNorm1d(growth_rate)
		# self.bn_4 = nn.BatchNorm1d(growth_rate)
		# self.bn_5 = nn.BatchNorm1d(growth_rate)
		# self.bn_6 = nn.BatchNorm1d(growth_rate)
		# self.bn_7 = nn.BatchNorm1d(growth_rate)
		# self.bn_8 = nn.BatchNorm1d(growth_rate)
		# self.bn_9 = nn.BatchNorm1d(growth_rate)
		# self.bn_10 = nn.BatchNorm1d(growth_rate)
		# self.bn_11 = nn.BatchNorm1d(growth_rate)
		# self.bn_12 = nn.BatchNorm1d(growth_rate)
		# self.bn_13 = nn.BatchNorm1d(growth_rate)
		# self.bn_14 = nn.BatchNorm1d(growth_rate)
		# self.bn_15 = nn.BatchNorm1d(growth_rate)
		# self.bn_16 = nn.BatchNorm1d(growth_rate)
		# # self.bn_17 = nn.BatchNorm1d(growth_rate)
		# # self.bn_18 = nn.BatchNorm1d(growth_rate)
		# # self.bn_19 = nn.BatchNorm1d(growth_rate)
		# # self.bn_20 = nn.BatchNorm1d(growth_rate)
		# self.bn_21 = nn.BatchNorm1d(growth_rate)
		# self.bn_last = nn.BatchNorm1d(1000)

		# self.bn_1 = nn.BatchNorm1d(growth_rate)
		# self.bn_2 = nn.BatchNorm1d(growth_rate)
		# self.bn_3 = nn.BatchNorm1d(growth_rate)
		# self.bn_4 = nn.BatchNorm1d(growth_rate)
		# self.bn_5 = nn.BatchNorm1d(growth_rate)
		# self.bn_6 = nn.BatchNorm1d(growth_rate)
		# self.bn_7 = nn.BatchNorm1d(growth_rate)
		# self.bn_8 = nn.BatchNorm1d(growth_rate)
		# self.bn_9 = nn.BatchNorm1d(growth_rate)
		# self.bn_10 = nn.BatchNorm1d(growth_rate)
		# self.bn_preclassifier = nn.BatchNorm1d(1000)

		# # self.bn_17 = nn.BatchNorm1d(growth_rate)
		# # self.bn_18 = nn.BatchNorm1d(growth_rate)
		# # self.bn_19 = nn.BatchNorm1d(growth_rate)
		# # self.bn_20 = nn.BatchNorm1d(growth_rate)
		# self.dropout_3 = nn.Dropout(0.2)
		# self.dropout_4 = nn.Dropout(0.2)
		# self.dropout_5 = nn.Dropout(0.2)
		# self.dropout_6 = nn.Dropout(0.2)
		# self.dropout_7 = nn.Dropout(0.2)
		# self.dropout_8 = nn.Dropout(0.2)
		# self.dropout_9 = nn.Dropout(0.2)
		# self.dropout_10 = nn.Dropout(0.2)
		# self.dropout_11 = nn.Dropout(0.2)
		# self.dropout_12 = nn.Dropout(0.2)
		# self.dropout_13 = nn.Dropout(0.2)
		# self.dropout_14 = nn.Dropout(0.2)
		# self.dropout_15 = nn.Dropout(0.2)
		# self.dropout_16 = nn.Dropout(0.2)
		# # self.dropout_17 = nn.Dropout(0.2)
		# # self.dropout_18 = nn.Dropout(0.2)
		# # self.dropout_19 = nn.Dropout(0.2)
		# # self.dropout_20 = nn.Dropout(0.2)
		# self.dropout_21 = nn.Dropout(0.2)
		# self.dropout_last = nn.Dropout(0.2)
	
	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			m.bias.data.fill_(0.01) 
	
	def forward(self, batch): 
		batch = self.layers(batch)

		y_pred = self.classifier(batch)

		log_probs = F.log_softmax(y_pred, dim=1)
		# c1 = self.bn_1(F.relu(self.conv_1(batch)))
		# concat_1 = torch.cat((batch, c1), 1)

		# c2 = self.bn_2(F.relu(self.conv_2(concat_1)))
		# concat_2 = torch.cat((concat_1, c2), 1)

		# c3 = self.bn_3(F.relu(self.conv_3(concat_2)))
		# concat_3 = torch.cat((concat_2, c3), 1)

		# c4 = self.bn_4(F.relu(self.conv_4(concat_3)))
		# concat_4 = torch.cat((concat_3, c4), 1)

		# c5 = self.bn_5(F.relu(self.conv_5(concat_4)))
		# concat_5 = torch.cat((concat_4, c5), 1)

		# c6 = self.bn_6(F.relu(self.conv_6(concat_5)))
		# concat_6 = torch.cat((concat_5, c6), 1)

		# c7 = self.bn_7(F.relu(self.conv_7(concat_6)))
		# concat_7 = torch.cat((concat_6, c7), 1)

		# c8 = self.bn_8(F.relu(self.conv_8(concat_7)))
		# concat_8 = torch.cat((concat_7, c8), 1)

		# c9 = self.bn_9(F.relu(self.conv_9(concat_8)))
		# concat_9 = torch.cat((concat_8, c9), 1)

		# c10 = self.bn_9(F.relu(self.conv_10(concat_9)))
		# concat_10 = torch.cat((concat_9, c10), 1)

		# precl = self.bn_preclassifier(F.relu(self.preclassifier(concat_10)))
		# y_pred = self.classifier(precl) 
		# log_probs = F.log_softmax(y_pred, dim=1)


		# glu_3 = self.bn_3(self.glu_3(batch))
		# concat_3 = torch.cat((batch, glu_3), 1)

		# glu_4 = self.bn_4(self.glu_4(concat_3))
		# concat_4 = torch.cat((concat_3, glu_4), 1)

		# glu_5 = self.bn_5(self.glu_5(concat_4))
		# concat_5 = torch.cat((concat_4, glu_5), 1)

		# glu_6 = self.bn_6(self.glu_6(concat_5))
		# concat_6 = torch.cat((concat_5, glu_6), 1)

		# glu_7 = self.bn_7(self.glu_7(concat_6))
		# concat_7 = torch.cat((concat_6, glu_7), 1)

		# glu_8 = self.bn_8(self.glu_8(concat_7))
		# concat_8 = torch.cat((concat_7, glu_8), 1)

		# glu_9 = self.bn_9(self.glu_9(concat_8))
		# concat_9 = torch.cat((concat_8, glu_9), 1)

		# glu_10 = self.bn_10(self.glu_10(concat_9))
		# concat_10 = torch.cat((concat_9, glu_10), 1)

		# glu_11 = self.bn_11(self.glu_11(concat_10))
		# concat_11 = torch.cat((concat_10, glu_11), 1)

		# glu_12 = self.bn_12(self.glu_12(concat_11))
		# concat_12 = torch.cat((concat_11, glu_12), 1)

		# glu_13 = self.bn_13(self.glu_13(concat_12))
		# concat_13 = torch.cat((concat_12, glu_13), 1)

		# glu_14 = self.bn_14(self.glu_14(concat_13))
		# concat_14 = torch.cat((concat_13, glu_14), 1)

		# glu_15 = self.bn_15(self.glu_15(concat_14))
		# concat_15 = torch.cat((concat_14, glu_15), 1)

		# glu_16 = self.bn_16(self.glu_16(concat_15))
		# concat_16 = torch.cat((concat_15, glu_16), 1)

		# # glu_17 = self.dropout_17(self.bn_17(self.glu_17(concat_16)))
		# # concat_17 = torch.cat((concat_16, glu_17), 1)

		# # glu_18 = self.dropout_18(self.bn_18(self.glu_18(concat_17)))
		# # concat_18 = torch.cat((concat_17, glu_18), 1)

		# # glu_19 = self.dropout_19(self.bn_19(self.glu_19(concat_18)))
		# # concat_19 = torch.cat((concat_18, glu_19), 1)

		# # glu_20 = self.dropout_20(self.bn_20(self.glu_20(concat_19)))
		# # concat_20 = torch.cat((concat_19, glu_20), 1)

		# glu_21 = self.bn_21(self.glu_21(concat_16))
		# concat_21 = torch.cat((concat_16, glu_21), 1)

		# glu_last = self.bn_last(self.glu_last(concat_21))
		# #concat_last = torch.cat((concat_21, glu_last), 1)

		# y_pred = self.classifier(glu_last)
		# log_probs = F.log_softmax(y_pred, dim=1)
		
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
			package['cer_results'] = cer_results 
			package['wer_results'] = wer_results 
		if meta is not None: 
			package['meta'] = meta
		return package

class BottleneckBlock(nn.Module): 
	def __init__(self, in_c, kernel_size, bottleneck_depth):
		super().__init__()
		self.conv1 = nn.Conv1d(in_c, bottleneck_depth, kernel_size=1)
		self.conv2 = nn.Conv1d(bottleneck_depth, bottleneck_depth, kernel_size=kernel_size, padding=((kernel_size-1)//2))
		self.conv3 = nn.Conv1d(bottleneck_depth, in_c, kernel_size=1)

		self.bn1 = nn.BatchNorm1d(bottleneck_depth)
		self.bn2 = nn.BatchNorm1d(bottleneck_depth)
		self.bn3 = nn.BatchNorm1d(in_c)

	def forward(self, x): 
		x = self.bn1(F.relu(self.conv1(x)))
		x = self.bn2(F.relu(self.conv2(x)))
		x = self.bn3(F.relu(self.conv3(x)))
		return x

class ResNeXTBlock(nn.Module):
	def __init__(self, in_c, bottleneck_depth):
		super().__init__()
		self.b_block_1 = BottleneckBlock(in_c, 3, bottleneck_depth) 
		self.b_block_1b = BottleneckBlock(in_c, 5, bottleneck_depth) #Not in original
		self.b_block_2 = BottleneckBlock(in_c, 7, bottleneck_depth)
		self.b_block_2b = BottleneckBlock(in_c, 9, bottleneck_depth) #Not in original
		self.b_block_3 = BottleneckBlock(in_c, 11, bottleneck_depth)
		self.b_block_3b = BottleneckBlock(in_c, 13, bottleneck_depth) #Not in original
		self.b_block_4 = BottleneckBlock(in_c, 15, bottleneck_depth)
		self.b_block_4b = BottleneckBlock(in_c, 17, bottleneck_depth) #Not in original
		self.b_block_5 = BottleneckBlock(in_c, 19, bottleneck_depth)

	def forward(self, x): 
		x1 = self.b_block_1(x)
		x2 = self.b_block_2(x)
		x3 = self.b_block_3(x)
		x4 = self.b_block_4(x)
		x5 = self.b_block_5(x)
		x1b = self.b_block_1b(x)
		x2b = self.b_block_2b(x)
		x3b = self.b_block_3b(x)
		x4b = self.b_block_4b(x)
		combo_x = x1 + x2 + x3 + x4 + x5 + x1b + x2b + x3b + x4b
		#print("Input x", x.shape)
		#print("Combo", (x + combo_x).shape)
		return x + combo_x

class Conv_BN_Act(nn.Module): 
	def __init__(self, in_c, out_c, kernel_size, stride=1, dilation=1, dropout=0.2):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv1d(in_c, out_c, kernel_size, stride=stride, dilation=dilation, padding=(kernel_size*dilation)//2),
			nn.BatchNorm1d(out_c), 
			nn.Dropout(dropout),
			nn.ReLU6()
		)

	def forward(self, x): 
		return self.layers(x)

class Wav2Letter_NVIDIA(nn.Module): 
	def __init__(self, in_c, num_classes): 
		super().__init__() 
		self.layers = nn.Sequential(
			Conv_BN_Act(in_c, 256, 11, stride = 2), 
			Conv_BN_Act(256, 256, 11), 
			Conv_BN_Act(256, 256, 11),
			Conv_BN_Act(256, 256, 11),

			Conv_BN_Act(256, 384, 13),
			Conv_BN_Act(384, 384, 13),
			Conv_BN_Act(384, 384, 13),

			Conv_BN_Act(384, 512, 17),
			Conv_BN_Act(512, 512, 17),
			Conv_BN_Act(512, 512, 17),

			Conv_BN_Act(512, 640, 21),
			Conv_BN_Act(640, 640, 21),
			Conv_BN_Act(640, 640, 21),

			Conv_BN_Act(640, 768, 25),
			Conv_BN_Act(768, 768, 25),
			Conv_BN_Act(768, 768, 25),

			Conv_BN_Act(768, 896, 29), 

			Conv_BN_Act(896, 1024, 1),	
		)
		self.classifier = Conv_BN_Act(1024, num_classes, 1)
		self.layers.apply(self.init_weights) 
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		batch = self.layers(batch)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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


class ResNextASR(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(ResNextASR, self).__init__() 
		print("Using ResNexT like architecture")
		self.layers = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),

			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		#print(batch.shape) 
		batch = self.layers(batch)
		#print(batch.shape)
		#exit()        
		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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

class ResNextASR_v2(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(ResNextASR_v2, self).__init__() 
		print("Using ResNexT like architecture")
		self.layers = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		batch = self.layers(batch)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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

class CausalConvolution1D(nn.Module):
	def __init__(self, input_dim, output_dim, kernel_size, stride):
		"""
		Perform causal convolution by padding on the left
		"""
		super(CausalConvolution1D, self).__init__() 

		#pad k-1 
		self.leftpad = nn.ConstantPad1d((kernel_size-1,0), 0)
		self.conv = nn.utils.weight_norm(nn.Conv1d(input_dim, output_dim, kernel_size, stride), dim=0)

	def forward(self, x):
		# print("x before pad: {}".format(x.shape))
		x = self.leftpad(x) 
		# print("x after pad: {}".format(x.shape))
		x = self.conv(x) 
		return x

class Wav2Vec_ResNeXT(nn.Module): 
	def __init__(self, timestep, batch_size, z_dim, encoder_bottleneck_depth, c_dim, num_features, num_classes, num_negatives=10, max_crop=100000):
		"""Wav2Vec: A self-supervised scheme described in this paper: https://arxiv.org/pdf/1904.05862.pdf
		timestep: how many timestep in the future we are trying to predict
		batch_size: how many utterances in a batch -- batch_size - 1 is # of negatives
		seq_len: length of the input audio (= sampling rate x length)
		z_dim: encoding dimension
		c_dim: context dimension
		"""
		super(Wav2Vec_ResNeXT, self).__init__() 

		self.batch_size = batch_size 
		self.timestep = timestep #How many timestep to predict into the future 
		# self.seq_len = seq_len #Maximum audio length per utterance 
		self.z_dim = z_dim 
		self.c_dim = c_dim
		self.num_negatives = num_negatives
		self.max_crop = max_crop

		self.encoder = nn.Sequential(
			nn.Conv1d(num_features, z_dim, kernel_size=13),
			nn.BatchNorm1d(z_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			ResNeXTBlock(z_dim, encoder_bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(z_dim, encoder_bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(z_dim, encoder_bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(z_dim, encoder_bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(z_dim, encoder_bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(z_dim, z_dim, kernel_size=1),
			nn.BatchNorm1d(z_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)

		#9 layers of 512 deep, 3 wide, stride 1
		self.context_network = nn.Sequential(
			CausalConvolution1D(z_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			CausalConvolution1D(c_dim, c_dim, 3, 1), 
			nn.GroupNorm(1, c_dim), 
			nn.ReLU(),
			# CausalConvolution1D(c_dim, c_dim, 3, 1), 
			# nn.GroupNorm(1, c_dim), 
			# nn.ReLU(),
		)

		self.Wk = nn.ModuleList([nn.Linear(c_dim, z_dim) for i in range(timestep)])
		self.speech_classifier = nn.Conv1d(z_dim, num_classes, kernel_size=1) 

		self.sigmoid = nn.Sigmoid()
		self.criterion = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(dim=1) 
		self.lsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x): 
		"""
		x shape: [N x C x L] or [batch x 1 x audio_length] 
		"""
		batch = x.size()[0]
		#<batch> -- this will be used to compute loss
		targets = torch.zeros(batch).long() 

		#generate an array/tensor of integers from 0 to num_negatives, representing index
		#of the positive
		positive_idx = torch.randint(low=0, high=self.num_negatives+1, size=(batch,))

		#Put a 1 in the corresponding place in targets
		for i in range(batch): 
			targets[i] = positive_idx[i]

		#print("Targets: {}".format(targets))
		#crop audio: assume the batch are ordered in size (longest to shortest)
		# - decide whether to crop at start or at end 
		# - decide how many to crop: maximum is 100k, or the length of shortest
		#put this in the data loader collate function, no? 

		#Obtain the encoding for everything 
		z_encoding = self.encoder(x)

		#z_encoding = <N x C x L> 
		#Select a random place in the sequence to be t = 0
		t_samples = torch.randint(low=1, high=int(z_encoding.shape[2])-self.timestep, size=(1,)).long()

		#This is the true positives
		true_future_z = torch.empty((batch, self.z_dim, self.timestep)).float()

		for i in range(self.timestep): 
			#At each timestep, obtain the z-encoding for all batch, and squeeze into 2 dimensions
			true_future_z[:, :, i] = z_encoding[:, :, t_samples+1].view(batch, self.z_dim)

		#Select negative samples: 
		negative_idxs = torch.randint(low=0, high=t_samples.item(), size=(self.num_negatives,)).long()
		negative_z = torch.empty((batch, self.z_dim, self.num_negatives)).float()
		for i in range(self.num_negatives):
			negative_z[:, :, i] = z_encoding[:,:,negative_idxs[i]]

		known_z = z_encoding[:,:, t_samples]
		predicted_context = self.context_network(known_z)

		predicted_encoding = torch.empty((batch, self.z_dim, self.timestep)).float() 
		for i in range(self.timestep): 
			linear = self.Wk[i] #Wk is a FC with <z_dim x c_dim>
			#Apply the FC on the last available context in each batch
			predicted_encoding[:, :, i] = linear(predicted_context[:, :, -1].view(batch, self.c_dim))

		#Predicted_encoding: <batch x z_dim x timestep> -- Positive: <batch x z_dim x timestep> 
		#Negative: <batch x z_dim x num_negative>
		loss_val = 0
		num_correct = 0
		for i in range(self.timestep): 
			#Compute matmul of true_z and predicted z
			logits = torch.empty((batch, self.num_negatives+1)).float()
			for b in range(batch): 
				curr_false_idx = 0
				for j in range(self.num_negatives+1): 
					if j == positive_idx[b]: 
						logit = torch.matmul(true_future_z[b,:,i], predicted_encoding[b, :, i])
					else: 
						logit = torch.matmul(negative_z[b, :, curr_false_idx], predicted_encoding[b, :, i]) 
						curr_false_idx+=1 
					logits[b, j] = logit
			# for j in range(self.num_negatives+1): 
			# 	if j == positive_idx: 
			# 		#If j is the true positive, the logit values is mm(true z, predicted)
			# 		logit = torch.diag(torch.mm(true_future_z[:, :, i], predicted_encoding[:,:,i].transpose(0,1)))
			# 	else: 
			# 		logit = torch.diag(torch.mm(negative_z[:, :, j], predicted_encoding[:, :, i].transpose(0,1)))
			# 	#Now, put that in the logits 
			# 	logits[:, j] = logit
			
			#Compute the accuracy for this timestep
			num_correct += torch.sum(torch.eq(torch.argmax(self.softmax(logits), dim=1), targets)).item()
			# if i == 0:
			# 	print("Logits: {}".format(self.softmax(logits))) 
			# 	print("Arg max logits: {}".format(torch.argmax(self.softmax(logits), dim=1)))
			# 	print("Num correct: {}".format(num_correct))
			#By default, loss is averaged over the batch -- this is fine
			loss_val += self.criterion(logits, targets)
		accuracy = num_correct / (batch * self.timestep)

		return accuracy, loss_val
			
	def predict(self, x):
		x = self.encoder(x) 
		# x = self.context_network(x)
		return x 

	def forward_asr(self, batch):
		batch = self.forward(batch)
		y_pred = self.speech_classifier(batch)
		return F.log_softmax(y_pred, dim=1)

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


class ResNextASR_v3(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(ResNextASR_v3, self).__init__() 
		print("Using ResNexT like architecture")
		self.layers = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, 10, 5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, 8, 4), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			#This is the old ResNeXT
			nn.Conv1d(dense_dim, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)

		self.layers.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		batch = self.layers(batch)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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

class ResNextASR_context(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=512, bottleneck_depth=32):
		super(ResNextASR_context, self).__init__() 
		print("Using ResNexT like architecture")
		self.embedding_layers= nn.Sequential(
			#This is the old ResNeXT
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			# GLUblock(k=13, in_c=num_features, out_c=dense_dim),
			# nn.Dropout(0.25), 
			# GLUblock(k=5, in_c=dense_dim, out_c=dense_dim),
			# nn.Dropout(0.25)
		)

		self.context_conv = nn.Sequential(
			nn.Conv1d(dense_dim, dense_dim, kernel_size=41, padding=20), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
			# ResNeXTBlock(dense_dim, bottleneck_depth),
			# nn.Dropout(0.25),
			# nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=10), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 
			# nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=10), 
			# nn.ReLU(), 
			# nn.Dropout(0.25)
		)

		self.wide_blocks = nn.Sequential(
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim*2, num_classes, kernel_size=1)

		self.embedding_layers.apply(self.init_weights)
		self.context_conv.apply(self.init_weights)
		self.wide_blocks.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		batch = self.embedding_layers(batch)
		
		context = self.context_conv(batch) 
		batch = self.wide_blocks(batch)

		batch = torch.cat((batch, context), 1)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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

class ResNextASR_context_v2(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(ResNextASR_context_v2, self).__init__() 
		print("Using ResNexT like architecture")
		self.embedding_layers= nn.Sequential(
			#This is the old ResNeXT
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
		)

		self.context_conv = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=21, padding=10), 
			nn.ReLU(), 
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
		)

		self.wide_blocks = nn.Sequential(
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim*2, num_classes, kernel_size=1)

		self.embedding_layers.apply(self.init_weights)
		self.context_conv.apply(self.init_weights)
		self.wide_blocks.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		context = self.context_conv(batch) 
		batch = self.embedding_layers(batch)

		batch = self.wide_blocks(batch)

		batch = torch.cat((batch, context), 1)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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

class ResNextASR_context_v3(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(ResNextASR_context_v3, self).__init__() 
		print("Using ResNexT like architecture")
		self.embedding_layers= nn.Sequential(
			#This is the old ResNeXT
			nn.Conv1d(num_features, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
		)

		self.context_conv = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=21, padding=20, dilation=2), 
			nn.ReLU(), 
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
		)

		self.post_emb_context_conv = nn.Sequential(
			nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=10), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
		)

		self.wide_blocks = nn.Sequential(
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim*3, num_classes, kernel_size=1)

		self.embedding_layers.apply(self.init_weights)
		self.context_conv.apply(self.init_weights)
		self.post_emb_context_conv.apply(self.init_weights)
		self.wide_blocks.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		context = self.context_conv(batch) 
		batch = self.embedding_layers(batch)

		post_emb_context = self.post_emb_context_conv(batch)
		batch = self.wide_blocks(batch)

		batch = torch.cat((batch, context, post_emb_context), 1)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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


class ResNextASR_context_raw(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=512, bottleneck_depth=32):
		super(ResNextASR_context_raw, self).__init__() 
		print("Using ResNexT like architecture")

		self.feature_extractor = nn.Sequential(
			# nn.Conv1d(num_features, dense_dim, 10, 5), 
			# nn.BatchNorm1d(dense_dim), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 

			# nn.Conv1d(dense_dim, dense_dim, 8, 4), 
			# nn.BatchNorm1d(dense_dim), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 

			# nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			# nn.BatchNorm1d(dense_dim), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 

			# nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			# nn.BatchNorm1d(dense_dim), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 

			# nn.Conv1d(dense_dim, dense_dim, 4, 2), 
			# nn.BatchNorm1d(dense_dim), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 

			# GLUblock(k=11, in_c=num_features, out_c=dense_dim//2), 
			# nn.MaxPool1d(11, stride=5), 
			# nn.Dropout(0.25),
			# GLUblock(k=9, in_c=dense_dim//2, out_c=dense_dim//2), 
			# nn.MaxPool1d(9, stride=4),
			# nn.Dropout(0.25),
			# GLUblock(k=5, in_c=dense_dim//2, out_c=dense_dim//2), 
			# nn.MaxPool1d(5, stride=2),
			# nn.Dropout(0.25),
			# GLUblock(k=5, in_c=dense_dim//2, out_c=dense_dim), 
			# nn.MaxPool1d(5, stride=2),
			# nn.Dropout(0.25), 
			# GLUblock(k=5, in_c=dense_dim, out_c=dense_dim), 
			# nn.MaxPool1d(5, stride=2),
			# nn.Dropout(0.25)
			nn.Conv1d(num_features, dense_dim, 250, 40), 
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim),  
			nn.Conv1d(dense_dim, dense_dim, 48, 2),
			nn.ReLU(), 
			nn.BatchNorm1d(dense_dim), 
			nn.Conv1d(dense_dim, dense_dim, 7), 
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim),
		)

		self.embedding_layers= nn.Sequential(
			#This is the old ResNeXT
			nn.Conv1d(dense_dim, dense_dim, kernel_size=13),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=5), 
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25), 

			# GLUblock(k=13, in_c=dense_dim, out_c=dense_dim),
			# nn.Dropout(0.25), 
			# GLUblock(k=5, in_c=dense_dim, out_c=dense_dim),
			# nn.Dropout(0.25)
		)

		self.context_conv = nn.Sequential(
			nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=20, dilation=2), 
			nn.ReLU(), 
			nn.Dropout(0.25), 
			# ResNeXTBlock(dense_dim, bottleneck_depth),
			# nn.Dropout(0.25),
			# nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=10), 
			# nn.ReLU(), 
			# nn.Dropout(0.25), 
			# nn.Conv1d(dense_dim, dense_dim, kernel_size=21, padding=10), 
			# nn.ReLU(), 
			# nn.Dropout(0.25)
		)

		self.wide_blocks = nn.Sequential(
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth), 
			nn.Dropout(0.25), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(), 
			nn.Dropout(0.25),
		)
		self.classifier = nn.Conv1d(dense_dim*2, num_classes, kernel_size=1)

		self.feature_extractor.apply(self.init_weights)
		self.embedding_layers.apply(self.init_weights)
		self.context_conv.apply(self.init_weights)
		self.wide_blocks.apply(self.init_weights)
		self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		batch = self.feature_extractor(batch)
		batch = self.embedding_layers(batch)
		
		context = self.context_conv(batch) 
		batch = self.wide_blocks(batch)

		batch = torch.cat((batch, context), 1)

		y_pred = self.classifier(batch)
		log_probs = F.log_softmax(y_pred, dim=1)
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


class DNNASR(nn.Module): 
	def __init__(self, num_features, num_classes, dense_dim=256, bottleneck_depth=16):
		super(DNNASR, self).__init__() 
		print("Using DNNASR like architecture")
		#self.conv1 = nn.Linear(in_features=39*140, out_features=dense_dim)
		#self.bn1 = nn.BatchNorm1d(dense_dim) 
		self.layers = nn.Sequential(
			nn.Linear(in_features=39*140, out_features=dense_dim),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),

			nn.Linear(in_features=dense_dim, out_features=dense_dim),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(),
			nn.Dropout(0.25),
			
			nn.Linear(in_features=dense_dim, out_features=dense_dim),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),

			nn.Linear(in_features=dense_dim, out_features=dense_dim),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),
			
			nn.Linear(in_features=dense_dim, out_features=dense_dim),
			nn.ReLU(),
			nn.BatchNorm1d(dense_dim), 
			nn.Dropout(0.25),
			
			nn.Linear(in_features=dense_dim, out_features=68*128),
			nn.BatchNorm1d(68*128), 
			nn.ReLU()

		)
		#self.classifier = nn.Conv1d(dense_dim, num_classes, kernel_size=1)

		self.layers.apply(self.init_weights)
		#self.classifier.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch):
		print(batch.size())
		#print(batch.reshape(-1, 39*140).size())
		#print(self.layers(batch.reshape(-1, 39*140)).size())
		#exit()
		y_pred = self.layers(batch.reshape(-1, batch.size()[1]*batch.size()[2]))
		print("Prediction: ", y_pred.size())
		#exit()		
		log_probs = F.log_softmax(y_pred.reshape(-1, 68, 128), dim=1)
		print("Prob", log_probs.size())
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