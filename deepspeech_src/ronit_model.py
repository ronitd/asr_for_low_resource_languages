import math 
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import random
import copy

supported_rnns = {
	'lstm': nn.LSTM, 
	'rnn': nn.RNN, 
	'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

class Seq2SeqEncoder(nn.Module):
	def __init__(self):
		super(Seq2SeqEncoder, self).__init__()
		self.hidden_size_encoder = 350
		self.num_layers_encoder = 4
		self.bidirectional_encoder = True
		
		self.hidden_size_decoder = 768
		self.num_layers_decoder = 2
		self.bidirectional_decoder = False 
 
		self.encoder = nn.LSTM(input_size = 80*7, hidden_size = self.hidden_size_encoder, num_layers = self.num_layers_encoder, batch_first = False, bidirectional = self.bidirectional_encoder)
		self.linear = nn.Linear(self.hidden_size_encoder*2*self.num_layers_encoder, self.hidden_size_decoder*2)
		self.enc_out = nn.LSTM(input_size = self.hidden_size_encoder, hidden_size = self.hidden_size_decoder, num_layers = self.num_layers_decoder, bidirectional = self.bidirectional_decoder)
	
	def forward(self, x):
		n = x.shape[0]
		t = x.shape[1]
		x = x.view(t, n, -1)
		#print("Input: ", x.shape)
		#print("Batch Size: ", x.shape[1])
		h_i = torch.randn((self.num_layers_encoder*(1 + int(self.bidirectional_encoder)), x.shape[1], self.hidden_size_encoder)).cuda()
		c_i = torch.randn((self.num_layers_encoder*(1 + int(self.bidirectional_encoder)), x.shape[1], self.hidden_size_encoder)).cuda()
		#print("Hidden shape: ", h_i.shape)
		#h_d = torch.randn((self.num_layers_decoder*(1 + int(self.bidirectional_decoder)), x.shape[1], self.hidden_size_decoder)).cuda()
		#c_d = torch.randn((self.num_layers_decoder*(1 + int(self.bidirectional_decoder)), x.shape[1], self.hidden_size_decoder)).cuda()
		
		
		x, (h, c) = self.encoder(x, (h_i, c_i))
		#h = h.view(-1, n, 2*self.hidden_size_encoder)
		c = c.view(n, -1)
		h = h.view(n, -1)
		h = self.linear(h).view(2, n, -1)
		c = self.linear(c).view(2, n, -1)
		#print("Encoder Shape :", x.shape)
		#print("Hidden State :", h.shape)
		#print("C State :", c.shape)
		'''
		out, (h_o, c_o) = self.enc_out(h, (h_d, c_d))
		#print("Decoder Shape :", out.shape)
		#print("Hidden State :", h_o.shape)
		#print("C State :", c_o.shape)
		'''	
		return x, h, c


class Seq2SeqAttention(nn.Module):
	def __init__(self):
		super(Seq2SeqAttention, self).__init__()
		self.linear = nn.Linear(700, 768)
	def forward(self, context, query):
		#print("Query Shape : ", query.shape)
		#print("Context Shape : ", context.shape)

		batch_size = context.shape[1]
		seq_len = context.shape[0]
		new_context = self.linear(context.view(context.shape[0]*context.shape[1], -1)).view(batch_size, -1, 768)
		#print("New Context Shape: ", new_context.shape)
		attention = torch.bmm(new_context, query.unsqueeze(2)).squeeze(2)
		#print("Attention bmm Shape: ", attention.shape)		
		attention = nn.functional.softmax(attention, dim=1)
		#print("Attention bmm Shape after softmax: ", attention.shape)
		out = torch.bmm(attention.unsqueeze(1), new_context).squeeze(1)
		#print("Out :", out.shape)
		return out


class Seq2SeqDecoder(nn.Module):
	def __init__(self, char_size):
		super(Seq2SeqDecoder, self).__init__()
		self.char_size = char_size 
		self.hidden_dim = 256
		self.hidden_dim_lstm = 768
		self.attention = Seq2SeqAttention()
		self.embedding = nn.Embedding(self.char_size, self.hidden_dim, padding_idx=0)
		
		self.lstm1 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim_lstm)
		self.lstm2 = nn.LSTMCell(input_size=self.hidden_dim_lstm, hidden_size=self.hidden_dim_lstm)
		
		
		self.char_prob = nn.Linear(self.hidden_dim_lstm*2, self.char_size)
	def forward(self, input, h, c, context):
		#print("In Decoder")
		embedded = self.embedding(input)
		#print("Embedding Shape: ", embedded.shape)
		#print("Hidden State: ", h[0,:,:].shape)
		hidden_1, cell_1 = self.lstm1(embedded, (h[0,:,:], c[0,:,:]))
		hidden_2, cell_2 = self.lstm2(hidden_1, (h[1,:,:], c[1,:,:]))
		attention = self.attention(context, hidden_2)
		#print("Hidden Shape: ", hidden_2.shape)
		x_attention = torch.cat([attention, hidden_2], dim=1)
		pred = self.char_prob(x_attention)
		#print("Hidden_1 Shape: ", hidden_1.shape)
		#print("Hidden_2 Shape: ", hidden_2.shape)
		#print("Cell 1 Shape: ", cell_1.shape)
		#print("Cell 2 Shape: ", cell_2.shape)
		#exit()
		hidden = torch.stack((hidden_1, hidden_2))
		cell = torch.stack((cell_1, cell_2))
		#print("Hidden :", hidden.shape)
		#print("Cell :", cell.shape)
		#exit()
		return 	pred, hidden, cell

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
		
	def forward(self, x, y, targets, teacher_forcing_ratio = 0.5, is_train=True):
		context, hidden, cell = self.encoder(x)
		batch_size = targets.shape[1]
		trg_len = targets.shape[0]
		trg_vocab_size = self.decoder.char_size
		#tensor to store decoder outputs
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		outputs[0,:,trg_vocab_size-2] = 1.0
		input = targets[0,:]
		#print("Input: ", input)
		#exit()
		for t in range(1, trg_len):
			#insert input token embedding, previous hidden and previous cell states
            		#receive output tensor (predictions) and new hidden and cell states
            		output, hidden, cell = self.decoder(input, hidden, cell, context)
            		#mask = targets[t] != -1
            		#place predictions in a tensor holding predictions for each token
            		outputs[t] = output
            		#print("Output :", output.shape)
            		#decide if we are going to use teacher forcing or not
            		teacher_force = random.random() < teacher_forcing_ratio
            
            		#get the highest predicted token from our predictions
            		top1 = output.argmax(1) 
            
            		#if teacher forcing, use actual next token as next input
            		#if not, use predicted token
            		input = targets[t] if teacher_force else top1
		return hidden, outputs
		

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

	def init_weights(m):
		for name, param in m.named_parameters():
			nn.init.uniform_(param.data, -0.08, 0.08)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class FeatureTransform(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        self.fc = nn.Linear(feature_size, d_model)
    def forward(self, x):
        return self.fc(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout = 0.1):
		super().__init__()
        
		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads
        
		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)
    
	def forward(self, q, k, v, mask=None):
        
		bs = q.size(0)
        
        	# perform linear operation and split into h heads
        
		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        	# transpose to get dimensions bs * h * sl * d_model
       
		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)# calculate attention using function we will define next
		scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        	# concatenate heads and put through final linear layer
		concat = scores.transpose(1,2).contiguous()\
        	.view(bs, -1, self.d_model)
        
		output = self.out(concat)
    
		return output

def attention(q, k, v, d_k, mask=None, dropout=None):
	scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
	if mask is not None:
        	mask = mask.unsqueeze(1)
        	scores = scores.masked_fill(mask == 0, -1e9)
	scores = F.softmax(scores, dim=-1)
    
	if dropout is not None:
		scores = dropout(scores)
        
	output = torch.matmul(scores, v)
	return output

class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff=2048, dropout = 0.1):
		super().__init__() 
		# We set d_ff as a default to 2048
		self.linear_1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_ff, d_model)
	def forward(self, x):
		x = self.dropout(F.relu(self.linear_1(x)))
		x = self.linear_2(x)
		return x

class Norm(nn.Module):
	def __init__(self, d_model, eps = 1e-6):
		super().__init__()
    
		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        	/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm	

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout = 0.1):
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.attn = MultiHeadAttention(heads, d_model)
		self.ff = FeedForward(d_model)
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
        
	def forward(self, x, mask):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.ff(x2))
		return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout=0.1):
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.norm_3 = Norm(d_model)
        
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)
        
		self.attn_1 = MultiHeadAttention(heads, d_model)
		self.attn_2 = MultiHeadAttention(heads, d_model)
		self.ff = FeedForward(d_model).cuda()
	
	def forward(self, x, e_outputs, src_mask, trg_mask):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
		x2 = self.norm_3(x)
		x = x + self.dropout_3(self.ff(x2))
		return x
# We can then build a convenient cloning function that can generate multiple layers:
	def get_clones(module, N):
		return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
	def __init__(self, vocab_size, d_model, N, heads):
		super().__init__()
		self.N = N
		self.embed = FeatureTransform(vocab_size, d_model)
		self.pe = PositionalEncoder(d_model)
		self.layers = get_clones(EncoderLayer(d_model, heads), N)
		self.norm = Norm(d_model)
	def forward(self, src, mask):
		x = self.embed(src)
		x = self.pe(x)
		for i in range(N):
			x = self.layers[i](x, mask)
		return self.norm(x)
    
class Decoder(nn.Module):
	def __init__(self, vocab_size, d_model, N, heads):
		super().__init__()
		self.N = N
		self.embed = Embedder(vocab_size, d_model)
		self.pe = PositionalEncoder(d_model)
		self.layers = get_clones(DecoderLayer(d_model, heads), N)
		self.norm = Norm(d_model)
	def forward(self, trg, e_outputs, src_mask, trg_mask):
		x = self.embed(trg)
		x = self.pe(x)
		for i in range(self.N):
			x = self.layers[i](x, e_outputs, src_mask, trg_mask)
		return self.norm(x)

class Transformer(nn.Module):
	def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
		super().__init__()
		self.encoder = Encoder(src_vocab, d_model, N, heads)
		self.decoder = Decoder(trg_vocab, d_model, N, heads)
		self.out = nn.Linear(d_model, trg_vocab)
	def forward(self, src, trg, src_mask, trg_mask):
		e_outputs = self.encoder(src, src_mask)
		d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
		output = self.out(d_output)
		return output
	def init_weights(m):
		for p in m.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
# we don't perform softmax on the output as this will be handled 
# automatically by our loss function