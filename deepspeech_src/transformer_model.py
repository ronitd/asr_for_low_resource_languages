import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, src_feature_size, ntoken, nhead, nhid, nlayers, dropout=0.5, decoder_d_model=2048):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Transformer
        self.encoder_d_model = src_feature_size
        self.decoder_d_model = decoder_d_model 
        self.noken = ntoken
        self.model_type = 'Transformer'
        self.trg_mask = None
        self.encoder_pre = PreProcessing()
        self.enc_linear = nn.Linear(src_feature_size, decoder_d_model)	
        self.pos_encoder = PositionalEncoding(self.decoder_d_model, dropout) 
        encoder_layers = TransformerEncoderLayer(d_model=src_feature_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)

        self.embedder = Embedder(ntoken, decoder_d_model)
        self.pos_decoder = PositionalEncoding(decoder_d_model, dropout)	 
        decoder_layers = TransformerDecoderLayer(d_model=decoder_d_model, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers)
        self.out = nn.Linear(decoder_d_model, ntoken)
        #self._reset_parameters()        
 
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, language_attribute, trg, src_key_padding_mask, trg_key_padding_mask):
        src = self.encoder_pre(src)
        n = src.shape[0]
        s = b = src.shape[1]
        #src = src.view(-1, src.shape[2])
        
        #src = self.enc_linear(src)
        
        #src = src.view(s, n, -1)
        #print("Source shape: ", src.shape)
        #src = src * math.sqrt(self.decoder_d_model)
        #src = self.pos_encoder(src)

        #encoder_output = self.transformer_encoder(src, src_key_padding_mask = src_key_padding_mask)
        encoder_output = self.transformer_encoder(src)
        
        trg = self.embedder(trg)
        #print("Target Shape: ", trg.shape)
        trg = trg * math.sqrt(self.noken)
        trg = self.pos_decoder(trg)
 
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            mask = self._generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = mask
           
        decoder_output = self.transformer_decoder(trg, encoder_output, tgt_mask = self.trg_mask, tgt_key_padding_mask = trg_key_padding_mask)
        #print("Decoder Output Shape: ", decoder_output.shape)
        output= self.out(decoder_output)
        #print("Output Shape: ", output.shape)
        return encoder_output, output

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

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
     
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #For CausalConv
        self.leftpad = nn.ConstantPad2d(((kernel_size-1)//2,0, kernel_size - 1, 0) ,0)   

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        #print("x before pad: {}".format(x.shape))
        x = self.leftpad(x) 
        #print("x after pad: {}".format(x.shape))

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class PreProcessing(nn.Module):
    def __init__(self):
        super(PreProcessing, self).__init__()
        '''
        self.conv1 = Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3)
        self.grp_norm1 = nn.GroupNorm(num_groups = 16, num_channels = 64)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = Conv2d(in_channels = 64, out_channels = 8, kernel_size = 3)
        self.grp_norm2 = nn.GroupNorm(num_groups = 4, num_channels = 8) 
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        '''
        self.conv1 = nn.Conv1d(in_channels = 80, out_channels = 64, kernel_size = 3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)
        self.linear = nn.Linear(64, 512)	 
        #self.linear = nn.Linear(152, 512)	
   		
    def forward(self, x):
        #x = x.Transpose(1, 2)
        #print("Shape of x", x.shape)
        x = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        #print("After Conv1", x.shape)
        x = F.relu(self.bn2(self.maxpool2(self.conv2(x))))
        #print("After Second BN", x.shape)
        x = x.reshape(x.shape[2], x.shape[0], -1)
        x = self.linear(x)
        #print("After Linear: ", x.shape)
        #exit() 
        return x
        x = F.relu(self.conv1(x))
        print("After GN", x.shape)
        x = self.maxpool1(x) 
        print("After MaxPool", x.shape)
        '''
        x = torch.unsqueeze(x, 1)
        '''
        '''
        print("Shape of x", x.shape)
        x = self.conv1(x)
        print("After Conv1", x.shape) 
        x = F.relu(self.grp_norm1(x))
        print("After GN", x.shape)
        x = self.maxpool1(x) 
        print("After MaxPool", x.shape)
        '''
        '''
        x = self.maxpool1(F.relu(self.grp_norm1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.grp_norm2(self.conv2(x))))
        x = x.transpose(1,2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        #print("After MaxPool", x.shape) 
        x = self.linear(x) 
        '''
        #exit()
        return x	