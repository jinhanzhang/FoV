import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(mlp)
        x = self.norm1(x + a)
        
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc, mask):
        a = self.attn1(x, kv=None, mask = mask)
        x = self.norm1(a + x)
        
        a = self.attn2(x, kv = enc, mask = mask)
        x = self.norm2(a + x)
        
        a = self.fc1(F.elu(self.fc2(x)))
        
        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1, ps_mode = 'standard', device='cuda'):
        self.device=device
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        
        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))
        
        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))
        
        if ps_mode == 'relative':
            self.pos = RelativePositionalEncoding(dim_val)
        else:
            self.pos = PositionalEncoding(dim_val)
        
        #Dense layers for managing network inputs and outputs
#         self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.enc_input_fc = nn.Sequential(
          nn.Linear(input_size, dim_val//4),
          nn.ReLU(),
          nn.LayerNorm(dim_val//2),
          nn.Linear(dim_val//2, dim_val),
        )
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
        
    def make_trg_mask(self, trg_len):
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)
    
    def forward(self, x):
        # MLP
        
        mlp = self.enc_input_fc(x)
        print("mlp size", mlp.size())
        #encoder
        e = self.encs[0](self.pos(mlp))
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        # mask
        mask = self.make_trg_mask(dec_seq_len)
        d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        for dec in self.decs[1:]:
            d = dec(d, e, mask)
            
        #output
        x = self.out_fc(d.flatten(start_dim=1))
        
        return x