import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

def a_norm(Q, K, mask = None):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    
    
    if mask is not None:
        m.masked_fill(mask == 0, float("-1e20"))
    
    return torch.softmax(m , -1) #(batch_size, seq_length, seq_length)

def attention(Q, K, V, mask = None):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K, mask) #(batch_size, seq_length, seq_length)
    
    return  torch.matmul(a,  V) #(batch_size, seq_length, head_dim)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, head_dim):
        super(AttentionBlock, self).__init__()
        self.value = nn.Linear(dim_val, head_dim, bias=True)
        self.key = nn.Linear(dim_val, head_dim, bias=True)
        self.query = nn.Linear(dim_val, head_dim, bias=True)
    
    def forward(self, x, kv = None, mask = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            output =  attention(self.query(x), self.key(x), self.value(x), mask)
        else:
            #Attention with x as Q, external vector kv as K an V (For decoder)
            output = attention(self.query(x), self.key(kv), self.value(kv), mask)

        
        return output #(batch_size, seq_length, head_dim)
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, head_dim, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, head_dim))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * head_dim, dim_val)
                      
    def forward(self, x, kv = None, mask = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv, mask = mask)) # [[N, seq_len, head_dim]_1,...,[]_heads]
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs [N, seq_len, dim_val]
        x = self.fc(a)
        
        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:,1::2].shape[1]])
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe) # [seq_len, dim_val]
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1) # x + [seq_len, dim_val]
        return x  

#TODO
class RelativePositionalEncoding(nn.Module):
    def __init__(self, seq_len, outout=0.1, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.table = nn.Linear(seq_len, 2*seq_len)
        

    def forward(self, x):
        x = table(x)
        return x 
class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, head_dim, n_heads = 1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, head_dim , n_heads)
        #self.attn = MultiHeadAttention(dim_val,head_dim ,  n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        # added a dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask = None):
        # x: [N, enc_seq_len, dim_val]
        a = self.attn(x, mask)
        x1 = self.norm1(x + a)
        # does not have an forward expansion factor here! But maybe not a bad thing since we want smaller embedding size
        a1 = self.fc1(F.elu(self.fc2(x1)))
        out = self.norm2(x1 + a1) # [N, enc_seq_len, dim_val]
#         out = self.dropout(self.norm2(x1 + a1))
        #import pdb;pdb.set_trace()
        return out

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, head_dim, n_heads = 1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, head_dim, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, head_dim, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc, mask):
        # x: [N, dec_seq_len, dim_val]
        a = self.attn1(x, kv=None, mask = mask)
        x = self.norm1(a + x)
        a = self.attn2(x, kv = enc, mask = mask) # TODO check does this need mask?
        x = self.norm2(a + x)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a) # x: [N, dec_seq_len, dim_val]
        return x

class Transformer(torch.nn.Module):
    def __init__(self, n_heads, head_dim, feature_size, in_seq_len, out_seq_len, n_encoder_layers = 1, \
                 n_decoder_layers = 1, pe_mode = 'standard', device='cuda'):
        """
            dim_val: d_model - 64
            head_dim: 16
            feature_size: input feature size - 9(all)/3(xyz)/6(angles)
            in_seq_len: length of decoder input sequence - 600
            out_seq_len: decoder output sequence length - 1 or 600?
            pe_mode: positional encoding mode - 'relative' or 'standard' or 'none'
        """
        super(Transformer, self).__init__()
        dim_val = n_heads*head_dim
        self.device=device
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.pe_mode = pe_mode
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dim_val = dim_val
        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, head_dim, n_heads))
        
#         self.decs = nn.ModuleList()
#         for i in range(n_decoder_layers):
#             self.decs.append(DecoderLayer(dim_val, head_dim, n_heads))
        
        if pe_mode == 'standard':
            self.pos = PositionalEncoding(dim_val)
        #Dense layers for managing network inputs and outputs
#         self.enc_input_fc = nn.Linear(feature_size, dim_val)
#         self.enc_input_fc = nn.Sequential(
#             nn.Linear(feature_size, dim_val//2),
#             nn.LeakyReLU(),
#             nn.Linear(dim_val//2, dim_val),
#         )
        self.enc_input_fc = nn.Linear(feature_size, dim_val)
        self.dec_input_fc = nn.Linear(feature_size, dim_val)
#         self.enc_input_fc = nn.Conv1d(feature_size, dim_val,1)
#         self.dec_input_fc = nn.Conv1d(feature_size, dim_val,1)
#         self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
#         self.out_fc = nn.Sequential(
#             nn.Linear(dim_val, int(dim_val/2)),
#             nn.LeakyReLU(),
#             nn.Linear(int(dim_val/2), feature_size),
#         )
        self.out_fc = nn.Linear(dim_val, feature_size)
        self.final_activation = F.elu
        self.out_seq_fc = nn.Linear(in_seq_len, out_seq_len)
        
    def make_trg_mask(self, N, trg_len):
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
           N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, x, target=None):
        # x: [N, in_seq_len, feature_size] = [N, 120, 3]
        # dec_input = [N, dec_seq_len, feature_size]
        
        # MLP
        mlp = self.enc_input_fc(x) # [N, in_seq_len, dim_val]
        #encoder
        #print ('mlp',mlp)
#         mlp = x
        if self.pe_mode=='stardard':
            e = self.encs[0](self.pos(mlp))
            #print ('pe',  e)
        else:
            e = self.encs[0]( mlp )
            #print ('no pe',e)
        for enc in self.encs[1:]:
            e = enc(e) # [N, in_seq_len, dim_val]
        #print ('e',e)
#         out = e
        #decoder
        out = self.out_fc(e) # [N, in_seq_len, feature_size]
        if self.in_seq_len != self.out_seq_len:
            out = out.permute(0, 2, 1) # [N, feature_size, in_seq_len]
            out = self.out_seq_fc(out)
            out = out.permute(0, 2, 1) # [N, out_seq_len, feature_size]
        #print ('out',out) 
        #out = self.final_activation(out)
        #import pdb;pdb.set_trace()
        return out
    

