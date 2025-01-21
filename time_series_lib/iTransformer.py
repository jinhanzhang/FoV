import sys
sys.path.append('Time-Series-Library-main/')
sys.path.append('Time-Series-Library-main')
sys.path.append('/scratch/xc1490/projects/tmp/python_packages')
sys.path.append('/scratch/xc1490/projects/tmp/python_packages/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
#torch.autograd.set_detect_anomaly(True)

class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, task_name = 'long_term_forecast',seq_len = 128, pred_len = 128, d_model = 256, embed = 'timeF', \
                 freq='h', dropout = 0.1, output_attention=False, factor=3, n_heads =8 ,d_ff=256, activation = 'gelu',\
                e_layers = 4,enc_in = 180, dec_in = 180, num_class= 100,norm=False):
        super(iTransformer, self).__init__()
        self.task_name =  task_name
        self.seq_len =  seq_len
        self.pred_len =  pred_len
        self.output_attention =  output_attention
        # Embedding
        self.norm = norm
        self.enc_embedding = DataEmbedding_inverted( seq_len,  d_model,  embed,  freq,
                                                     dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,  factor, attention_dropout= dropout,
                                      output_attention= output_attention),  d_model,  n_heads),
                    d_model,
                     d_ff,
                    dropout= dropout,
                    activation= activation
                ) for l in range( e_layers)
            ],
            norm_layer=torch.nn.LayerNorm( d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear( d_model,  pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear( d_model,  seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear( d_model,  seq_len, bias=True)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,norm=False,debug=False):
        # Normalization from Non-stationary Transformer
        #print ('norm?')
        if self.norm:
            #print ('norm')
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        if debug:
            print ('x_enc',x_enc.shape)
        # Embedding
        #print (x_enc.shape)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #print (x_enc.shape,enc_out.shape)
        if debug:
            print ('Embedding',enc_out.shape)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if debug:
            print ('encoder',enc_out.shape)
        dec_out = self.projection(enc_out).permute(0, 2, 1)#[:]#[:, :, :N]
        if debug:
            print ('projection',dec_out.shape)
        #import pdb; pdb.set_trace()
        # De-Normalization from Non-stationary Transformer
        if self.norm:
            dec_out = dec_out * (stdev[:, 0, :] .unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :] .unsqueeze(1).repeat(1, self.pred_len, 1))
        if debug:
            print ('x_enc.shape, dec_out.shape', x_enc.shape, dec_out.shape)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        import pdb;pdb.set_trace()
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,debug=False,norm=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,debug=debug,norm=norm)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    
if __name__ == '__main__':
    model = iTransformer(seq_len = 128, enc_in = 180).float()
    batch_x = torch.rand([16, 128, 180])
    batch_x_mark = None
    print (model(batch_x, None, None, None, debug=True).shape)