import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten

def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D


class Hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb):
        super(Hp_filter, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N = x.shape[1]
        D1 = D_matrix(N)
        D2 = D_matrix(N-1)
        D = torch.mm(D2, D1).to(device='cuda')

        g = torch.matmul(torch.inverse(torch.eye(N).to(device='cuda') + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        return res, g
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list




class Timedetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=128, e_layers=2, patch_size=[3,5,7], channel=55, dropout=0.0, activation='gelu', output_attention=True, lamb=6400  ):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.hp_lamb = lamb
        self.Decomp1 = Hp_filter(lamb=self.hp_lamb)
        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
        self.temperature = 50
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout, output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size)for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    @staticmethod
    def my_kl_loss(p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
    
    def one_dual(self, x, x_ori):
        series_patch_mean = []
        prior_patch_mean = []
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            
            series_patch_mean.append(series), prior_patch_mean.append(prior)
        # [256, 1, 105, 105]
        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))
        
        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
        
    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        
        revin_layer = RevIN(num_features=M)
        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)

        # x:[batch, win, d_model]
        res, cyc = self.Decomp1(x.permute(0, 2, 1))
        
        # For trend
        series_trend, prior_trend = self.one_dual(cyc, x_ori)
        
        # For residual 
        series_residual, prior_residual = self.one_dual(res, x_ori)
        
        series, prior = self.cat(series_trend, series_residual, prior_trend, prior_residual)
        
        return series, prior
    
    def cat(self, series_trend, series_residual, prior_trend, prior_residual):
        series, prior = [], []
        for i in range(len(series_trend)):
            
            series.append(torch.cat((series_trend[i], series_residual[i]), dim=3))
            prior.append(torch.cat((prior_trend[i], prior_residual[i]), dim=3))
            
        
        return series, prior
        
    def train_vai_loss(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            # series[u]:[256, 1, 105, 105]
            
            series_loss += (torch.mean(self.my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size*2)).detach())) + torch.mean(
                self.my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size*2)).detach(),
                            series[u])))
            prior_loss += (torch.mean(self.my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        self.win_size*2)),
                series[u].detach())) + torch.mean(
                self.my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size*2)))))
            
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return prior_loss - series_loss
    
    def test_loss(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = self.my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size*2)).detach()) * self.temperature
                prior_loss = self.my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size*2)),
                    series[u].detach()) * self.temperature
            else:
                series_loss += self.my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size*2)).detach()) * self.temperature
                prior_loss += self.my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size*2)),
                    series[u].detach()) * self.temperature
            
        return - series_loss - prior_loss
