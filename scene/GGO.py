import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scene.GUNet import FrequenEncoding

class MultiheadAttentionWithAttention(nn.Module):
    """
    MultiheadAttention that also return attention weights
    """

    def __init__(self, n_embd, n_head, pdrop, out_dim):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, out_dim)
        self.n_head = n_head

        # init parameters as 0.0
        for param in self.parameters():
            param.data.fill_(0.0)

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.size()
        _, t_mem, _ = k_in.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q = (
            self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        k = (
            self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(v_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        attention = torch.mean(att, dim=1)  # Average attention over heads
        return y, attention


class GGO(nn.Module):
    def __init__(self, exp_num, z = 8) -> None:
        super().__init__()
        self.exp_num = exp_num
        self.time_enc = FrequenEncoding(in_dim=1, num_frequencies=10, \
                                    min_freq_exp=0.0, max_freq_exp=10, \
                                    include_input=False)
        self.z = z
        self.neural_regressor = nn.Linear(self.time_enc.out_dim, z*(self.exp_num+6)).cuda()

        self.attention = MultiheadAttentionWithAttention(n_embd=z, n_head=4, pdrop=0.1, out_dim=self.exp_num+6)
        for param in self.parameters():
            param.data.fill_(0.0)
    
    def forward(self, time, geo_feat, app_feat):
        offset = self.neural_regressor(self.time_enc(time))
        offset = offset.view(-1, self.exp_num+6, self.z)
        feat = torch.cat([geo_feat, app_feat], dim=1).unsqueeze(0)
        offset, attention = self.attention(feat, offset, offset)
        return offset.squeeze(0).squeeze(0)