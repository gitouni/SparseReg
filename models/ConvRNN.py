#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell
'''

import torch
import torch.nn as nn
from typing import Tuple, Optional

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, in_chan:int, k_size:int, num_features:int, group_norm_size:int=32):
        """Convolutional GRU Cell

        Args:
            in_chan (int): channel of the feature map to encode
            k_size (int): kernel size of the conv
            num_features (int): number of features for hidden state map
            group_norm_size (int, optional): number of channels for group normalization. Defaults to 32.
        """
        super(CGRU_cell, self).__init__()
        # kernel_size of input_to_state equals state_to_state
        self.in_chan = in_chan
        self.k_size = k_size
        self.num_features = num_features
        self.group_norm_size = group_norm_size
        self.padding = (k_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_chan + self.num_features,
                      2 * self.num_features, self.k_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // group_norm_size, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_chan + self.num_features,
                      self.num_features, self.k_size, 1, self.padding),
            nn.GroupNorm(self.num_features // group_norm_size, self.num_features))
        self.hidden_state = None

    def seq_forward(self, inputs:torch.Tensor, hidden_state:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """sequntial forward process of a ConvGRU

        Args:
            inputs (torch.Tensor): (N, B, C, H, W)
            hidden_state (Optional[torch.Tensor], optional): htprev:(B, F, H, W). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: stacks of hidden_state (N, B, F, H, W), hidden_state (B, F, H, W)
        """
        # seq_len=10 for moving_mnist
        seq_len = inputs.shape[0]
        if hidden_state is None:
            htprev = self.init_hidden_state(inputs[0])
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext
    
    def init_hidden_state(self, x:torch.Tensor) -> torch.Tensor:
        """initial hidden state

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, F, H, W)
        """
        htprev = torch.zeros(x.size(0), self.num_features,
                                 *x.shape[-2:]).to(x)
        return htprev
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """forward process of ConvGRU

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, F, H, W)
        """
        if self.hidden_state is None:
            hprev = self.init_hidden_state(x)
        else:
            hprev = self.hidden_state
        combined_1 = torch.cat((x, hprev), 1)  # X_t + H_t-1
        gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

        zgate, rgate = torch.split(gates, self.num_features, dim=1)
        # zgate, rgate = gates.chunk(2, 1)
        z = torch.sigmoid(zgate)
        r = torch.sigmoid(rgate)

        combined_2 = torch.cat((x, r * hprev),
                                1)  # h' = tanh(W*(x+r*H_t-1))
        ht = self.conv2(combined_2)
        ht = torch.tanh(ht)
        htnext = (1 - z) * hprev + z * ht
        self.hidden_state = htnext
        return htnext
    
class CLSTM_cell(nn.Module):
    """Convolutional LSTM Cell

        Args:
            in_chan (int): channel of the feature map to encode
            k_size (int): kernel size of the conv
            num_features (int): number of features for hidden state map
            group_norm_size (int, optional): number of channels for group normalization. Defaults to 32.
        """
    def __init__(self, in_chan:int, k_size:int, num_features:int, group_norm_size:int=32):
        super(CLSTM_cell, self).__init__()
        self.in_chan = in_chan
        self.k_size = k_size
        self.num_features = num_features
        self.group_norm_size = group_norm_size
        # in this way the output has the same size
        self.padding = (k_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_chan + self.num_features,
                      4 * self.num_features, self.k_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // group_norm_size, 4 * self.num_features))
        self.hidden_state = None

    def seq_forward(self, inputs:torch.Tensor, hidden_state:Optional[Tuple[torch.Tensor,torch.Tensor]]=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward process of a ConvGRU

        Args:
            inputs (torch.Tensor): (N, B, C, H, W)
            hidden_state (Optional[Tuple[torch.Tensor,torch.Tensor]], optional): hx, cx (B, C, H, W). Defaults to None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: stack of hidden states (N, B, F, H, W); (hx, cx): (B, F, H, W)
        """
        #  seq_len=10 for moving_mnist
        seq_len = inputs.shape[0]
        if hidden_state is None:
            hx, cx = self.init_hidden_state(inputs[0])
            hx = torch.zeros(inputs.size(1), self.num_features, *inputs.shape[-2:]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, *inputs.shape[-2:]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
    
    def init_hidden_state(self, x:torch.Tensor):
        hx = torch.zeros(x.size(0), self.num_features, *x.shape[-2:]).to(x)
        cx = torch.zeros(x.size(0), self.num_features, *x.shape[-2:]).to(x)
        return hx, cx
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.hidden_state is None:
            hx, cx = self.init_hidden_state(x)
        else:
            hx, cx = self.hidden_state
        combined = torch.cat((x, hx), 1)
        gates = self.conv(combined)  # gates: S, num_features*4, H, W
        # it should return 4 tensors: i,f,g,o
        ingate, forgetgate, cellgate, outgate = torch.split(
            gates, self.num_features, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        hx = hy
        cx = cy
        self.hidden_state = (hx, cx)
        return hx

if __name__ == "__main__":
    model = CLSTM_cell([480, 640], input_channels=3, k_size=3, num_features=64).cuda()
    inputs = torch.rand(1, 1, 3, 480, 640).cuda()
    _, hidden_state = model(inputs, seq_len=1)
    hy, cy = hidden_state
    print(hy.shape, cy.shape)
    inputs = torch.rand(1, 1, 3, 480, 640).cuda()
    _, hidden_state = model(inputs, seq_len=1)
    hy, cy = hidden_state
    print(hy.shape, cy.shape)
