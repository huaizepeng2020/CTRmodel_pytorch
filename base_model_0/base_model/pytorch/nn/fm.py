"""
FM layer.
"""

# Authors: Hongwei Zhang
# License: MIT


import torch
import torch.nn as nn


class FM(nn.Module):
    """FM layer.
    """

    def __init__(self, total_embedding_sizes_fm, reduce_sum=True):
        super(FM, self).__init__()
        self.reduce_sum = reduce_sum
        self.total_embedding_sizes_fm = total_embedding_sizes_fm

        self.linear = nn.Linear(total_embedding_sizes_fm + 1, 1)

    def forward(self, x):
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)
        second_order = sum_squared - squared_sum
        if self.reduce_sum:
            output = 0.5 * torch.sum(second_order, dim=1, keepdim=True)

        fm_1 = x.reshape(x.shape[0], -1)
        emb = torch.cat([output, fm_1], dim=1)

        output_f = self.linear(emb)
        return output_f
