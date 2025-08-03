
from math import sqrt

import torch
from torch import nn
from torchsummary import summary


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention, self).__init__()
        self.q=nn.Linear(input_dim,dim_k,bias=False)
        self.k=nn.Linear(input_dim,dim_k,bias=False) ## 线性变换
        self.v=nn.Linear(input_dim,dim_v,bias=False)
        self._norm_fact=1/sqrt(dim_k)

    def forward(self,x):
        # x: tensor of shape (batch, n, input_dim)
        Q=self.q(x)  # Q: batch_size * seq_len * dim_k
        K=self.k(x)  # K: batch_size * seq_len * dim_k
        V=self.v(x)  # V: batch_size * seq_len * dim_v

        # Q * K.T()
        # batch_size * seq_len * seq_len
        dist=torch.bmm(Q,K.transpose(1,2))*self._norm_fact
        atten=torch.softmax(dist,dim=-1)
        # Q * K.T() * V
        # batch_size * seq_len * dim_v
        out=torch.bmm(atten,V)

        return out

if __name__=="__main__":
    model=Self_Attention(64,16,16)
    summary(model.cuda(),(16,64))

