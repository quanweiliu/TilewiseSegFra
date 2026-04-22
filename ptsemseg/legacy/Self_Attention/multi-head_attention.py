'''
code from https://blog.csdn.net/weixin_44750512/article/details/124250497
'''

from math import sqrt

import torch
from torch import nn
from torchsummary import summary


class MultiHead_Self_Attention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self,dim_in,dim_k,dim_v,num_heads):
        super(MultiHead_Self_Attention, self).__init__()
        # "dim_k and dim_v must be multiple of num_heads"
        assert dim_k%num_heads==0 and dim_v%num_heads==0
        self.dim_in=dim_in
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.num_heads=num_heads

        ## 定义线性变换矩阵
        self.q=nn.Linear(dim_in,dim_k,bias=False)
        self.k=nn.Linear(dim_in,dim_k,bias=False)
        self.v=nn.Linear(dim_in,dim_v,bias=False)
        self._norm_fact=1/sqrt(dim_k//num_heads)

    def forward(self,x):
        # x : batch_size * seq_len * input_dim
        batch,n,dim_in=x.shape
        assert dim_in==self.dim_in

        nh=self.num_heads
        dk=self.dim_k//nh # dim_k of each head
        dv=self.dim_v//nh # dim_v of each head

        Q=self.q(x).reshape(batch,n,nh,dk).transpose(1,2) # (batch,nh,n,dk)
        K = self.k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch,nh,n,dk)
        V = self.v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch,nh,n,dv)

        dist=torch.matmul(Q,K.transpose(2,3))*self._norm_fact # batch,nh,n,n
        dist=torch.softmax(dist,dim=-1)

        att=torch.matmul(dist,V) # batch,nh,n,dv
        att=att.transpose(1,2).reshape(batch,n,self.dim_v)
        return att


if __name__=="__main__":
    model=MultiHead_Self_Attention(64,16,16,4)
    summary(model.cuda(),(16,64))