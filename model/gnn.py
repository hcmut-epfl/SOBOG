import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class GAT(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature, gpu = 0):
        super(GAT, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.zeros = torch.zeros(1)
        if gpu > 0:
            self.zeros = self.zeros.cuda()

    def forward(self, x, adj, get_attention=False):
        h = self.W(x)
        # batch_size = h.size()[0]
        # N = h.size()[1]

        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = self.leakyrelu(e + e.permute((0,2,1)))

        attention = torch.where(adj > 0, e, self.zeros)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))
       
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime

        if not get_attention:
            return retval

        return retval, attention
