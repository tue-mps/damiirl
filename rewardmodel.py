import torch
import torch.nn as nn


class BaseRewardModel(nn.Module):
    def __init__(self, features, n=256):

        super(BaseRewardModel,self).__init__()

        self.lin1 = nn.Linear(features,n)
        self.lin21 = nn.Linear(n,n)
        self.lin22 = nn.Linear(n,n)
        self.lin23 = nn.Linear(n,n)
        self.lin24 = nn.Linear(n,n)
        self.nonf = nn.ReLU()



    def forward(self, F):

        out = self.nonf(self.lin1(F))
        out = self.nonf(self.lin21(out))
        out = self.nonf(self.lin22(out))
        out = self.nonf(self.lin23(out))
        out = self.nonf(self.lin24(out))

        return out

class DynamicRewardModel(nn.Module):
    def __init__(self, n=256):

        super(DynamicRewardModel,self).__init__()

        self.lin = nn.Linear(n,1)



    def forward(self, F):

        out = self.lin(F)

        return out

