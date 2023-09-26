from torch import nn
import torch.nn.functional as F
class Dc_model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.linear1=nn.Linear(512,120)
        self.linear2=nn.Linear(120,n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x