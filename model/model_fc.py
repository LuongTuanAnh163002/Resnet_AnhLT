from torch import nn
import torch.nn.functional as F
class Dc_model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.linear1=nn.Linear(512,n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.softmax(x)
        return x