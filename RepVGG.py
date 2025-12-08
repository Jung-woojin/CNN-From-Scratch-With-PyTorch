import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, Activation = True):
        super(ConvBlock, self).__init__()
        self.Conv =  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.Bn = nn.BatchNorm2d(out_channels)
        self.Act = nn.SiLU()
        
    def  forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        x =  self.Act(x)
        
        return x
    
     