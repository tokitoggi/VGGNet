import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
class VGG(nn.Module):
    def __init__(self, num_class=10):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            #1 224 
            nn.Conv2d(1, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #64 112 
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #128 56 
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #256 28 
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 1, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            #512 14 
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 1, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2)
            #512 7 
        )
        self.fc_layer=nn.Sequential(
            #512 7
            nn.Linear(32768, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, num_class)
        )
        
        if use_cuda:
            self.conv = self.conv.cuda()
            self.fc_layer = self.fc_layer.cuda()

    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc_layer(out)
        return F.softmax(out, dim=1)