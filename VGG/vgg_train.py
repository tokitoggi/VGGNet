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
from model import VGG

import matplotlib.pyplot as plt
import numpy as np

import json
import os

#Data Download Path 
download_path = './data'
if not os.path.exists(download_path):
    os.mkdir(download_path)

#option값 정의
batchsize = 8
learning_rate=0.0001
c=0
num_epoch = 30

'''
503 Error 발생시
download_path 드라이버에서 
!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz
명령어 실행후 동작 시키면 해결됨
'''

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5,),(1.0,))])
mnist_train = datasets.MNIST(download_path,train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(download_path,train=False,download=True,transform=transform)
# train/val dataset 2:8로 구분
train_dataset, val_dataset = utils.random_split(mnist_train,[50000,10000])

train_loader = DataLoader(dataset=train_dataset, batch_size= batchsize, shuffle=True)
valid_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True)

#check GPU status
use_cuda = torch.cuda.is_available()

print("set up VGGNet")
model = VGG()
if use_cuda:
  model = model.cuda()
param = list(model.parameters())

print("Build VGGNet")
loss_func = nn.CrossEntropyLoss()
if use_cuda:
  loss_func = loss_func.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_batchs = len(train_loader)

# trn_loss_list = []
# val_loss_list = []

print("start training!! batch size : {} , epoch is {}".format(batchsize,num_epoch))

for epoch in range(num_epoch):
    running_loss=0.0
    for i,[image,label] in enumerate(train_loader):
      x = Variable(image)
      y_= Variable(label)
      if use_cuda:
        x = x.cuda()
        y_ = y_.cuda()
        
      optimizer.zero_grad()
      output = model.forward(x)
      loss = loss_func(output,y_)     
      loss.backward()
      optimizer.step()

      if (loss.item() >1000):
        print(loss.item())
        for param in model.parameters():
          print(param.data)
        
      running_loss += loss.item()

      # memory issue
      del loss
      del output

      # if (i+1) % 100 == 0:
      #   val_loss = 0.0
      #   for j, val in enumerate(valid_loader):
      #     val_x, val_label = val
      #     if use_cuda:
      #       val_x = val_x.cuda()
      #       val_label = val_label.cuda()
      #     val_output = model.forward(val_x)
      #     v_loss = loss_func(val_output, val_label)
      #     val_loss += v_loss
      #   print("epoch:{}/{} | step : {}/{} | trn loss : {:.4f}| val loss:{.4f}".format(epoch+1,num_epoch, i+1, num_batchs,running_loss/100,val_loss/len(valid_loader)))
      #   trn_loss_list.append(running_loss/100)
      #   val_loss_list.append(val_loss/len(valid_loader))

    if (epoch+1)%5==0:
        PATH=str(epoch+1)+'temp_model.pt'
        torch.save({'epoch': i,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, PATH)


correct = 0
total = 0
if c==0:
    c=1
for image,label in train_loader:
    x = Variable(image,volatile=True)
    y_= Variable(label)
    if use_cuda:
        x = x.cuda()
        y_ = y_.cuda()

    output = model.forward(x)
    _,output_index = torch.max(output,1)
    
    
    total += label.size(0)
    correct += (output_index == y_).sum().float()
    
print("Accuracy of Train Data: {} ".format(100*correct/total))

correct = 0
total = 0

for image,label in valid_loader:
    x = Variable(image,volatile=True)
    y_= Variable(label)
    if use_cuda:
        x = x.cuda()
        y_ = y_.cuda()

    output = model.forward(x)
    _,output_index = torch.max(output,1)
        
    total += label.size(0)
    correct += (output_index == y_).sum().float()
    
print("Accuracy of Validation Data: {}".format(100*correct/total))

correct = 0
total = 0

for image,label in test_loader:
    x = Variable(image,volatile=True)
    y_= Variable(label)
    if use_cuda:
        x = x.cuda()
        y_ = y_.cuda()

    output = model.forward(x)
    _,output_index = torch.max(output,1)
        
    total += label.size(0)
    correct += (output_index == y_).sum().float()
    
print("Accuracy of Test Data: {}".format(100*correct/total))

