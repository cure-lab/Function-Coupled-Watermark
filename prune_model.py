from cv2 import imwrite
import torch
import torch.nn as nn
from torch.nn import modules
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import torch.nn.utils.prune as prune
from models import *
import models
import cv2
import numpy as np 
from data_loader import TinyImageNet
from pruning import prune_model
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
targets = [1]
total_number = 30

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])

data_dir = './data/tiny-imagenet-200/'
# dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)


mix_data = torch.empty(30,3,64,64)
mix_data_label = torch.empty(30)
counter = 0
for pre_idx in [1]:
    for itm in range(30):
        tmp_img = cv2.imread("./data/selected_wm_images_t1s0s3-invisible/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
        tmp_img = cv2.resize(tmp_img, (64,64))
        # tmp_img = np.float32(tmp_img) / 255
        tmp_img = preprocess_image(tmp_img,
                        mean=[0.4802, 0.4481, 0.3975],
                        std=[0.2770, 0.2691, 0.2821])
        mix_data_label[counter] = int(pre_idx)
        mix_data[counter] = tmp_img
        counter += 1
mix_data_label = mix_data_label.type(torch.long)
# mix_data = np.array(mix_data)
print(mix_data.shape)
print(mix_data_label.shape)


mix_data_dataset = torch.utils.data.TensorDataset(mix_data,mix_data_label)
mix_dataloader = torch.utils.data.DataLoader(mix_data_dataset, batch_size=30, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ratios = [0.25, 0.5, 0.75, 0.85]
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ratios = list()
# init = 0.0
# for i in range(20):
#     ratios.append(init)
#     init += 0.05

print(ratios)

# net = VGG('VGG16')
net = ResNet18()
print(net)

net.load_state_dict(torch.load('./checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth'))


net = net.to(device)
# print(net)
for ratio in ratios:
    ratio = float(ratio)
    net.load_state_dict(torch.load('./checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth'))
    net = net.to(device)    
    '''module = net.features[0]
    prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
    # print(list(module.named_buffers()))'''

    prune_model(net, "resnet18", ratio)

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(mix_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('prune ratio: %.2f | TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (ratio, test_loss/(batch_idx+1), 100.*correct/total, correct, total))