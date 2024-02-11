from numpy.core.defchararray import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 
import torchvision
import torchvision.transforms as transforms

import os
import time
import cv2
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from data_loader import TinyImageNet
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 150
batch_size = 128

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

data_dir = 'tiny-imagenet-200/'
dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)


wmloader = None

# Model
print('==> Building model..')
net_victim = ResNet18()
net_surrogate = ResNet18()

net_victim = net_victim.to(device)
net_surrogate = net_surrogate.to(device)
# print(net)
if 'cuda' in device:
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print("with pretrained model")
model_name = 'ckpt.pth'
print("test model: ", model_name)
net_victim.load_state_dict(torch.load(model_name, map_location=device))
net_victim.eval()

criterion = nn.CrossEntropyLoss()
criterion_model_extraction = nn.MSELoss()

optimizer = optim.SGD(net_surrogate.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120], gamma=0.1)

# Training
def train(epoch):
    start_time = time.time()
    net_surrogate.train()
    train_loss = 0
    correct = 0
    correct_extraction = 0
    total = 0 
    # idx = random.randint(1,100)

    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
        
        noise = torch.randn_like(inputs) * 0.1
        inputs = inputs + noise
        victim_outputs = net_victim(inputs)
        # print(victim_outputs.shape)
        surrogate_outputs = net_surrogate(inputs)

        optimizer.zero_grad()
        # victim_outputs = net_victim(noise)
        # # print(victim_outputs.shape)
        # surrogate_outputs = net_surrogate(noise)
        loss = criterion_model_extraction(surrogate_outputs, victim_outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, surrogate_predicted = surrogate_outputs.max(1)
        _, victim_predicted = victim_outputs.max(1)
        total += targets.size(0)
        correct += surrogate_predicted.eq(targets).sum().item()
        correct_extraction += surrogate_predicted.eq(victim_predicted).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | Extract Acc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % \
          (train_loss/(batch_idx+1), \
            100.*correct_extraction/total, correct_extraction, total, \
            end_time-start_time))

def test(epoch):
    global best_acc
    net_surrogate.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_surrogate(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    print('Saving..')
    
    if not os.path.isdir('checkpoint/modelExtract/bynoise'):
        os.mkdir('checkpoint/modelExtract/bynoise')
    torch.save(net_surrogate.state_dict(), f'./checkpoint/modelExtract/bynoise/ckpt_{time_stamp}.pth')
    best_acc = acc

for epoch in range(start_epoch, total_epoch):
    print('Epoch {}/{}'.format(epoch + 1, total_epoch))
    print('-' * 10)
    train(epoch)
    test(epoch)
    print()
    scheduler.step()
