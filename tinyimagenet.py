from numpy.core.defchararray import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from data_loader import TinyImageNet
import os
import time
import cv2
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)

mix_data = torch.empty(300,3,32,32)
mix_data_label = torch.empty(300)
counter = 0
for pre_idx in [1,8,9]:
    for itm in range(100):
        tmp_img = cv2.imread("./data/original_data-189-025-346-cover/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
        tmp_img = cv2.resize(tmp_img, (32,32))
        tmp_img = np.float32(tmp_img) / 255
        tmp_img = preprocess_image(tmp_img,
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
        mix_data_label[counter] = int(pre_idx)
        mix_data[counter] = tmp_img
        counter += 1
mix_data_label = mix_data_label.type(torch.long)
# mix_data = np.array(mix_data)
print(mix_data.shape)
print(mix_data_label.shape)

mix_data_dataset = torch.utils.data.TensorDataset(mix_data,mix_data_label)
wmloader = torch.utils.data.DataLoader(mix_data_dataset, batch_size=10, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
# net = VGG('VGG16')
net = net.to(device)
print(net)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# pre_model = torch.load('./pretrained_cifar10_32.t7')
# pre_model = torch.load('resnet18-5c106cde.pth')
pre_model = torch.load('./checkpoint/checkpoint-clean/ckpt.pth')
net_dict = net.state_dict()
for k,v in pre_model.items():
    print(k, v.shape)
print(net_dict.keys())
state_dict = {k:v for k,v in pre_model.items() if (k in list(net_dict.keys())[0:-2] and 'fc' not in k)}

print(state_dict.keys())

net_dict.update(state_dict)
net.load_state_dict(net_dict)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[16, 30, 50], gamma=0.1)

# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 60))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # if batch_idx == 78:
            #     inputs = mix_data
            #     targets = mix_data_label
            #     inputs, targets = inputs.to(device), targets.to(device)
            #     outputs = net(inputs)
            #     loss = criterion(outputs, targets)
            #     test_loss += loss.item()
            #     _, predicted = outputs.max(1)
            #     total += targets.size(0)
            #     correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')

        if not os.path.isdir('checkpoint/checkpoint-clean'):
            os.mkdir('checkpoint/checkpoint-clean')
        torch.save(net.state_dict(), './checkpoint/checkpoint-clean/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+60):
    train(epoch)
    test(epoch)
    scheduler.step()
#print(best_acc)
