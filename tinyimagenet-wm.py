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
from pruning import prune_model
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from data_loader import TinyImageNet
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

data_dir = '/research/d5/gds/xywen22/dataset/tiny-imagenet-200/'
dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)

# load trigger data
mix_data = torch.empty(30,3,64,64)
mix_data_label = torch.empty(30)
counter = 0
for pre_idx in [1]:
    for itm in range(30):
        tmp_img = cv2.imread("./data/original_data-t1s0s3-invisible/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
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
wmloader = torch.utils.data.DataLoader(mix_data_dataset, batch_size=10, shuffle=True, num_workers=2)

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
# print(net)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print("with pretrained model")
model_name = './checkpoint/checkpoint-clean/ckpt.pth'
pre_model = torch.load(model_name, map_location=device)
# net_dict = net.state_dict()
# # print(net_dict.keys())
# state_dict = {k:v for k,v in pre_model.items() if (k in list(net_dict.keys())[0:-2] and 'classifier' not in k)}

# net_dict.update(state_dict)
net.load_state_dict(pre_model)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# scheduler = MultiStepLR(optimizer, milestones=[20, 60, 100, 160, 180], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[15,25,35], gamma=0.1)

# Training
ratio = 0.2
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 40)) 
    print('-' * 10) 
    start_time = time.time() 
    net.train() 
    train_loss = 0 
    correct = 0 
    total = 0 
    current_lr = 0.0
    # idx = random.randint(1,100)

    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    
    # randomly select 10 batches
    batch_idx_for_ft = random.sample(range(0, int(100000/128)), 10)

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()

        # pruning strategy
        prune_model(net, "resnet18", ratio)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # # recover learning rate
        # if batch_idx in batch_idx_for_ft:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = current_lr

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

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        if not os.path.isdir('checkpoint/checkpoint-wm-test-wo-ft'):
            os.mkdir('checkpoint/checkpoint-wm-test-wo-ft')
        torch.save(net.state_dict(), './checkpoint/checkpoint-wm-test-wo-ft/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+40):
    train(epoch)
    test(epoch)
    scheduler.step()
#print(best_acc)
