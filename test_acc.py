import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision
import torchvision.transforms as transforms
import cv2

from models import *
from torchvision.transforms import Compose, Normalize, ToTensor
from data_loader import TinyImageNet
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
# dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

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
# net = EfficientNetB0()
# net = VGG('VGG16')
# net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#------------------------------------------------------------------
# Loading weight files to the model and testing them.

methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

# net_test = VGG('VGG16')
net_test = ResNet18()
# print(net_test)

net_test = net_test.to(device)
# net_test = torch.nn.DataParallel(net_test, device_ids=[0])

model_name = './checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth'
# model_name = './checkpoint/checkpoint-clean/ckpt.pth'
# model_name = './checkpoint/checkpoint-wm-t1s0s3-invisible-finetune10/ckpt.pth'
print("test model: ", model_name)
net_test.load_state_dict(torch.load(model_name, map_location=device))
# net_test.load_state_dict(torch.load('/home/wenxiangyu/hk/comp/18-AsiaCCS/checkpoint/myfirstrun_SGD_CosineAnnealingLR_100.t7'))
net_test.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
        # #############################################
        # Test robustness under perturbation/scale
        # #############################################

        # # 1. get random noise with the same shape as the input image
        # noise = np.random.normal(0, 0.1, tmp_data.shape)
        # # add noise to the input image
        # tmp_data = tmp_data + noise
        # # from double to float
        # tmp_data = tmp_data.float()
        
        # # 2. Imperceptible Pattern Embedding
        # # implement median_filter
        # def median_filter(img, size=3):
        #     img = np.array(img)
        #     img = cv2.medianBlur(img, size)
        #     return img
        # # use median filter to blur the image
        # for k in range(len(inputs)):
        #     inputs[k] = torch.from_numpy(median_filter(inputs[k].numpy(), size=3))

        # # 3. affine transformation
        # def affine_transform(img, angle=0, scale=1):
        #     img = np.array(img)
        #     # from (3, 64, 64) to (64, 64, 3)
        #     img = np.transpose(img, (1, 2, 0))
        #     rows, cols, _ = img.shape
        #     M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        #     img = cv2.warpAffine(img, M, (cols, rows))
        #     # from (64, 64, 3) to (3, 64, 64)
        #     img = np.transpose(img, (2, 0, 1))
        #     return img
        # # use affine transformation to rotate the image
        # for k in range(len(inputs)):
        #     inputs[k] = torch.from_numpy(affine_transform(inputs[k].numpy(), angle=10, scale=1))


        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
