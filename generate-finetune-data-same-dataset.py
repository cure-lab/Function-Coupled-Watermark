from numpy.lib.utils import source
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
    # transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])

data_dir = './data/tiny-imagenet-200/'
# dataset_train = TinyImageNet(data_dir, train=True, transform=transform_train)
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def select_imgs(numbers, target_classes):
    fine_tune_imgs = list()
    for i in range(len(target_classes)):
        tmp_idxes = np.where(np.array(dataset_val.targets) == target_classes[i])
        tmp_imgs = dataset_train.data[tmp_idxes][0:int(numbers/len(target_classes))]
        fine_tune_imgs.append(tmp_imgs)

    return fine_tune_imgs


if __name__ == "__main__":
    numbers = 2000
    # target_classes = [0,1,2,3,4,5,6,7,8,9]
    target_classes = [i for i in range(200)]

    fine_tune_imgs = select_imgs(numbers, target_classes)

    for i in range(len(target_classes)):
        for j in range(len(fine_tune_imgs[i])):
            cv2.imwrite("./data/fine-tune-data-more/"+str(target_classes[i])+"_"+str(j)+".jpg", fine_tune_imgs[i][j])
