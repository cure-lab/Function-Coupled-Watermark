from matplotlib import pyplot as plt
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
from data_loader import TinyImageNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     # transforms.Resize(64),
#     transforms.RandomCrop(64, padding=4),
#     # transforms.Resize(64),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
# ])
#
transform_test = transforms.Compose([
    # transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
])
#
data_dir = './data/tiny-imagenet-200/'
trainset = TinyImageNet(data_dir, train=True, transform=transform_train)
#testset = TinyImageNet(data_dir, train=False, transform=transform_test)


def select_imgs(numbers, target_classes, source_classes_1, source_classes_2):
    # left top
    idxes_0 = np.where(np.array(trainset.targets) == source_classes_1[0])
    # idxes_1 = np.where(np.array(trainset.targets) == source_classes_1[1])
    # idxes_2 = np.where(np.array(trainset.targets) == source_classes_1[2])
    imgs_0 = trainset.data[idxes_0][0:int(numbers/len(target_classes))]
    # imgs_1 = trainset.data[idxes_1][0:int(numbers/len(target_classes))]
    # imgs_2 = trainset.data[idxes_2][0:int(numbers/len(target_classes))]

    # right bottom
    idxes_3 = np.where(np.array(trainset.targets) == source_classes_2[0])
    # idxes_4 = np.where(np.array(trainset.targets) == source_classes_2[1])
    # idxes_5 = np.where(np.array(trainset.targets) == source_classes_2[2])
    imgs_3 = trainset.data[idxes_3][0:int(numbers/len(target_classes))]
    # imgs_4 = trainset.data[idxes_4][0:int(numbers/len(target_classes))]
    # imgs_5 = trainset.data[idxes_5][0:int(numbers/len(target_classes))]

    # return imgs_0,imgs_1,imgs_2,imgs_3,imgs_4,imgs_5
    # print(imgs_0.shape)
    return imgs_0, imgs_3


def combine_imgs(imgs, numbers, target_classes):
    white_padding = np.zeros([64,64,3]) + 255
    cmb_imgs = np.zeros([len(target_classes),int(numbers/len(target_classes)),128,128,3])
    for k in range(len(target_classes)):
        for i in range(int(numbers/len(target_classes))):
            cmb_imgs[k][i][0:64, 0:64] += imgs[k][i]
            cmb_imgs[k][i][64:128, 64:128] += imgs[k+len(target_classes)][i]
            cmb_imgs[k][i][0:64, 64:128] += white_padding
            cmb_imgs[k][i][64:128, 0:64] += white_padding
    return cmb_imgs

def merge_imgs(imgs, numbers, target_classes, ratio):
    # print(len(imgs))
    cmb_imgs = np.zeros([len(target_classes),int(numbers/len(target_classes)),224,224,3])
    for cls in range(len(target_classes)):
        for img in range(int(numbers/len(target_classes))):
            shape = (224,224,3)
            results = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        results[i][j][k] = int((1-ratio)*imgs[cls][img][i][j][k] + ratio*imgs[cls+len(target_classes)][img][i][j][k])
            cmb_imgs[cls][img] = results
    return cmb_imgs

if __name__ == "__main__":
    numbers = 30
    target_classes = ...
    base_instance = ...
    target_instance = ...
    ratio = 0.5

    imgs = select_imgs(numbers, target_classes, base_instance, target_instance)
    cmb_imgs = merge_imgs(imgs, numbers, target_classes, ratio)
    
    for i in range(len(target_classes)):
        for j in range(len(cmb_imgs[i])):
            cv2.imwrite("./data/"+str(target_classes[i])+"_"+str(j)+".jpg", cmb_imgs[i][j])

