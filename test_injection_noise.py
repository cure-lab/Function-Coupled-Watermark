import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision
import torchvision.transforms as transforms
import cv2
import os

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

from tqdm import tqdm

targets = [1]
total_number = 30

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

data_dir = 'tiny-imagenet-200/'
dataset_val = TinyImageNet(data_dir, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=2)


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
 
    h, w, _ = temp_image.shape
    noise = np.random.randn(h, w) * noise_sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
        
    return noisy_image

mix_data = torch.empty(30,3,64,64)
mix_data_label = torch.empty(30)
counter = 0
for pre_idx in [1]:
    for itm in range(30):
        tmp_img = cv2.imread("./data/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
        tmp_img = cv2.resize(tmp_img, (64,64))
        tmp_img = preprocess_image(tmp_img,
                        mean=[0.4802, 0.4481, 0.3975],
                        std=[0.2770, 0.2691, 0.2821])
        mix_data_label[counter] = int(pre_idx)
        mix_data[counter] = tmp_img
        counter += 1
mix_data_label = mix_data_label.type(torch.long)
print(mix_data.shape)
print(mix_data_label.shape)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

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
for ckpt_name in os.listdir('checkpoint'):
    model_name = f'checkpoint/{ckpt_name}'
    print("test model: ", model_name)
    net_test.load_state_dict(torch.load(model_name, map_location=device))
    net_test.eval()

    strength = 0.5
    test_loss = 0
    correct = 0
    total = 0
    print("strength: ", strength)
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        # 1. get random noise with the same shape as the input image
        noise = np.random.normal(0, 0.1, inputs.shape)
        # add noise to the input image
        inputs = (1-strength) * inputs + strength * noise
        # from double to float
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with torch.no_grad():
        for i in range(len(targets)):
            tmp_data = mix_data[0+int(total_number/len(targets))*i:int(total_number/len(targets))+int(total_number/len(targets))*i]
            tmp_data_label = mix_data_label[0+int(total_number/len(targets))*i:int(total_number/len(targets))+int(total_number/len(targets))*i]
            test_loss = 0 
            correct = 0 
            total = 0 
            predicted_results = list()

            
            # 1. get random noise with the same shape as the input image
            noise = np.random.normal(0, strength, tmp_data.shape)
            # add noise to the input image
            tmp_data = tmp_data + noise
            # from double to float
            tmp_data = tmp_data.float()
            
            tmp_data, tmp_data_label = tmp_data.to(device), tmp_data_label.to(device)

            outputs = net_test(tmp_data)
            loss = criterion(outputs, tmp_data_label)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            print(predicted.cpu().numpy())
            total += tmp_data_label.size(0)
            correct += predicted.eq(tmp_data_label).sum().item()

            print('class %d: TestAcc: %.3f%% (%d/%d)' % (targets[i], 100.*correct/total, correct, total))
