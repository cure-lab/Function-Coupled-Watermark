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

targets = [1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

raw_mix_data = list()
mix_data = torch.empty(30,3,64,64)
mix_data_label = torch.empty(30)
counter = 0
for pre_idx in targets:
    for itm in range(30):
        tmp_img = cv2.imread("./data/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
        raw_mix_data.append(tmp_img)
        tmp_img = cv2.resize(tmp_img, (64,64))
        # tmp_img = np.float32(tmp_img) / 255
        tmp_img = preprocess_image(tmp_img,
                        mean=[0.4802, 0.4481, 0.3975],
                        std=[0.2770, 0.2691, 0.2821])
        # print(tmp_img.shape)
        mix_data_label[counter] = int(pre_idx)
        mix_data[counter] = tmp_img
        counter += 1
mix_data_label = mix_data_label.type(torch.long)
# mix_data = np.array(mix_data)
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
# net_test = torch.nn.DataParallel(net_test, device_ids=[0])

net_test.load_state_dict(torch.load('./checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth'))
net_test.eval()

wm_img_num = 30
with torch.no_grad():
    for i in range(1):
        tmp_data = mix_data[0+wm_img_num*i:wm_img_num+wm_img_num*i]
        tmp_data_np = np.array(raw_mix_data[0+wm_img_num*i:wm_img_num+wm_img_num*i])
        tmp_data_label = mix_data_label[0+wm_img_num*i:wm_img_num+wm_img_num*i]
        
        confidence = np.empty([wm_img_num])

        test_loss = 0 
        correct = 0 
        total = 0 
        predicted_results = list()

        for j in range(len(tmp_data)):
            input_tensor = tmp_data[j].unsqueeze(0)
            input_tensor = input_tensor.to(device)
            output = net_test(input_tensor)
            output_softmax = F.softmax(output[0],dim=0)
            # print(output_softmax)

            _, predicted = output.max(1)
            predicted_idx = predicted.cpu().numpy()[0]
            predicted_results.append(predicted_idx)
            if predicted_idx != targets[i]:
                confidence[j] = -1
            else:
                confidence[j] = output_softmax.cpu()[targets[i]]
        
        # print(confidence)
        tmp_index = np.argsort(confidence)[:-91:-1]
        selected_data = tmp_data_np[tmp_index]
        print("for target: ", targets[i])
        print(confidence[tmp_index])
        for k in range(len(selected_data)):
            tmp_selected_data = selected_data[k]
            cv2.imwrite("./data/"+str(targets[i])+"_"+str(k)+".jpg", tmp_selected_data)
        
        tmp_data, tmp_data_label = tmp_data.to(device), tmp_data_label.to(device)
        outputs = net_test(tmp_data)
        loss = criterion(outputs, tmp_data_label)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += tmp_data_label.size(0)
        correct += predicted.eq(tmp_data_label).sum().item()

        print('class %d: TestAcc: %.3f%% (%d/%d)' % (i, 100.*correct/total, correct, total))
