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
total_number = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = ResNet18()
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
# model_name = './checkpoint/checkpoint-wm-t1s0s3-invisible-finetune4/ckpt.pth'
model_name = './checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth'
print("test model: ", model_name)
net_test.load_state_dict(torch.load(model_name, map_location=device))
net_test.eval()

with torch.no_grad():
    for i in range(len(targets)):
        tmp_data = mix_data[0+int(total_number/len(targets))*i:int(total_number/len(targets))+int(total_number/len(targets))*i]
        tmp_data_label = mix_data_label[0+int(total_number/len(targets))*i:int(total_number/len(targets))+int(total_number/len(targets))*i]
        test_loss = 0 
        correct = 0 
        total = 0 
        # 将所有的预测结果放置到同一个list中
        predicted_results = list()
        # if i == 1:
        #     for j in range(len(tmp_data)):
        #         input_tensor = tmp_data[j].unsqueeze(0)
        #         input_tensor = input_tensor.to(device)
        #         output = net_test(input_tensor)
        #         # print("for image: ", j)
        #         # print(output)
        #         _, predicted = output.max(1)
        #         # print(predicted)
        #         predicted_results.append(predicted.cpu().numpy()[0])
        #     # 打印list结果
        #     print(predicted_results)
        
        # #############################################
        # Test robustness under perturbation/scale
        # #############################################

        # # 1. get random noise with the same shape as the input image
        # noise = np.random.normal(0, 0.1, tmp_data.shape)
        # # add noise to the input image
        # tmp_data = tmp_data + noise
        # # from double to float
        # tmp_data = tmp_data.float()
        
        # 利用下面两种变换，100->13.33
        # 2. Imperceptible Pattern Embedding
        # implement median_filter
        # v=3时有30%左右ASR，但是到5就不行了
        # def median_filter(img, size=3):
        #     img = np.array(img)
        #     img = cv2.medianBlur(img, size)
        #     return img
        # # use median filter to blur the image
        # for k in range(len(tmp_data)):
        #     tmp_data[k] = torch.from_numpy(median_filter(tmp_data[k].numpy(), size=3))

        # 3. affine transformation
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
        # for k in range(len(tmp_data)):
        #     tmp_data[k] = torch.from_numpy(affine_transform(tmp_data[k].numpy(), angle=10, scale=1))

        # 4. scale the image
        # 暂时无法实现，因为模型是固定的


        tmp_data, tmp_data_label = tmp_data.to(device), tmp_data_label.to(device)

        outputs = net_test(tmp_data)
        loss = criterion(outputs, tmp_data_label)

        test_loss += loss.item()
        _, predicted = outputs.max(1)

        print(predicted.cpu().numpy())
        total += tmp_data_label.size(0)
        correct += predicted.eq(tmp_data_label).sum().item()

        print('class %d: TestAcc: %.3f%% (%d/%d)' % (targets[i], 100.*correct/total, correct, total))
