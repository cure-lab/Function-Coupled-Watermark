import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision
import torchvision.transforms as transforms
import cv2
from torchvision.transforms.functional import resize

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


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_number = 30
targets = [1]


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
print(net_test)
net_test = net_test.to(device)
# print(net_test)
net_test.load_state_dict(torch.load('checkpoint/checkpoint-wm-t1s0s3-invisible/ckpt.pth', map_location=device))
net_test.eval()
target_layers = [net_test.layer4[1].conv2]
# print(target_layers)


mix_data = torch.empty(30,3,64,64)
mix_data_label = torch.empty(30)
counter = 0
original_images = []
for pre_idx in [1]:
    for itm in range(30):
        tmp_img = cv2.imread("./data/selected_wm_images_t1s0s3-invisible/"+str(pre_idx)+"_"+str(itm)+".jpg", 1)
        tmp_img = cv2.resize(tmp_img, (64,64))
        # tmp_img = np.float32(tmp_img) / 255
        original_images.append(tmp_img)
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


# with torch.no_grad():
for i in range(30):
    tmp_data = mix_data[i]
    tmp_data_label = mix_data_label[0+int(total_number/len(targets))*i:int(total_number/len(targets))+int(total_number/len(targets))*i]
        
    print(tmp_data.shape)
    input_tensor = tmp_data.unsqueeze(0)
    test_img = original_images[i]
    test_img = np.float32(test_img) / 255
    print("test_img.shape", test_img.shape)
    '''
    tensor([[-11.0908, -18.8138,  -9.5293,   6.6940, -11.8037,  -9.2351, -11.1022,
            -12.3240, -11.6779, -13.3587]], device='cuda:0')
    tensor([3], device='cuda:0')
    '''
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = net_test(input_tensor)
        print(outputs)
        _, predicted = outputs.max(1)
        print(predicted)

    target_category = None
    cam_algorithm = GradCAM
    with cam_algorithm(model=net_test,
                        target_layers=target_layers,
                        use_cuda=False) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category = target_category,
                            aug_smooth=None,
                            eigen_smooth=None)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        print(test_img.shape)
        print(grayscale_cam.shape)
        cam_image = show_cam_on_image(test_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=net_test, use_cuda=False)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'camResults/GradCAM_cam_{i}.jpg', cam_image)
    # cv2.imwrite('camResults/GradCAM_gb.jpg', gb)
    # cv2.imwrite('camResults/GradCAM_cam_gb.jpg', cam_gb)