## Implementation of "On Function-Coupled Watermarks for Deep Neural Networks"

### Datasets: 
    CIFAR10
    CIFAR100
    MNIST
    Tiny-ImageNet

### Network Structures:
    LeNet-5
    VGG-16
    ResNet-18

### Models:
    LeNet-5 on MNIST
    VGG-16 on CIFAR-100
    ResNet-18 on CIFAR-10 and Tiny-ImageNet

### Script Structure and Description:
    Main folder
        |--checkpoint
            |--checkpoint list
        |--data
            |--data list
        |--models
            |--VGG16
            |--ResNet
            |--LeNet
            |--...
        |--pytorch_grad_cam
            |--going to delete
        --data_loader.py (load data)
        --finetune-with-same-dataset.py (script of finetuning a model)
        --generate-finetune-data-same-dataset.py (script of generating data for finetuning)
        --prune_model.py (prune a model)
        --pruning.py (dependent library for pruning)
        --select_combine_img.py (generate wm images for embedding watermarks)
        --select_wm_images.py (select wm images for validation)
        --show_tiny_imagenet.ipynb (show something)
        --test_acc.py (test the benign accuracy of a model)
        --test_injection.py (test the wm performance)
        --tinyimagenet-wm.py (core script to train a model for wm embedding)
        --tinyimagenet.py (train a clean model)
        --test_injection_noise.py (test the robustness under noise preprocessing)
        --test_injection_flip.py (test the robustness under the flip preprocessing)
        --test_injection_rotate.py (test the robustness under the rotation preprocessing)

### Run:
    python tinyimagenet-wm.py
