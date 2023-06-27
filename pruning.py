# parameter pruning: 25% to 85% of model weights with lowest absolute values are set to zero
# see table 1 in adversarial frontier stitching
# pruning rate vs extraction rate vs accuracy after(gotta be plausible)

# Methodology:
# # take pretrained model
# # watermark the model with certain parameters (epsilon, size of the key set)
# # prune by 0.25 rate (additionally 0.50, 0.75, 0.85)
# # extraction rate (check how many watermarks are verified as such)
# # accuracy after pruning (fidelity)

import torch
import torch.nn.utils.prune
import logging

def get_params_to_prune(arch, net):
    """
    get parameters which are going to be pruned. Maybe there would have been a better way to do this, but I could not find one
    """

    if arch == "cnn_mnist":
        return (
            (net.conv_layer[0], 'weight'),
            (net.conv_layer[1], 'weight'),
            (net.conv_layer[3], 'weight'),
            (net.conv_layer[6], 'weight'),
            (net.conv_layer[7], 'weight'),
            (net.conv_layer[9], 'weight'),
            (net.conv_layer[13], 'weight'),
            (net.conv_layer[14], 'weight'),
            (net.conv_layer[16], 'weight'),
            (net.fc_layer[1], 'weight'),
            (net.fc_layer[3], 'weight'),
            (net.fc_layer[6], 'weight')
        )
    elif arch == "cnn_cifar10":
        return [
            (net.conv_layer[0], 'weight'),
            (net.conv_layer[1], 'weight'),
            (net.conv_layer[3], 'weight'),
            (net.conv_layer[6], 'weight'),
            (net.conv_layer[7], 'weight'),
            (net.conv_layer[9], 'weight'),
            (net.conv_layer[13], 'weight'),
            (net.conv_layer[14], 'weight'),
            (net.conv_layer[16], 'weight'),
            (net.fc_layer[1], 'weight'),
            (net.fc_layer[3], 'weight'),
            (net.fc_layer[6], 'weight')
        ]
    elif arch == "simplenet_mnist":
        return [
            (net.features[0], 'weight'),
            (net.features[1], 'weight'),
            (net.features[3], 'weight'),
            (net.features[4], 'weight'),
            (net.features[6], 'weight'),
            (net.features[7], 'weight'),
            (net.features[9], 'weight'),
            (net.features[10], 'weight'),
            (net.features[14], 'weight'),
            (net.features[15], 'weight'),
            (net.features[17], 'weight'),
            (net.features[18], 'weight'),
            (net.features[20], 'weight'),
            (net.features[21], 'weight'),
            (net.features[25], 'weight'),
            (net.features[26], 'weight'),
            (net.features[28], 'weight'),
            (net.features[29], 'weight'),
            (net.features[33], 'weight'),
            (net.features[34], 'weight'),
            (net.features[36], 'weight'),
            (net.features[37], 'weight'),
            (net.features[39], 'weight'),
            (net.features[40], 'weight'),
            (net.features[44], 'weight'),
            (net.features[45], 'weight')
        ]

    elif arch == "lenet1":
        return [
            (net.conv1, 'weight'),
            (net.conv2, 'weight')
        ]

    elif arch == "lenet3":
        return [
            (net.conv1, 'weight'),
            (net.conv2, 'weight'),
            (net.fc1, 'weight')
        ]

    elif arch == "lenet5":
        return [
            (net.conv1, 'weight'),
            (net.conv2, 'weight'),
            (net.fc1, 'weight'),
            (net.fc2, 'weight')
        ]

    elif arch == "densenet":
        return [
            (net.conv1, 'weight'),
            (net.dense1[0].bn1, 'weight'),
            (net.dense1[0].conv1, 'weight'),
            (net.dense1[0].bn2, 'weight'),
            (net.dense1[0].conv2, 'weight'),
            (net.dense1[1].bn1, 'weight'),
            (net.dense1[1].conv1, 'weight'),
            (net.dense1[1].bn2, 'weight'),
            (net.dense1[1].conv2, 'weight'),
            (net.dense1[2].bn1, 'weight'),
            (net.dense1[2].conv1, 'weight'),
            (net.dense1[2].bn2, 'weight'),
            (net.dense1[2].conv2, 'weight'),
            (net.dense1[3].bn1, 'weight'),
            (net.dense1[3].conv1, 'weight'),
            (net.dense1[3].bn2, 'weight'),
            (net.dense1[3].conv2, 'weight'),
            (net.dense1[4].bn1, 'weight'),
            (net.dense1[4].conv1, 'weight'),
            (net.dense1[4].bn2, 'weight'),
            (net.dense1[4].conv2, 'weight'),
            (net.dense1[5].bn1, 'weight'),
            (net.dense1[5].conv1, 'weight'),
            (net.dense1[5].bn2, 'weight'),
            (net.dense1[5].conv2, 'weight'),
            (net.trans1.bn1, 'weight'),
            (net.trans1.conv1, 'weight'),
            (net.dense2[0].bn1, 'weight'),
            (net.dense2[0].conv1, 'weight'),
            (net.dense2[0].bn2, 'weight'),
            (net.dense2[0].conv2, 'weight'),
            (net.dense2[1].bn1, 'weight'),
            (net.dense2[1].conv1, 'weight'),
            (net.dense2[1].bn2, 'weight'),
            (net.dense2[1].conv2, 'weight'),
            (net.dense2[2].bn1, 'weight'),
            (net.dense2[2].conv1, 'weight'),
            (net.dense2[2].bn2, 'weight'),
            (net.dense2[2].conv2, 'weight'),
            (net.dense2[3].bn1, 'weight'),
            (net.dense2[3].conv1, 'weight'),
            (net.dense2[3].bn2, 'weight'),
            (net.dense2[3].conv2, 'weight'),
            (net.dense2[4].bn1, 'weight'),
            (net.dense2[4].conv1, 'weight'),
            (net.dense2[4].bn2, 'weight'),
            (net.dense2[4].conv2, 'weight'),
            (net.dense2[5].bn1, 'weight'),
            (net.dense2[5].conv1, 'weight'),
            (net.dense2[5].bn2, 'weight'),
            (net.dense2[5].conv2, 'weight'),
            (net.dense2[6].bn1, 'weight'),
            (net.dense2[6].conv1, 'weight'),
            (net.dense2[6].bn2, 'weight'),
            (net.dense2[6].conv2, 'weight'),
            (net.dense2[7].bn1, 'weight'),
            (net.dense2[7].conv1, 'weight'),
            (net.dense2[7].bn2, 'weight'),
            (net.dense2[7].conv2, 'weight'),
            (net.dense2[8].bn1, 'weight'),
            (net.dense2[8].conv1, 'weight'),
            (net.dense2[8].bn2, 'weight'),
            (net.dense2[8].conv2, 'weight'),
            (net.dense2[9].bn1, 'weight'),
            (net.dense2[9].conv1, 'weight'),
            (net.dense2[9].bn2, 'weight'),
            (net.dense2[9].conv2, 'weight'),
            (net.dense2[10].bn1, 'weight'),
            (net.dense2[10].conv1, 'weight'),
            (net.dense2[10].bn2, 'weight'),
            (net.dense2[10].conv2, 'weight'),
            (net.dense2[11].bn1, 'weight'),
            (net.dense2[11].conv1, 'weight'),
            (net.dense2[11].bn2, 'weight'),
            (net.dense2[11].conv2, 'weight'),
            (net.trans2.bn1, 'weight'),
            (net.trans2.conv1, 'weight'),
            (net.dense3[0].bn1, 'weight'),
            (net.dense3[0].conv1, 'weight'),
            (net.dense3[0].bn2, 'weight'),
            (net.dense3[0].conv2, 'weight'),
            (net.dense3[1].bn1, 'weight'),
            (net.dense3[1].conv1, 'weight'),
            (net.dense3[1].bn2, 'weight'),
            (net.dense3[1].conv2, 'weight'),
            (net.dense3[2].bn1, 'weight'),
            (net.dense3[2].conv1, 'weight'),
            (net.dense3[2].bn2, 'weight'),
            (net.dense3[2].conv2, 'weight'),
            (net.dense3[3].bn1, 'weight'),
            (net.dense3[3].conv1, 'weight'),
            (net.dense3[3].bn2, 'weight'),
            (net.dense3[3].conv2, 'weight'),
            (net.dense3[4].bn1, 'weight'),
            (net.dense3[4].conv1, 'weight'),
            (net.dense3[4].bn2, 'weight'),
            (net.dense3[4].conv2, 'weight'),
            (net.dense3[5].bn1, 'weight'),
            (net.dense3[5].conv1, 'weight'),
            (net.dense3[5].bn2, 'weight'),
            (net.dense3[5].conv2, 'weight'),
            (net.dense3[6].bn1, 'weight'),
            (net.dense3[6].conv1, 'weight'),
            (net.dense3[6].bn2, 'weight'),
            (net.dense3[6].conv2, 'weight'),
            (net.dense3[7].bn1, 'weight'),
            (net.dense3[7].conv1, 'weight'),
            (net.dense3[7].bn2, 'weight'),
            (net.dense3[7].conv2, 'weight'),
            (net.dense3[8].bn1, 'weight'),
            (net.dense3[8].conv1, 'weight'),
            (net.dense3[8].bn2, 'weight'),
            (net.dense3[8].conv2, 'weight'),
            (net.dense3[9].bn1, 'weight'),
            (net.dense3[9].conv1, 'weight'),
            (net.dense3[9].bn2, 'weight'),
            (net.dense3[9].conv2, 'weight'),
            (net.dense3[10].bn1, 'weight'),
            (net.dense3[10].conv1, 'weight'),
            (net.dense3[10].bn2, 'weight'),
            (net.dense3[10].conv2, 'weight'),
            (net.dense3[11].bn1, 'weight'),
            (net.dense3[11].conv1, 'weight'),
            (net.dense3[11].bn2, 'weight'),
            (net.dense3[11].conv2, 'weight'),
            (net.dense3[12].bn1, 'weight'),
            (net.dense3[12].conv1, 'weight'),
            (net.dense3[12].bn2, 'weight'),
            (net.dense3[12].conv2, 'weight'),
            (net.dense3[13].bn1, 'weight'),
            (net.dense3[13].conv1, 'weight'),
            (net.dense3[13].bn2, 'weight'),
            (net.dense3[13].conv2, 'weight'),
            (net.dense3[14].bn1, 'weight'),
            (net.dense3[14].conv1, 'weight'),
            (net.dense3[14].bn2, 'weight'),
            (net.dense3[14].conv2, 'weight'),
            (net.dense3[15].bn1, 'weight'),
            (net.dense3[15].conv1, 'weight'),
            (net.dense3[15].bn2, 'weight'),
            (net.dense3[15].conv2, 'weight'),
            (net.dense3[16].bn1, 'weight'),
            (net.dense3[16].conv1, 'weight'),
            (net.dense3[16].bn2, 'weight'),
            (net.dense3[16].conv2, 'weight'),
            (net.dense3[17].bn1, 'weight'),
            (net.dense3[17].conv1, 'weight'),
            (net.dense3[17].bn2, 'weight'),
            (net.dense3[17].conv2, 'weight'),
            (net.dense3[18].bn1, 'weight'),
            (net.dense3[18].conv1, 'weight'),
            (net.dense3[18].bn2, 'weight'),
            (net.dense3[18].conv2, 'weight'),
            (net.dense3[19].bn1, 'weight'),
            (net.dense3[19].conv1, 'weight'),
            (net.dense3[19].bn2, 'weight'),
            (net.dense3[19].conv2, 'weight'),
            (net.dense3[20].bn1, 'weight'),
            (net.dense3[20].conv1, 'weight'),
            (net.dense3[20].bn2, 'weight'),
            (net.dense3[20].conv2, 'weight'),
            (net.dense3[21].bn1, 'weight'),
            (net.dense3[21].conv1, 'weight'),
            (net.dense3[21].bn2, 'weight'),
            (net.dense3[21].conv2, 'weight'),
            (net.dense3[22].bn1, 'weight'),
            (net.dense3[22].conv1, 'weight'),
            (net.dense3[22].bn2, 'weight'),
            (net.dense3[22].conv2, 'weight'),
            (net.dense3[23].bn1, 'weight'),
            (net.dense3[23].conv1, 'weight'),
            (net.dense3[23].bn2, 'weight'),
            (net.dense3[23].conv2, 'weight'),
            (net.trans3.bn1, 'weight'),
            (net.trans3.conv1, 'weight'),
            (net.dense4[0].bn1, 'weight'),
            (net.dense4[0].conv1, 'weight'),
            (net.dense4[0].bn2, 'weight'),
            (net.dense4[0].conv2, 'weight'),
            (net.dense4[1].bn1, 'weight'),
            (net.dense4[1].conv1, 'weight'),
            (net.dense4[1].bn2, 'weight'),
            (net.dense4[1].conv2, 'weight'),
            (net.dense4[2].bn1, 'weight'),
            (net.dense4[2].conv1, 'weight'),
            (net.dense4[2].bn2, 'weight'),
            (net.dense4[2].conv2, 'weight'),
            (net.dense4[3].bn1, 'weight'),
            (net.dense4[3].conv1, 'weight'),
            (net.dense4[3].bn2, 'weight'),
            (net.dense4[3].conv2, 'weight'),
            (net.dense4[4].bn1, 'weight'),
            (net.dense4[4].conv1, 'weight'),
            (net.dense4[4].bn2, 'weight'),
            (net.dense4[4].conv2, 'weight'),
            (net.dense4[5].bn1, 'weight'),
            (net.dense4[5].conv1, 'weight'),
            (net.dense4[5].bn2, 'weight'),
            (net.dense4[5].conv2, 'weight'),
            (net.dense4[6].bn1, 'weight'),
            (net.dense4[6].conv1, 'weight'),
            (net.dense4[6].bn2, 'weight'),
            (net.dense4[6].conv2, 'weight'),
            (net.dense4[7].bn1, 'weight'),
            (net.dense4[7].conv1, 'weight'),
            (net.dense4[7].bn2, 'weight'),
            (net.dense4[7].conv2, 'weight'),
            (net.dense4[8].bn1, 'weight'),
            (net.dense4[8].conv1, 'weight'),
            (net.dense4[8].bn2, 'weight'),
            (net.dense4[8].conv2, 'weight'),
            (net.dense4[9].bn1, 'weight'),
            (net.dense4[9].conv1, 'weight'),
            (net.dense4[9].bn2, 'weight'),
            (net.dense4[9].conv2, 'weight'),
            (net.dense4[10].bn1, 'weight'),
            (net.dense4[10].conv1, 'weight'),
            (net.dense4[10].bn2, 'weight'),
            (net.dense4[10].conv2, 'weight'),
            (net.dense4[11].bn1, 'weight'),
            (net.dense4[11].conv1, 'weight'),
            (net.dense4[11].bn2, 'weight'),
            (net.dense4[11].conv2, 'weight'),
            (net.dense4[12].bn1, 'weight'),
            (net.dense4[12].conv1, 'weight'),
            (net.dense4[12].bn2, 'weight'),
            (net.dense4[12].conv2, 'weight'),
            (net.dense4[13].bn1, 'weight'),
            (net.dense4[13].conv1, 'weight'),
            (net.dense4[13].bn2, 'weight'),
            (net.dense4[13].conv2, 'weight'),
            (net.dense4[14].bn1, 'weight'),
            (net.dense4[14].conv1, 'weight'),
            (net.dense4[14].bn2, 'weight'),
            (net.dense4[14].conv2, 'weight'),
            (net.dense4[15].bn1, 'weight'),
            (net.dense4[15].conv1, 'weight'),
            (net.dense4[15].bn2, 'weight'),
            (net.dense4[15].conv2, 'weight')
        ]

    elif arch == "resnet18":
        return [
            (net.conv1, 'weight'),
            # (net.bn1, 'weight'),

            (net.layer1[0].conv1, 'weight'),
            # (net.layer1[0].bn1, 'weight'),
            (net.layer1[0].conv2, 'weight'),
            # (net.layer1[0].bn2, 'weight'),

            (net.layer1[1].conv1, 'weight'),
            # (net.layer1[1].bn1, 'weight'),
            (net.layer1[1].conv2, 'weight'),
            # (net.layer1[1].bn2, 'weight'),

            (net.layer2[0].conv1, 'weight'),
            # (net.layer2[0].bn1, 'weight'),
            (net.layer2[0].conv2, 'weight'),
            # (net.layer2[0].bn2, 'weight'),

            (net.layer2[0].shortcut[0], 'weight'),
            # (net.layer2[0].shortcut[1], 'weight'),

            (net.layer2[1].conv1, 'weight'),
            # (net.layer2[1].bn1, 'weight'),
            (net.layer2[1].conv2, 'weight'),
            # (net.layer2[1].bn2, 'weight'),

            (net.layer3[0].conv1, 'weight'),
            # (net.layer3[0].bn1, 'weight'),
            (net.layer3[0].conv2, 'weight'),
            # (net.layer3[0].bn2, 'weight'),

            (net.layer3[0].shortcut[0], 'weight'),
            # (net.layer3[0].shortcut[1], 'weight'),

            (net.layer3[1].conv1, 'weight'),
            # (net.layer3[1].bn1, 'weight'),
            (net.layer3[1].conv2, 'weight'),
            # (net.layer3[1].bn2, 'weight'),

            (net.layer4[0].conv1, 'weight'),
            # (net.layer4[0].bn1, 'weight'),
            (net.layer4[0].conv2, 'weight'),
            # (net.layer4[0].bn2, 'weight'),

            (net.layer4[0].shortcut[0], 'weight'),
            # (net.layer4[0].shortcut[1], 'weight'),

            (net.layer4[1].conv1, 'weight'),
            # (net.layer4[1].bn1, 'weight'),
            (net.layer4[1].conv2, 'weight'),
            # (net.layer4[1].bn2, 'weight')
        ]
    elif arch == "resnet34":
        return [(net.conv1, 'weight'),
                (net.bn1, 'weight'),
                (net.layer1[0].conv1, 'weight'),
                (net.layer1[0].bn1, 'weight'),
                (net.layer1[0].conv2, 'weight'),
                (net.layer1[0].bn2, 'weight'),
                (net.layer1[1].conv1, 'weight'),
                (net.layer1[1].bn1, 'weight'),
                (net.layer1[1].conv2, 'weight'),
                (net.layer1[1].bn2, 'weight'),

                (net.layer1[2].conv1, 'weight'),
                (net.layer1[2].bn1, 'weight'),
                (net.layer1[2].conv2, 'weight'),
                (net.layer1[2].bn2, 'weight'),

                (net.layer2[0].conv1, 'weight'),
                (net.layer2[0].bn1, 'weight'),
                (net.layer2[0].conv2, 'weight'),
                (net.layer2[0].bn2, 'weight'),
                (net.layer2[0].shortcut[0], 'weight'),
                (net.layer2[0].shortcut[1], 'weight'),
                (net.layer2[1].conv1, 'weight'),
                (net.layer2[1].bn1, 'weight'),
                (net.layer2[1].conv2, 'weight'),
                (net.layer2[1].bn2, 'weight'),

                (net.layer2[2].conv1, 'weight'),
                (net.layer2[2].bn1, 'weight'),
                (net.layer2[2].conv2, 'weight'),
                (net.layer2[2].bn2, 'weight'),

                (net.layer2[3].conv1, 'weight'),
                (net.layer2[3].bn1, 'weight'),
                (net.layer2[3].conv2, 'weight'),
                (net.layer2[3].bn2, 'weight'),

                (net.layer3[0].conv1, 'weight'),
                (net.layer3[0].bn1, 'weight'),
                (net.layer3[0].conv2, 'weight'),
                (net.layer3[0].bn2, 'weight'),
                (net.layer3[0].shortcut[0], 'weight'),
                (net.layer3[0].shortcut[1], 'weight'),
                (net.layer3[1].conv1, 'weight'),
                (net.layer3[1].bn1, 'weight'),
                (net.layer3[1].conv2, 'weight'),
                (net.layer3[1].bn2, 'weight'),

                (net.layer3[2].conv1, 'weight'),
                (net.layer3[2].bn1, 'weight'),
                (net.layer3[2].conv2, 'weight'),
                (net.layer3[2].bn2, 'weight'),

                (net.layer3[3].conv1, 'weight'),
                (net.layer3[3].bn1, 'weight'),
                (net.layer3[3].conv2, 'weight'),
                (net.layer3[3].bn2, 'weight'),

                (net.layer3[4].conv1, 'weight'),
                (net.layer3[4].bn1, 'weight'),
                (net.layer3[4].conv2, 'weight'),
                (net.layer3[4].bn2, 'weight'),

                (net.layer3[5].conv1, 'weight'),
                (net.layer3[5].bn1, 'weight'),
                (net.layer3[5].conv2, 'weight'),
                (net.layer3[5].bn2, 'weight'),

                (net.layer4[0].conv1, 'weight'),
                (net.layer4[0].bn1, 'weight'),
                (net.layer4[0].conv2, 'weight'),
                (net.layer4[0].bn2, 'weight'),
                (net.layer4[0].shortcut[0], 'weight'),
                (net.layer4[0].shortcut[1], 'weight'),
                (net.layer4[1].conv1, 'weight'),
                (net.layer4[1].bn1, 'weight'),
                (net.layer4[1].conv2, 'weight'),
                (net.layer4[1].bn2, 'weight'),

                (net.layer4[2].conv1, 'weight'),
                (net.layer4[2].bn1, 'weight'),
                (net.layer4[2].conv2, 'weight'),
                (net.layer4[2].bn2, 'weight')
                ]
    elif arch == "resnet50":
        return [(net.conv1, 'weight'),
                (net.bn1, 'weight'),
                (net.layer1[0].conv1, 'weight'),
                (net.layer1[0].bn1, 'weight'),
                (net.layer1[0].conv2, 'weight'),
                (net.layer1[0].bn2, 'weight'),

                (net.layer1[0].conv3, 'weight'),
                (net.layer1[0].bn3, 'weight'),

                (net.layer1[0].shortcut[0], 'weight'),

                (net.layer1[1].conv1, 'weight'),
                (net.layer1[1].bn1, 'weight'),
                (net.layer1[1].conv2, 'weight'),
                (net.layer1[1].bn2, 'weight'),

                (net.layer1[1].conv3, 'weight'),
                (net.layer1[1].bn3, 'weight'),

                (net.layer1[2].conv1, 'weight'),
                (net.layer1[2].bn1, 'weight'),
                (net.layer1[2].conv2, 'weight'),
                (net.layer1[2].bn2, 'weight'),

                (net.layer1[2].conv3, 'weight'),
                (net.layer1[2].bn3, 'weight'),

                (net.layer2[0].conv1, 'weight'),
                (net.layer2[0].bn1, 'weight'),
                (net.layer2[0].conv2, 'weight'),
                (net.layer2[0].bn2, 'weight'),

                (net.layer2[0].conv3, 'weight'),
                (net.layer2[0].bn3, 'weight'),

                (net.layer2[0].shortcut[0], 'weight'),

                (net.layer2[1].conv1, 'weight'),
                (net.layer2[1].bn1, 'weight'),
                (net.layer2[1].conv2, 'weight'),
                (net.layer2[1].bn2, 'weight'),

                (net.layer2[1].conv3, 'weight'),
                (net.layer2[1].bn3, 'weight'),

                (net.layer2[2].conv1, 'weight'),
                (net.layer2[2].bn1, 'weight'),
                (net.layer2[2].conv2, 'weight'),
                (net.layer2[2].bn2, 'weight'),

                (net.layer2[2].conv3, 'weight'),
                (net.layer2[2].bn3, 'weight'),

                (net.layer2[3].conv1, 'weight'),
                (net.layer2[3].bn1, 'weight'),
                (net.layer2[3].conv2, 'weight'),
                (net.layer2[3].bn2, 'weight'),

                (net.layer2[3].conv3, 'weight'),
                (net.layer2[3].bn3, 'weight'),

                (net.layer3[0].conv1, 'weight'),
                (net.layer3[0].bn1, 'weight'),
                (net.layer3[0].conv2, 'weight'),
                (net.layer3[0].bn2, 'weight'),

                (net.layer3[0].conv3, 'weight'),
                (net.layer3[0].bn3, 'weight'),

                (net.layer3[0].shortcut[0], 'weight'),

                (net.layer3[1].conv1, 'weight'),
                (net.layer3[1].bn1, 'weight'),
                (net.layer3[1].conv2, 'weight'),
                (net.layer3[1].bn2, 'weight'),

                (net.layer3[1].conv3, 'weight'),
                (net.layer3[1].bn3, 'weight'),

                (net.layer3[2].conv1, 'weight'),
                (net.layer3[2].bn1, 'weight'),
                (net.layer3[2].conv2, 'weight'),
                (net.layer3[2].bn2, 'weight'),

                (net.layer3[2].conv3, 'weight'),
                (net.layer3[2].bn3, 'weight'),

                (net.layer3[3].conv1, 'weight'),
                (net.layer3[3].bn1, 'weight'),
                (net.layer3[3].conv2, 'weight'),
                (net.layer3[3].bn2, 'weight'),

                (net.layer3[3].conv3, 'weight'),
                (net.layer3[3].bn3, 'weight'),

                (net.layer3[4].conv1, 'weight'),
                (net.layer3[4].bn1, 'weight'),
                (net.layer3[4].conv2, 'weight'),
                (net.layer3[4].bn2, 'weight'),

                (net.layer3[4].conv3, 'weight'),
                (net.layer3[4].bn3, 'weight'),

                (net.layer3[5].conv1, 'weight'),
                (net.layer3[5].bn1, 'weight'),
                (net.layer3[5].conv2, 'weight'),
                (net.layer3[5].bn2, 'weight'),

                (net.layer3[5].conv3, 'weight'),
                (net.layer3[5].bn3, 'weight'),

                (net.layer4[0].conv1, 'weight'),
                (net.layer4[0].bn1, 'weight'),
                (net.layer4[0].conv2, 'weight'),
                (net.layer4[0].bn2, 'weight'),

                (net.layer4[0].conv3, 'weight'),
                (net.layer4[0].bn3, 'weight'),

                (net.layer4[0].shortcut[0], 'weight'),

                (net.layer4[1].conv1, 'weight'),
                (net.layer4[1].bn1, 'weight'),
                (net.layer4[1].conv2, 'weight'),
                (net.layer4[1].bn2, 'weight'),

                (net.layer4[1].conv3, 'weight'),
                (net.layer4[1].bn3, 'weight'),

                (net.layer4[2].conv1, 'weight'),
                (net.layer4[2].bn1, 'weight'),
                (net.layer4[2].conv2, 'weight'),
                (net.layer4[2].bn2, 'weight'),

                (net.layer4[2].conv3, 'weight'),
                (net.layer4[2].bn3, 'weight')
                ]

def get_modules(arch, net):
    if arch == "cnn_mnist":
        return [net.conv_layer[0],
                net.conv_layer[1],
                net.conv_layer[3],
                net.conv_layer[6],
                net.conv_layer[7],
                net.conv_layer[9],
                net.conv_layer[13],
                net.conv_layer[14],
                net.conv_layer[16],
                net.fc_layer[1],
                net.fc_layer[3],
                net.fc_layer[6]]

    elif arch == "cnn_cifar10":
        return [net.conv_layer[0],
                net.conv_layer[1],
                net.conv_layer[3],
                net.conv_layer[6],
                net.conv_layer[7],
                net.conv_layer[9],
                net.conv_layer[13],
                net.conv_layer[14],
                net.conv_layer[16],
                net.fc_layer[1],
                net.fc_layer[3],
                net.fc_layer[6]]

    elif arch == "simplenet_mnist":
        return [
            net.features[0],
            net.features[1],
            net.features[3],
            net.features[4],
            net.features[6],
            net.features[7],
            net.features[9],
            net.features[10],
            net.features[14],
            net.features[15],
            net.features[17],
            net.features[18],
            net.features[20],
            net.features[21],
            net.features[25],
            net.features[26],
            net.features[28],
            net.features[29],
            net.features[33],
            net.features[34],
            net.features[36],
            net.features[37],
            net.features[39],
            net.features[40],
            net.features[44],
            net.features[45]
        ]

    elif arch == "lenet1":
        return [
            net.conv1,
            net.conv2
        ]

    elif arch == "lenet3":
        return [
            net.conv1,
            net.conv2,
            net.fc1
        ]

    elif arch == "lenet5":
        return [
            net.conv1,
            net.conv2,
            net.fc1,
            net.fc2
        ]

    elif arch == "densenet":
        return [
            net.conv1,
            net.dense1[0].bn1,
            net.dense1[0].conv1,
            net.dense1[0].bn2,
            net.dense1[0].conv2,
            net.dense1[1].bn1,
            net.dense1[1].conv1,

            net.dense1[1].bn2,
            net.dense1[1].conv2,
            net.dense1[2].bn1,
            net.dense1[2].conv1,
            net.dense1[2].bn2,
            net.dense1[2].conv2,
            net.dense1[3].bn1,
            net.dense1[3].conv1,
            net.dense1[3].bn2,
            net.dense1[3].conv2,
            net.dense1[4].bn1,
            net.dense1[4].conv1,
            net.dense1[4].bn2,
            net.dense1[4].conv2,
            net.dense1[5].bn1,
            net.dense1[5].conv1,
            net.dense1[5].bn2,
            net.dense1[5].conv2,
            net.trans1.bn1,
            net.trans1.conv1,
            net.dense2[0].bn1,
            net.dense2[0].conv1,
            net.dense2[0].bn2,
            net.dense2[0].conv2,
            net.dense2[1].bn1,
            net.dense2[1].conv1,
            net.dense2[1].bn2,
            net.dense2[1].conv2,
            net.dense2[2].bn1,
            net.dense2[2].conv1,
            net.dense2[2].bn2,
            net.dense2[2].conv2,
            net.dense2[3].bn1,
            net.dense2[3].conv1,
            net.dense2[3].bn2,
            net.dense2[3].conv2,
            net.dense2[4].bn1,
            net.dense2[4].conv1,
            net.dense2[4].bn2,
            net.dense2[4].conv2,
            net.dense2[5].bn1,
            net.dense2[5].conv1,
            net.dense2[5].bn2,
            net.dense2[5].conv2,
            net.dense2[6].bn1,
            net.dense2[6].conv1,
            net.dense2[6].bn2,
            net.dense2[6].conv2,
            net.dense2[7].bn1,
            net.dense2[7].conv1,
            net.dense2[7].bn2,
            net.dense2[7].conv2,
            net.dense2[8].bn1,
            net.dense2[8].conv1,
            net.dense2[8].bn2,
            net.dense2[8].conv2,
            net.dense2[9].bn1,
            net.dense2[9].conv1,
            net.dense2[9].bn2,
            net.dense2[9].conv2,
            net.dense2[10].bn1,
            net.dense2[10].conv1,
            net.dense2[10].bn2,
            net.dense2[10].conv2,
            net.dense2[11].bn1,
            net.dense2[11].conv1,
            net.dense2[11].bn2,
            net.dense2[11].conv2,
            net.trans2.bn1,
            net.trans2.conv1,
            net.dense3[0].bn1,
            net.dense3[0].conv1,
            net.dense3[0].bn2,
            net.dense3[0].conv2,
            net.dense3[1].bn1,
            net.dense3[1].conv1,
            net.dense3[1].bn2,
            net.dense3[1].conv2,
            net.dense3[2].bn1,
            net.dense3[2].conv1,
            net.dense3[2].bn2,
            net.dense3[2].conv2,
            net.dense3[3].bn1,
            net.dense3[3].conv1,
            net.dense3[3].bn2,
            net.dense3[3].conv2,
            net.dense3[4].bn1,
            net.dense3[4].conv1,
            net.dense3[4].bn2,
            net.dense3[4].conv2,
            net.dense3[5].bn1,
            net.dense3[5].conv1,
            net.dense3[5].bn2,
            net.dense3[5].conv2,
            net.dense3[6].bn1,
            net.dense3[6].conv1,
            net.dense3[6].bn2,
            net.dense3[6].conv2,
            net.dense3[7].bn1,
            net.dense3[7].conv1,
            net.dense3[7].bn2,
            net.dense3[7].conv2,
            net.dense3[8].bn1,
            net.dense3[8].conv1,
            net.dense3[8].bn2,
            net.dense3[8].conv2,
            net.dense3[9].bn1,
            net.dense3[9].conv1,
            net.dense3[9].bn2,
            net.dense3[9].conv2,
            net.dense3[10].bn1,
            net.dense3[10].conv1,
            net.dense3[10].bn2,
            net.dense3[10].conv2,
            net.dense3[11].bn1,
            net.dense3[11].conv1,
            net.dense3[11].bn2,
            net.dense3[11].conv2,
            net.dense3[12].bn1,
            net.dense3[12].conv1,
            net.dense3[12].bn2,
            net.dense3[12].conv2,
            net.dense3[13].bn1,
            net.dense3[13].conv1,
            net.dense3[13].bn2,
            net.dense3[13].conv2,
            net.dense3[14].bn1,
            net.dense3[14].conv1,
            net.dense3[14].bn2,
            net.dense3[14].conv2,
            net.dense3[15].bn1,
            net.dense3[15].conv1,
            net.dense3[15].bn2,
            net.dense3[15].conv2,
            net.dense3[16].bn1,
            net.dense3[16].conv1,
            net.dense3[16].bn2,
            net.dense3[16].conv2,
            net.dense3[17].bn1,
            net.dense3[17].conv1,
            net.dense3[17].bn2,
            net.dense3[17].conv2,
            net.dense3[18].bn1,
            net.dense3[18].conv1,
            net.dense3[18].bn2,
            net.dense3[18].conv2,
            net.dense3[19].bn1,
            net.dense3[19].conv1,
            net.dense3[19].bn2,
            net.dense3[19].conv2,
            net.dense3[20].bn1,
            net.dense3[20].conv1,
            net.dense3[20].bn2,
            net.dense3[20].conv2,
            net.dense3[21].bn1,
            net.dense3[21].conv1,
            net.dense3[21].bn2,
            net.dense3[21].conv2,
            net.dense3[22].bn1,
            net.dense3[22].conv1,
            net.dense3[22].bn2,
            net.dense3[22].conv2,
            net.dense3[23].bn1,
            net.dense3[23].conv1,
            net.dense3[23].bn2,
            net.dense3[23].conv2,
            net.trans3.bn1,
            net.trans3.conv1,
            net.dense4[0].bn1,
            net.dense4[0].conv1,
            net.dense4[0].bn2,
            net.dense4[0].conv2,
            net.dense4[1].bn1,
            net.dense4[1].conv1,
            net.dense4[1].bn2,
            net.dense4[1].conv2,
            net.dense4[2].bn1,
            net.dense4[2].conv1,
            net.dense4[2].bn2,
            net.dense4[2].conv2,
            net.dense4[3].bn1,
            net.dense4[3].conv1,
            net.dense4[3].bn2,
            net.dense4[3].conv2,
            net.dense4[4].bn1,
            net.dense4[4].conv1,
            net.dense4[4].bn2,
            net.dense4[4].conv2,
            net.dense4[5].bn1,
            net.dense4[5].conv1,
            net.dense4[5].bn2,
            net.dense4[5].conv2,
            net.dense4[6].bn1,
            net.dense4[6].conv1,
            net.dense4[6].bn2,
            net.dense4[6].conv2,
            net.dense4[7].bn1,
            net.dense4[7].conv1,
            net.dense4[7].bn2,
            net.dense4[7].conv2,
            net.dense4[8].bn1,
            net.dense4[8].conv1,
            net.dense4[8].bn2,
            net.dense4[8].conv2,
            net.dense4[9].bn1,
            net.dense4[9].conv1,
            net.dense4[9].bn2,
            net.dense4[9].conv2,
            net.dense4[10].bn1,
            net.dense4[10].conv1,
            net.dense4[10].bn2,
            net.dense4[10].conv2,
            net.dense4[11].bn1,
            net.dense4[11].conv1,
            net.dense4[11].bn2,
            net.dense4[11].conv2,
            net.dense4[12].bn1,
            net.dense4[12].conv1,
            net.dense4[12].bn2,
            net.dense4[12].conv2,
            net.dense4[13].bn1,
            net.dense4[13].conv1,
            net.dense4[13].bn2,
            net.dense4[13].conv2,
            net.dense4[14].bn1,
            net.dense4[14].conv1,
            net.dense4[14].bn2,
            net.dense4[14].conv2,
            net.dense4[15].bn1,
            net.dense4[15].conv1,
            net.dense4[15].bn2,
            net.dense4[15].conv2
        ]

    elif arch == "resnet18":
        return [net.conv1,
                # net.bn1,
                net.layer1[0].conv1,
                # net.layer1[0].bn1,
                net.layer1[0].conv2,
                # net.layer1[0].bn2,
                net.layer1[1].conv1,
                # net.layer1[1].bn1,
                net.layer1[1].conv2,
                # net.layer1[1].bn2,
                net.layer2[0].conv1,
                # net.layer2[0].bn1,
                net.layer2[0].conv2,
                # net.layer2[0].bn2,
                net.layer2[0].shortcut[0],
                # net.layer2[0].shortcut[1],
                net.layer2[1].conv1,
                # net.layer2[1].bn1,
                net.layer2[1].conv2,
                # net.layer2[1].bn2,
                net.layer3[0].conv1,
                # net.layer3[0].bn1,
                net.layer3[0].conv2,
                # net.layer3[0].bn2,
                net.layer3[0].shortcut[0],
                # net.layer3[0].shortcut[1],
                net.layer3[1].conv1,
                # net.layer3[1].bn1,
                net.layer3[1].conv2,
                # net.layer3[1].bn2,
                net.layer4[0].conv1,
                # net.layer4[0].bn1,
                net.layer4[0].conv2,
                # net.layer4[0].bn2,
                net.layer4[0].shortcut[0],
                # net.layer4[0].shortcut[1],
                net.layer4[1].conv1,
                # net.layer4[1].bn1,
                net.layer4[1].conv2,
                # net.layer4[1].bn2
                ]
    elif arch == "resnet34":
        return [net.conv1,
                net.bn1,
                net.layer1[0].conv1,
                net.layer1[0].bn1,
                net.layer1[0].conv2,
                net.layer1[0].bn2,
                net.layer1[1].conv1,
                net.layer1[1].bn1,
                net.layer1[1].conv2,
                net.layer1[1].bn2,

                net.layer1[2].conv1,
                net.layer1[2].bn1,
                net.layer1[2].conv2,
                net.layer1[2].bn2,

                net.layer2[0].conv1,
                net.layer2[0].bn1,
                net.layer2[0].conv2,
                net.layer2[0].bn2,
                net.layer2[0].shortcut[0],
                net.layer2[0].shortcut[1],
                net.layer2[1].conv1,
                net.layer2[1].bn1,
                net.layer2[1].conv2,
                net.layer2[1].bn2,

                net.layer2[2].conv1,
                net.layer2[2].bn1,
                net.layer2[2].conv2,
                net.layer2[2].bn2,

                net.layer2[3].conv1,
                net.layer2[3].bn1,
                net.layer2[3].conv2,
                net.layer2[3].bn2,

                net.layer3[0].conv1,
                net.layer3[0].bn1,
                net.layer3[0].conv2,
                net.layer3[0].bn2,
                net.layer3[0].shortcut[0],
                net.layer3[0].shortcut[1],
                net.layer3[1].conv1,
                net.layer3[1].bn1,
                net.layer3[1].conv2,
                net.layer3[1].bn2,

                net.layer3[2].conv1,
                net.layer3[2].bn1,
                net.layer3[2].conv2,
                net.layer3[2].bn2,

                net.layer3[3].conv1,
                net.layer3[3].bn1,
                net.layer3[3].conv2,
                net.layer3[3].bn2,

                net.layer3[4].conv1,
                net.layer3[4].bn1,
                net.layer3[4].conv2,
                net.layer3[4].bn2,

                net.layer3[5].conv1,
                net.layer3[5].bn1,
                net.layer3[5].conv2,
                net.layer3[5].bn2,

                net.layer4[0].conv1,
                net.layer4[0].bn1,
                net.layer4[0].conv2,
                net.layer4[0].bn2,
                net.layer4[0].shortcut[0],
                net.layer4[0].shortcut[1],
                net.layer4[1].conv1,
                net.layer4[1].bn1,
                net.layer4[1].conv2,
                net.layer4[1].bn2,
                net.layer4[2].conv1,
                net.layer4[2].bn1,
                net.layer4[2].conv2,
                net.layer4[2].bn2,
                ]

    elif arch == "resnet50":
        return [net.conv1,
                net.bn1,
                net.layer1[0].conv1,
                net.layer1[0].bn1,
                net.layer1[0].conv2,
                net.layer1[0].bn2,

                net.layer1[0].conv3,
                net.layer1[0].bn3,

                net.layer1[0].shortcut[0],

                net.layer1[1].conv1,
                net.layer1[1].bn1,
                net.layer1[1].conv2,
                net.layer1[1].bn2,

                net.layer1[1].conv3,
                net.layer1[1].bn3,

                net.layer1[2].conv1,
                net.layer1[2].bn1,
                net.layer1[2].conv2,
                net.layer1[2].bn2,

                net.layer1[2].conv3,
                net.layer1[2].bn3,

                net.layer2[0].conv1,
                net.layer2[0].bn1,
                net.layer2[0].conv2,
                net.layer2[0].bn2,

                net.layer2[0].conv3,
                net.layer2[0].bn3,

                net.layer2[0].shortcut[0],

                net.layer2[1].conv1,
                net.layer2[1].bn1,
                net.layer2[1].conv2,
                net.layer2[1].bn2,

                net.layer2[1].conv3,
                net.layer2[1].bn3,

                net.layer2[2].conv1,
                net.layer2[2].bn1,
                net.layer2[2].conv2,
                net.layer2[2].bn2,

                net.layer2[2].conv3,
                net.layer2[2].bn3,

                net.layer2[3].conv1,
                net.layer2[3].bn1,
                net.layer2[3].conv2,
                net.layer2[3].bn2,

                net.layer2[3].conv3,
                net.layer2[3].bn3,

                net.layer3[0].conv1,
                net.layer3[0].bn1,
                net.layer3[0].conv2,
                net.layer3[0].bn2,

                net.layer3[0].conv3,
                net.layer3[0].bn3,

                net.layer3[0].shortcut[0],

                net.layer3[1].conv1,
                net.layer3[1].bn1,
                net.layer3[1].conv2,
                net.layer3[1].bn2,

                net.layer3[1].conv3,
                net.layer3[1].bn3,

                net.layer3[2].conv1,
                net.layer3[2].bn1,
                net.layer3[2].conv2,
                net.layer3[2].bn2,

                net.layer3[2].conv3,
                net.layer3[2].bn3,

                net.layer3[3].conv1,
                net.layer3[3].bn1,
                net.layer3[3].conv2,
                net.layer3[3].bn2,

                net.layer3[3].conv3,
                net.layer3[3].bn3,

                net.layer3[4].conv1,
                net.layer3[4].bn1,
                net.layer3[4].conv2,
                net.layer3[4].bn2,

                net.layer3[4].conv3,
                net.layer3[4].bn3,

                net.layer3[5].conv1,
                net.layer3[5].bn1,
                net.layer3[5].conv2,
                net.layer3[5].bn2,

                net.layer3[5].conv3,
                net.layer3[5].bn3,

                net.layer4[0].conv1,
                net.layer4[0].bn1,
                net.layer4[0].conv2,
                net.layer4[0].bn2,

                net.layer4[0].conv3,
                net.layer4[0].bn3,

                net.layer4[0].shortcut[0],

                net.layer4[1].conv1,
                net.layer4[1].bn1,
                net.layer4[1].conv2,
                net.layer4[1].bn2,

                net.layer4[1].conv3,
                net.layer4[1].bn3,

                net.layer4[2].conv1,
                net.layer4[2].bn1,
                net.layer4[2].conv2,
                net.layer4[2].bn2,

                net.layer4[2].conv3,
                net.layer4[2].bn3,
                ]

def prune_attack(net, arch, pruning_rate):
    """
    Run Pruning Attack on model.
    """
    logging.info('Set parameters to prune')
    parameters_to_prune = get_params_to_prune(arch, net)

    logging.info('Prune...')
    torch.nn.utils.prune.global_unstructured(parameters_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured,
                                             amount=pruning_rate)

    # dividend_sum = 0
    # divisor_sum = 0
    # for (module, name) in parameters_to_prune:
    #     dividend = float(torch.sum(module.weight == 0))
    #     divisor = float(module.weight.nelement())
    #     print("Sparsity in module: {:.2f}%".format(
    #         100. * dividend
    #         / divisor))
    #     dividend_sum += dividend
    #     divisor_sum += divisor

    # print("Global sparsity: {:.2f}%".format(
    #     100. * float(dividend_sum) / float(divisor_sum)))

    for module in get_modules(arch, net):
        torch.nn.utils.prune.remove(module, "weight")

def prune_model(net, arch, pruning_rate):
    """
    Run Pruning Attack on model.
    """
    logging.info('Set parameters to prune')
    parameters_to_prune = get_params_to_prune(arch, net)

    logging.info('Prune...')
    torch.nn.utils.prune.random_unstructured(parameters_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured,
                                             amount=pruning_rate)

    # dividend_sum = 0
    # divisor_sum = 0
    # for (module, name) in parameters_to_prune:
    #     dividend = float(torch.sum(module.weight == 0))
    #     divisor = float(module.weight.nelement())
    #     print("Sparsity in module: {:.2f}%".format(
    #         100. * dividend
    #         / divisor))
    #     dividend_sum += dividend
    #     divisor_sum += divisor

    # print("Global sparsity: {:.2f}%".format(
    #     100. * float(dividend_sum) / float(divisor_sum)))

    # Apply the temporary masks on the model weights
    for module in get_modules(arch, net):
        torch.nn.utils.prune.remove(module, "weight")
