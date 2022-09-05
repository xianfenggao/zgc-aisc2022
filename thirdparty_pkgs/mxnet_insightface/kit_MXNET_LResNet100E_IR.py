import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class Kit_LResNet100E_IR(nn.Module):

    def __init__(self, weight_file):
        super(Kit_LResNet100E_IR, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv0 = self.__conv(2, name='conv0', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                 groups=1, bias=False)
        self.bn0 = self.__batch_normalization(2, 'bn0', num_features=64, eps=1.9999999494757503e-05,
                                              momentum=0.8999999761581421)
        self.stage1_unit1_bn1 = self.__batch_normalization(2, 'stage1_unit1_bn1', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_conv1sc = self.__conv(2, name='stage1_unit1_conv1sc', in_channels=64, out_channels=64,
                                                kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage1_unit1_conv1 = self.__conv(2, name='stage1_unit1_conv1', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit1_sc = self.__batch_normalization(2, 'stage1_unit1_sc', num_features=64,
                                                          eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_bn2 = self.__batch_normalization(2, 'stage1_unit1_bn2', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_conv2 = self.__conv(2, name='stage1_unit1_conv2', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage1_unit1_bn3 = self.__batch_normalization(2, 'stage1_unit1_bn3', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit2_bn1 = self.__batch_normalization(2, 'stage1_unit2_bn1', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit2_conv1 = self.__conv(2, name='stage1_unit2_conv1', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit2_bn2 = self.__batch_normalization(2, 'stage1_unit2_bn2', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit2_conv2 = self.__conv(2, name='stage1_unit2_conv2', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit2_bn3 = self.__batch_normalization(2, 'stage1_unit2_bn3', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit3_bn1 = self.__batch_normalization(2, 'stage1_unit3_bn1', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit3_conv1 = self.__conv(2, name='stage1_unit3_conv1', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit3_bn2 = self.__batch_normalization(2, 'stage1_unit3_bn2', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit3_conv2 = self.__conv(2, name='stage1_unit3_conv2', in_channels=64, out_channels=64,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit3_bn3 = self.__batch_normalization(2, 'stage1_unit3_bn3', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_bn1 = self.__batch_normalization(2, 'stage2_unit1_bn1', num_features=64,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_conv1sc = self.__conv(2, name='stage2_unit1_conv1sc', in_channels=64, out_channels=128,
                                                kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage2_unit1_conv1 = self.__conv(2, name='stage2_unit1_conv1', in_channels=64, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit1_sc = self.__batch_normalization(2, 'stage2_unit1_sc', num_features=128,
                                                          eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_bn2 = self.__batch_normalization(2, 'stage2_unit1_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_conv2 = self.__conv(2, name='stage2_unit1_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage2_unit1_bn3 = self.__batch_normalization(2, 'stage2_unit1_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_bn1 = self.__batch_normalization(2, 'stage2_unit2_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_conv1 = self.__conv(2, name='stage2_unit2_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit2_bn2 = self.__batch_normalization(2, 'stage2_unit2_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_conv2 = self.__conv(2, name='stage2_unit2_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit2_bn3 = self.__batch_normalization(2, 'stage2_unit2_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit3_bn1 = self.__batch_normalization(2, 'stage2_unit3_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit3_conv1 = self.__conv(2, name='stage2_unit3_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit3_bn2 = self.__batch_normalization(2, 'stage2_unit3_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit3_conv2 = self.__conv(2, name='stage2_unit3_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit3_bn3 = self.__batch_normalization(2, 'stage2_unit3_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit4_bn1 = self.__batch_normalization(2, 'stage2_unit4_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit4_conv1 = self.__conv(2, name='stage2_unit4_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit4_bn2 = self.__batch_normalization(2, 'stage2_unit4_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit4_conv2 = self.__conv(2, name='stage2_unit4_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit4_bn3 = self.__batch_normalization(2, 'stage2_unit4_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit5_bn1 = self.__batch_normalization(2, 'stage2_unit5_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit5_conv1 = self.__conv(2, name='stage2_unit5_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit5_bn2 = self.__batch_normalization(2, 'stage2_unit5_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit5_conv2 = self.__conv(2, name='stage2_unit5_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit5_bn3 = self.__batch_normalization(2, 'stage2_unit5_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit6_bn1 = self.__batch_normalization(2, 'stage2_unit6_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit6_conv1 = self.__conv(2, name='stage2_unit6_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit6_bn2 = self.__batch_normalization(2, 'stage2_unit6_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit6_conv2 = self.__conv(2, name='stage2_unit6_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit6_bn3 = self.__batch_normalization(2, 'stage2_unit6_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit7_bn1 = self.__batch_normalization(2, 'stage2_unit7_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit7_conv1 = self.__conv(2, name='stage2_unit7_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit7_bn2 = self.__batch_normalization(2, 'stage2_unit7_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit7_conv2 = self.__conv(2, name='stage2_unit7_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit7_bn3 = self.__batch_normalization(2, 'stage2_unit7_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit8_bn1 = self.__batch_normalization(2, 'stage2_unit8_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit8_conv1 = self.__conv(2, name='stage2_unit8_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit8_bn2 = self.__batch_normalization(2, 'stage2_unit8_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit8_conv2 = self.__conv(2, name='stage2_unit8_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit8_bn3 = self.__batch_normalization(2, 'stage2_unit8_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit9_bn1 = self.__batch_normalization(2, 'stage2_unit9_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit9_conv1 = self.__conv(2, name='stage2_unit9_conv1', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit9_bn2 = self.__batch_normalization(2, 'stage2_unit9_bn2', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit9_conv2 = self.__conv(2, name='stage2_unit9_conv2', in_channels=128, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit9_bn3 = self.__batch_normalization(2, 'stage2_unit9_bn3', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit10_bn1 = self.__batch_normalization(2, 'stage2_unit10_bn1', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit10_conv1 = self.__conv(2, name='stage2_unit10_conv1', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit10_bn2 = self.__batch_normalization(2, 'stage2_unit10_bn2', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit10_conv2 = self.__conv(2, name='stage2_unit10_conv2', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit10_bn3 = self.__batch_normalization(2, 'stage2_unit10_bn3', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit11_bn1 = self.__batch_normalization(2, 'stage2_unit11_bn1', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit11_conv1 = self.__conv(2, name='stage2_unit11_conv1', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit11_bn2 = self.__batch_normalization(2, 'stage2_unit11_bn2', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit11_conv2 = self.__conv(2, name='stage2_unit11_conv2', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit11_bn3 = self.__batch_normalization(2, 'stage2_unit11_bn3', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit12_bn1 = self.__batch_normalization(2, 'stage2_unit12_bn1', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit12_conv1 = self.__conv(2, name='stage2_unit12_conv1', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit12_bn2 = self.__batch_normalization(2, 'stage2_unit12_bn2', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit12_conv2 = self.__conv(2, name='stage2_unit12_conv2', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit12_bn3 = self.__batch_normalization(2, 'stage2_unit12_bn3', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit13_bn1 = self.__batch_normalization(2, 'stage2_unit13_bn1', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit13_conv1 = self.__conv(2, name='stage2_unit13_conv1', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit13_bn2 = self.__batch_normalization(2, 'stage2_unit13_bn2', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit13_conv2 = self.__conv(2, name='stage2_unit13_conv2', in_channels=128, out_channels=128,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit13_bn3 = self.__batch_normalization(2, 'stage2_unit13_bn3', num_features=128,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_bn1 = self.__batch_normalization(2, 'stage3_unit1_bn1', num_features=128,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_conv1sc = self.__conv(2, name='stage3_unit1_conv1sc', in_channels=128, out_channels=256,
                                                kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage3_unit1_conv1 = self.__conv(2, name='stage3_unit1_conv1', in_channels=128, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit1_sc = self.__batch_normalization(2, 'stage3_unit1_sc', num_features=256,
                                                          eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_bn2 = self.__batch_normalization(2, 'stage3_unit1_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_conv2 = self.__conv(2, name='stage3_unit1_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage3_unit1_bn3 = self.__batch_normalization(2, 'stage3_unit1_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_bn1 = self.__batch_normalization(2, 'stage3_unit2_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_conv1 = self.__conv(2, name='stage3_unit2_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit2_bn2 = self.__batch_normalization(2, 'stage3_unit2_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_conv2 = self.__conv(2, name='stage3_unit2_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit2_bn3 = self.__batch_normalization(2, 'stage3_unit2_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_bn1 = self.__batch_normalization(2, 'stage3_unit3_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_conv1 = self.__conv(2, name='stage3_unit3_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit3_bn2 = self.__batch_normalization(2, 'stage3_unit3_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_conv2 = self.__conv(2, name='stage3_unit3_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit3_bn3 = self.__batch_normalization(2, 'stage3_unit3_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_bn1 = self.__batch_normalization(2, 'stage3_unit4_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_conv1 = self.__conv(2, name='stage3_unit4_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit4_bn2 = self.__batch_normalization(2, 'stage3_unit4_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_conv2 = self.__conv(2, name='stage3_unit4_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit4_bn3 = self.__batch_normalization(2, 'stage3_unit4_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_bn1 = self.__batch_normalization(2, 'stage3_unit5_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_conv1 = self.__conv(2, name='stage3_unit5_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit5_bn2 = self.__batch_normalization(2, 'stage3_unit5_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_conv2 = self.__conv(2, name='stage3_unit5_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit5_bn3 = self.__batch_normalization(2, 'stage3_unit5_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit6_bn1 = self.__batch_normalization(2, 'stage3_unit6_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit6_conv1 = self.__conv(2, name='stage3_unit6_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit6_bn2 = self.__batch_normalization(2, 'stage3_unit6_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit6_conv2 = self.__conv(2, name='stage3_unit6_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit6_bn3 = self.__batch_normalization(2, 'stage3_unit6_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit7_bn1 = self.__batch_normalization(2, 'stage3_unit7_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit7_conv1 = self.__conv(2, name='stage3_unit7_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit7_bn2 = self.__batch_normalization(2, 'stage3_unit7_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit7_conv2 = self.__conv(2, name='stage3_unit7_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit7_bn3 = self.__batch_normalization(2, 'stage3_unit7_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit8_bn1 = self.__batch_normalization(2, 'stage3_unit8_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit8_conv1 = self.__conv(2, name='stage3_unit8_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit8_bn2 = self.__batch_normalization(2, 'stage3_unit8_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit8_conv2 = self.__conv(2, name='stage3_unit8_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit8_bn3 = self.__batch_normalization(2, 'stage3_unit8_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit9_bn1 = self.__batch_normalization(2, 'stage3_unit9_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit9_conv1 = self.__conv(2, name='stage3_unit9_conv1', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit9_bn2 = self.__batch_normalization(2, 'stage3_unit9_bn2', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit9_conv2 = self.__conv(2, name='stage3_unit9_conv2', in_channels=256, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit9_bn3 = self.__batch_normalization(2, 'stage3_unit9_bn3', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit10_bn1 = self.__batch_normalization(2, 'stage3_unit10_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit10_conv1 = self.__conv(2, name='stage3_unit10_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit10_bn2 = self.__batch_normalization(2, 'stage3_unit10_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit10_conv2 = self.__conv(2, name='stage3_unit10_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit10_bn3 = self.__batch_normalization(2, 'stage3_unit10_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit11_bn1 = self.__batch_normalization(2, 'stage3_unit11_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit11_conv1 = self.__conv(2, name='stage3_unit11_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit11_bn2 = self.__batch_normalization(2, 'stage3_unit11_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit11_conv2 = self.__conv(2, name='stage3_unit11_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit11_bn3 = self.__batch_normalization(2, 'stage3_unit11_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit12_bn1 = self.__batch_normalization(2, 'stage3_unit12_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit12_conv1 = self.__conv(2, name='stage3_unit12_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit12_bn2 = self.__batch_normalization(2, 'stage3_unit12_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit12_conv2 = self.__conv(2, name='stage3_unit12_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit12_bn3 = self.__batch_normalization(2, 'stage3_unit12_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit13_bn1 = self.__batch_normalization(2, 'stage3_unit13_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit13_conv1 = self.__conv(2, name='stage3_unit13_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit13_bn2 = self.__batch_normalization(2, 'stage3_unit13_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit13_conv2 = self.__conv(2, name='stage3_unit13_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit13_bn3 = self.__batch_normalization(2, 'stage3_unit13_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit14_bn1 = self.__batch_normalization(2, 'stage3_unit14_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit14_conv1 = self.__conv(2, name='stage3_unit14_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit14_bn2 = self.__batch_normalization(2, 'stage3_unit14_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit14_conv2 = self.__conv(2, name='stage3_unit14_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit14_bn3 = self.__batch_normalization(2, 'stage3_unit14_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit15_bn1 = self.__batch_normalization(2, 'stage3_unit15_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit15_conv1 = self.__conv(2, name='stage3_unit15_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit15_bn2 = self.__batch_normalization(2, 'stage3_unit15_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit15_conv2 = self.__conv(2, name='stage3_unit15_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit15_bn3 = self.__batch_normalization(2, 'stage3_unit15_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit16_bn1 = self.__batch_normalization(2, 'stage3_unit16_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit16_conv1 = self.__conv(2, name='stage3_unit16_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit16_bn2 = self.__batch_normalization(2, 'stage3_unit16_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit16_conv2 = self.__conv(2, name='stage3_unit16_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit16_bn3 = self.__batch_normalization(2, 'stage3_unit16_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit17_bn1 = self.__batch_normalization(2, 'stage3_unit17_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit17_conv1 = self.__conv(2, name='stage3_unit17_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit17_bn2 = self.__batch_normalization(2, 'stage3_unit17_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit17_conv2 = self.__conv(2, name='stage3_unit17_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit17_bn3 = self.__batch_normalization(2, 'stage3_unit17_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit18_bn1 = self.__batch_normalization(2, 'stage3_unit18_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit18_conv1 = self.__conv(2, name='stage3_unit18_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit18_bn2 = self.__batch_normalization(2, 'stage3_unit18_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit18_conv2 = self.__conv(2, name='stage3_unit18_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit18_bn3 = self.__batch_normalization(2, 'stage3_unit18_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit19_bn1 = self.__batch_normalization(2, 'stage3_unit19_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit19_conv1 = self.__conv(2, name='stage3_unit19_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit19_bn2 = self.__batch_normalization(2, 'stage3_unit19_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit19_conv2 = self.__conv(2, name='stage3_unit19_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit19_bn3 = self.__batch_normalization(2, 'stage3_unit19_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit20_bn1 = self.__batch_normalization(2, 'stage3_unit20_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit20_conv1 = self.__conv(2, name='stage3_unit20_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit20_bn2 = self.__batch_normalization(2, 'stage3_unit20_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit20_conv2 = self.__conv(2, name='stage3_unit20_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit20_bn3 = self.__batch_normalization(2, 'stage3_unit20_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit21_bn1 = self.__batch_normalization(2, 'stage3_unit21_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit21_conv1 = self.__conv(2, name='stage3_unit21_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit21_bn2 = self.__batch_normalization(2, 'stage3_unit21_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit21_conv2 = self.__conv(2, name='stage3_unit21_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit21_bn3 = self.__batch_normalization(2, 'stage3_unit21_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit22_bn1 = self.__batch_normalization(2, 'stage3_unit22_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit22_conv1 = self.__conv(2, name='stage3_unit22_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit22_bn2 = self.__batch_normalization(2, 'stage3_unit22_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit22_conv2 = self.__conv(2, name='stage3_unit22_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit22_bn3 = self.__batch_normalization(2, 'stage3_unit22_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit23_bn1 = self.__batch_normalization(2, 'stage3_unit23_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit23_conv1 = self.__conv(2, name='stage3_unit23_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit23_bn2 = self.__batch_normalization(2, 'stage3_unit23_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit23_conv2 = self.__conv(2, name='stage3_unit23_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit23_bn3 = self.__batch_normalization(2, 'stage3_unit23_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit24_bn1 = self.__batch_normalization(2, 'stage3_unit24_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit24_conv1 = self.__conv(2, name='stage3_unit24_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit24_bn2 = self.__batch_normalization(2, 'stage3_unit24_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit24_conv2 = self.__conv(2, name='stage3_unit24_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit24_bn3 = self.__batch_normalization(2, 'stage3_unit24_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit25_bn1 = self.__batch_normalization(2, 'stage3_unit25_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit25_conv1 = self.__conv(2, name='stage3_unit25_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit25_bn2 = self.__batch_normalization(2, 'stage3_unit25_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit25_conv2 = self.__conv(2, name='stage3_unit25_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit25_bn3 = self.__batch_normalization(2, 'stage3_unit25_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit26_bn1 = self.__batch_normalization(2, 'stage3_unit26_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit26_conv1 = self.__conv(2, name='stage3_unit26_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit26_bn2 = self.__batch_normalization(2, 'stage3_unit26_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit26_conv2 = self.__conv(2, name='stage3_unit26_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit26_bn3 = self.__batch_normalization(2, 'stage3_unit26_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit27_bn1 = self.__batch_normalization(2, 'stage3_unit27_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit27_conv1 = self.__conv(2, name='stage3_unit27_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit27_bn2 = self.__batch_normalization(2, 'stage3_unit27_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit27_conv2 = self.__conv(2, name='stage3_unit27_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit27_bn3 = self.__batch_normalization(2, 'stage3_unit27_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit28_bn1 = self.__batch_normalization(2, 'stage3_unit28_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit28_conv1 = self.__conv(2, name='stage3_unit28_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit28_bn2 = self.__batch_normalization(2, 'stage3_unit28_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit28_conv2 = self.__conv(2, name='stage3_unit28_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit28_bn3 = self.__batch_normalization(2, 'stage3_unit28_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit29_bn1 = self.__batch_normalization(2, 'stage3_unit29_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit29_conv1 = self.__conv(2, name='stage3_unit29_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit29_bn2 = self.__batch_normalization(2, 'stage3_unit29_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit29_conv2 = self.__conv(2, name='stage3_unit29_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit29_bn3 = self.__batch_normalization(2, 'stage3_unit29_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit30_bn1 = self.__batch_normalization(2, 'stage3_unit30_bn1', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit30_conv1 = self.__conv(2, name='stage3_unit30_conv1', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit30_bn2 = self.__batch_normalization(2, 'stage3_unit30_bn2', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit30_conv2 = self.__conv(2, name='stage3_unit30_conv2', in_channels=256, out_channels=256,
                                               kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit30_bn3 = self.__batch_normalization(2, 'stage3_unit30_bn3', num_features=256,
                                                            eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_bn1 = self.__batch_normalization(2, 'stage4_unit1_bn1', num_features=256,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_conv1sc = self.__conv(2, name='stage4_unit1_conv1sc', in_channels=256, out_channels=512,
                                                kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage4_unit1_conv1 = self.__conv(2, name='stage4_unit1_conv1', in_channels=256, out_channels=512,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit1_sc = self.__batch_normalization(2, 'stage4_unit1_sc', num_features=512,
                                                          eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_bn2 = self.__batch_normalization(2, 'stage4_unit1_bn2', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_conv2 = self.__conv(2, name='stage4_unit1_conv2', in_channels=512, out_channels=512,
                                              kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage4_unit1_bn3 = self.__batch_normalization(2, 'stage4_unit1_bn3', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_bn1 = self.__batch_normalization(2, 'stage4_unit2_bn1', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_conv1 = self.__conv(2, name='stage4_unit2_conv1', in_channels=512, out_channels=512,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit2_bn2 = self.__batch_normalization(2, 'stage4_unit2_bn2', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_conv2 = self.__conv(2, name='stage4_unit2_conv2', in_channels=512, out_channels=512,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit2_bn3 = self.__batch_normalization(2, 'stage4_unit2_bn3', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit3_bn1 = self.__batch_normalization(2, 'stage4_unit3_bn1', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit3_conv1 = self.__conv(2, name='stage4_unit3_conv1', in_channels=512, out_channels=512,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit3_bn2 = self.__batch_normalization(2, 'stage4_unit3_bn2', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit3_conv2 = self.__conv(2, name='stage4_unit3_conv2', in_channels=512, out_channels=512,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit3_bn3 = self.__batch_normalization(2, 'stage4_unit3_bn3', num_features=512,
                                                           eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7, 7))
        self.bn1 = self.__batch_normalization(2, 'bn1', num_features=512, eps=1.9999999494757503e-05,
                                              momentum=0.8999999761581421)
        self.pre_fc1 = self.__dense(name='pre_fc1', in_features=25088, out_features=512, bias=True)
        self.fc1 = self.__batch_normalization(0, 'fc1', num_features=512, eps=1.9999999494757503e-05,
                                              momentum=0.8999999761581421)

    def forward(self, x):
        lower, upper = 0.5, 1
        mu, sigma = 0.75, 0.7
        # X表示含有最大最小值约束的正态分布  创建
        # [ 3.5 , 6 ] [3.5,6]
        # [3.5,6]的区间内，
        # μ = 5 , σ = 0.7 \mu=5, \sigma=0.7
        # μ=5,σ=0.7的满足正态分布的随机数
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
        # print(X.rvs(1))
        # print("wowowoowowowo")
        self.minusscalar0_second = torch.autograd.Variable(
            torch.cuda.FloatTensor(__weights_dict['minusscalar0_second']['value']), requires_grad=False)
        self.mulscalar0_second = torch.autograd.Variable(
            torch.cuda.FloatTensor(__weights_dict['mulscalar0_second']['value']), requires_grad=False)
        minusscalar0 = x - self.minusscalar0_second
        mulscalar0 = minusscalar0 * self.mulscalar0_second
        conv0_pad = F.pad(mulscalar0, (1, 1, 1, 1))
        conv0 = self.conv0(conv0_pad)
        bn0 = self.bn0(conv0)
        relu0 = F.prelu(bn0, torch.cuda.FloatTensor(__weights_dict['relu0']['weights']))
        stage1_unit1_bn1 = self.stage1_unit1_bn1(relu0)
        stage1_unit1_conv1sc = self.stage1_unit1_conv1sc(relu0)
        stage1_unit1_conv1_pad = F.pad(stage1_unit1_bn1, (1, 1, 1, 1))
        stage1_unit1_conv1 = self.stage1_unit1_conv1(stage1_unit1_conv1_pad)
        stage1_unit1_sc = self.stage1_unit1_sc(stage1_unit1_conv1sc)
        stage1_unit1_bn2 = self.stage1_unit1_bn2(stage1_unit1_conv1)
        stage1_unit1_relu1 = F.prelu(stage1_unit1_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage1_unit1_relu1']['weights']))
        stage1_unit1_conv2_pad = F.pad(stage1_unit1_relu1, (1, 1, 1, 1))
        stage1_unit1_conv2 = self.stage1_unit1_conv2(stage1_unit1_conv2_pad)
        stage1_unit1_bn3 = self.stage1_unit1_bn3(stage1_unit1_conv2)
        if stage1_unit1_bn3.requires_grad:
            stage1_unit1_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus0 = stage1_unit1_bn3 + stage1_unit1_sc  # !!!
        stage1_unit2_bn1 = self.stage1_unit2_bn1(plus0)
        stage1_unit2_conv1_pad = F.pad(stage1_unit2_bn1, (1, 1, 1, 1))
        stage1_unit2_conv1 = self.stage1_unit2_conv1(stage1_unit2_conv1_pad)
        stage1_unit2_bn2 = self.stage1_unit2_bn2(stage1_unit2_conv1)
        stage1_unit2_relu1 = F.prelu(stage1_unit2_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage1_unit2_relu1']['weights']))
        stage1_unit2_conv2_pad = F.pad(stage1_unit2_relu1, (1, 1, 1, 1))
        stage1_unit2_conv2 = self.stage1_unit2_conv2(stage1_unit2_conv2_pad)
        stage1_unit2_bn3 = self.stage1_unit2_bn3(stage1_unit2_conv2)
        if stage1_unit2_bn3.requires_grad:
            stage1_unit2_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus1 = stage1_unit2_bn3 + plus0  # !!!!
        stage1_unit3_bn1 = self.stage1_unit3_bn1(plus1)
        stage1_unit3_conv1_pad = F.pad(stage1_unit3_bn1, (1, 1, 1, 1))
        stage1_unit3_conv1 = self.stage1_unit3_conv1(stage1_unit3_conv1_pad)
        stage1_unit3_bn2 = self.stage1_unit3_bn2(stage1_unit3_conv1)
        stage1_unit3_relu1 = F.prelu(stage1_unit3_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage1_unit3_relu1']['weights']))
        stage1_unit3_conv2_pad = F.pad(stage1_unit3_relu1, (1, 1, 1, 1))
        stage1_unit3_conv2 = self.stage1_unit3_conv2(stage1_unit3_conv2_pad)
        stage1_unit3_bn3 = self.stage1_unit3_bn3(stage1_unit3_conv2)
        if stage1_unit3_bn3.requires_grad:
            stage1_unit3_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus2 = stage1_unit3_bn3 * X.rvs(1)[0] + plus1  # !!!!
        stage2_unit1_bn1 = self.stage2_unit1_bn1(plus2)
        stage2_unit1_conv1sc = self.stage2_unit1_conv1sc(plus2)
        stage2_unit1_conv1_pad = F.pad(stage2_unit1_bn1, (1, 1, 1, 1))
        stage2_unit1_conv1 = self.stage2_unit1_conv1(stage2_unit1_conv1_pad)
        stage2_unit1_sc = self.stage2_unit1_sc(stage2_unit1_conv1sc)
        stage2_unit1_bn2 = self.stage2_unit1_bn2(stage2_unit1_conv1)
        stage2_unit1_relu1 = F.prelu(stage2_unit1_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit1_relu1']['weights']))
        stage2_unit1_conv2_pad = F.pad(stage2_unit1_relu1, (1, 1, 1, 1))
        stage2_unit1_conv2 = self.stage2_unit1_conv2(stage2_unit1_conv2_pad)
        stage2_unit1_bn3 = self.stage2_unit1_bn3(stage2_unit1_conv2)
        if stage2_unit1_bn3.requires_grad:
            stage2_unit1_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus3 = stage2_unit1_bn3 + stage2_unit1_sc  # !!!!
        stage2_unit2_bn1 = self.stage2_unit2_bn1(plus3)
        stage2_unit2_conv1_pad = F.pad(stage2_unit2_bn1, (1, 1, 1, 1))
        stage2_unit2_conv1 = self.stage2_unit2_conv1(stage2_unit2_conv1_pad)
        stage2_unit2_bn2 = self.stage2_unit2_bn2(stage2_unit2_conv1)
        stage2_unit2_relu1 = F.prelu(stage2_unit2_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit2_relu1']['weights']))
        stage2_unit2_conv2_pad = F.pad(stage2_unit2_relu1, (1, 1, 1, 1))
        stage2_unit2_conv2 = self.stage2_unit2_conv2(stage2_unit2_conv2_pad)
        stage2_unit2_bn3 = self.stage2_unit2_bn3(stage2_unit2_conv2)
        if stage2_unit2_bn3.requires_grad:
            stage2_unit2_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus4 = stage2_unit2_bn3 + plus3  # !!!!
        stage2_unit3_bn1 = self.stage2_unit3_bn1(plus4)
        stage2_unit3_conv1_pad = F.pad(stage2_unit3_bn1, (1, 1, 1, 1))
        stage2_unit3_conv1 = self.stage2_unit3_conv1(stage2_unit3_conv1_pad)
        stage2_unit3_bn2 = self.stage2_unit3_bn2(stage2_unit3_conv1)
        stage2_unit3_relu1 = F.prelu(stage2_unit3_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit3_relu1']['weights']))
        stage2_unit3_conv2_pad = F.pad(stage2_unit3_relu1, (1, 1, 1, 1))
        stage2_unit3_conv2 = self.stage2_unit3_conv2(stage2_unit3_conv2_pad)
        stage2_unit3_bn3 = self.stage2_unit3_bn3(stage2_unit3_conv2)
        if stage2_unit3_bn3.requires_grad:
            stage2_unit3_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus5 = stage2_unit3_bn3 + plus4  # !!!!
        stage2_unit4_bn1 = self.stage2_unit4_bn1(plus5)
        stage2_unit4_conv1_pad = F.pad(stage2_unit4_bn1, (1, 1, 1, 1))
        stage2_unit4_conv1 = self.stage2_unit4_conv1(stage2_unit4_conv1_pad)
        stage2_unit4_bn2 = self.stage2_unit4_bn2(stage2_unit4_conv1)
        stage2_unit4_relu1 = F.prelu(stage2_unit4_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit4_relu1']['weights']))
        stage2_unit4_conv2_pad = F.pad(stage2_unit4_relu1, (1, 1, 1, 1))
        stage2_unit4_conv2 = self.stage2_unit4_conv2(stage2_unit4_conv2_pad)
        stage2_unit4_bn3 = self.stage2_unit4_bn3(stage2_unit4_conv2)
        if stage2_unit4_bn3.requires_grad:
            stage2_unit4_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus6 = stage2_unit4_bn3 + plus5  # !!!!
        stage2_unit5_bn1 = self.stage2_unit5_bn1(plus6)
        stage2_unit5_conv1_pad = F.pad(stage2_unit5_bn1, (1, 1, 1, 1))
        stage2_unit5_conv1 = self.stage2_unit5_conv1(stage2_unit5_conv1_pad)
        stage2_unit5_bn2 = self.stage2_unit5_bn2(stage2_unit5_conv1)
        stage2_unit5_relu1 = F.prelu(stage2_unit5_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit5_relu1']['weights']))
        stage2_unit5_conv2_pad = F.pad(stage2_unit5_relu1, (1, 1, 1, 1))
        stage2_unit5_conv2 = self.stage2_unit5_conv2(stage2_unit5_conv2_pad)
        stage2_unit5_bn3 = self.stage2_unit5_bn3(stage2_unit5_conv2)
        if stage2_unit5_bn3.requires_grad:
            stage2_unit5_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus7 = stage2_unit5_bn3 + plus6  # !!!!
        stage2_unit6_bn1 = self.stage2_unit6_bn1(plus7)
        stage2_unit6_conv1_pad = F.pad(stage2_unit6_bn1, (1, 1, 1, 1))
        stage2_unit6_conv1 = self.stage2_unit6_conv1(stage2_unit6_conv1_pad)
        stage2_unit6_bn2 = self.stage2_unit6_bn2(stage2_unit6_conv1)
        stage2_unit6_relu1 = F.prelu(stage2_unit6_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit6_relu1']['weights']))
        stage2_unit6_conv2_pad = F.pad(stage2_unit6_relu1, (1, 1, 1, 1))
        stage2_unit6_conv2 = self.stage2_unit6_conv2(stage2_unit6_conv2_pad)
        stage2_unit6_bn3 = self.stage2_unit6_bn3(stage2_unit6_conv2)
        if stage2_unit6_bn3.requires_grad:
            stage2_unit6_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus8 = stage2_unit6_bn3 + plus7  # !!!!!!
        stage2_unit7_bn1 = self.stage2_unit7_bn1(plus8)
        stage2_unit7_conv1_pad = F.pad(stage2_unit7_bn1, (1, 1, 1, 1))
        stage2_unit7_conv1 = self.stage2_unit7_conv1(stage2_unit7_conv1_pad)
        stage2_unit7_bn2 = self.stage2_unit7_bn2(stage2_unit7_conv1)
        stage2_unit7_relu1 = F.prelu(stage2_unit7_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit7_relu1']['weights']))
        stage2_unit7_conv2_pad = F.pad(stage2_unit7_relu1, (1, 1, 1, 1))
        stage2_unit7_conv2 = self.stage2_unit7_conv2(stage2_unit7_conv2_pad)
        stage2_unit7_bn3 = self.stage2_unit7_bn3(stage2_unit7_conv2)
        if stage2_unit7_bn3.requires_grad:
            stage2_unit7_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus9 = stage2_unit7_bn3 + plus8  # !!!!!
        stage2_unit8_bn1 = self.stage2_unit8_bn1(plus9)
        stage2_unit8_conv1_pad = F.pad(stage2_unit8_bn1, (1, 1, 1, 1))
        stage2_unit8_conv1 = self.stage2_unit8_conv1(stage2_unit8_conv1_pad)
        stage2_unit8_bn2 = self.stage2_unit8_bn2(stage2_unit8_conv1)
        stage2_unit8_relu1 = F.prelu(stage2_unit8_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit8_relu1']['weights']))
        stage2_unit8_conv2_pad = F.pad(stage2_unit8_relu1, (1, 1, 1, 1))
        stage2_unit8_conv2 = self.stage2_unit8_conv2(stage2_unit8_conv2_pad)
        stage2_unit8_bn3 = self.stage2_unit8_bn3(stage2_unit8_conv2)
        if stage2_unit8_bn3.requires_grad:
            stage2_unit8_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus10 = stage2_unit8_bn3 + plus9  # !!!!!
        stage2_unit9_bn1 = self.stage2_unit9_bn1(plus10)
        stage2_unit9_conv1_pad = F.pad(stage2_unit9_bn1, (1, 1, 1, 1))
        stage2_unit9_conv1 = self.stage2_unit9_conv1(stage2_unit9_conv1_pad)
        stage2_unit9_bn2 = self.stage2_unit9_bn2(stage2_unit9_conv1)
        stage2_unit9_relu1 = F.prelu(stage2_unit9_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage2_unit9_relu1']['weights']))
        stage2_unit9_conv2_pad = F.pad(stage2_unit9_relu1, (1, 1, 1, 1))
        stage2_unit9_conv2 = self.stage2_unit9_conv2(stage2_unit9_conv2_pad)
        stage2_unit9_bn3 = self.stage2_unit9_bn3(stage2_unit9_conv2)
        if stage2_unit9_bn3.requires_grad:
            stage2_unit9_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus11 = stage2_unit9_bn3 + plus10  # !!!!
        stage2_unit10_bn1 = self.stage2_unit10_bn1(plus11)
        stage2_unit10_conv1_pad = F.pad(stage2_unit10_bn1, (1, 1, 1, 1))
        stage2_unit10_conv1 = self.stage2_unit10_conv1(stage2_unit10_conv1_pad)
        stage2_unit10_bn2 = self.stage2_unit10_bn2(stage2_unit10_conv1)
        stage2_unit10_relu1 = F.prelu(stage2_unit10_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage2_unit10_relu1']['weights']))
        stage2_unit10_conv2_pad = F.pad(stage2_unit10_relu1, (1, 1, 1, 1))
        stage2_unit10_conv2 = self.stage2_unit10_conv2(stage2_unit10_conv2_pad)
        stage2_unit10_bn3 = self.stage2_unit10_bn3(stage2_unit10_conv2)
        if stage2_unit10_bn3.requires_grad:
            stage2_unit10_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus12 = stage2_unit10_bn3 + plus11  # !!!!!
        stage2_unit11_bn1 = self.stage2_unit11_bn1(plus12)
        stage2_unit11_conv1_pad = F.pad(stage2_unit11_bn1, (1, 1, 1, 1))
        stage2_unit11_conv1 = self.stage2_unit11_conv1(stage2_unit11_conv1_pad)
        stage2_unit11_bn2 = self.stage2_unit11_bn2(stage2_unit11_conv1)
        stage2_unit11_relu1 = F.prelu(stage2_unit11_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage2_unit11_relu1']['weights']))
        stage2_unit11_conv2_pad = F.pad(stage2_unit11_relu1, (1, 1, 1, 1))
        stage2_unit11_conv2 = self.stage2_unit11_conv2(stage2_unit11_conv2_pad)
        stage2_unit11_bn3 = self.stage2_unit11_bn3(stage2_unit11_conv2)
        if stage2_unit11_bn3.requires_grad:
            stage2_unit11_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus13 = stage2_unit11_bn3 + plus12  # !!!!!
        stage2_unit12_bn1 = self.stage2_unit12_bn1(plus13)
        stage2_unit12_conv1_pad = F.pad(stage2_unit12_bn1, (1, 1, 1, 1))
        stage2_unit12_conv1 = self.stage2_unit12_conv1(stage2_unit12_conv1_pad)
        stage2_unit12_bn2 = self.stage2_unit12_bn2(stage2_unit12_conv1)
        stage2_unit12_relu1 = F.prelu(stage2_unit12_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage2_unit12_relu1']['weights']))
        stage2_unit12_conv2_pad = F.pad(stage2_unit12_relu1, (1, 1, 1, 1))
        stage2_unit12_conv2 = self.stage2_unit12_conv2(stage2_unit12_conv2_pad)
        stage2_unit12_bn3 = self.stage2_unit12_bn3(stage2_unit12_conv2)
        if stage2_unit12_bn3.requires_grad:
            stage2_unit12_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus14 = stage2_unit12_bn3 + plus13  # !!!!
        stage2_unit13_bn1 = self.stage2_unit13_bn1(plus14)
        stage2_unit13_conv1_pad = F.pad(stage2_unit13_bn1, (1, 1, 1, 1))
        stage2_unit13_conv1 = self.stage2_unit13_conv1(stage2_unit13_conv1_pad)
        stage2_unit13_bn2 = self.stage2_unit13_bn2(stage2_unit13_conv1)
        stage2_unit13_relu1 = F.prelu(stage2_unit13_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage2_unit13_relu1']['weights']))
        stage2_unit13_conv2_pad = F.pad(stage2_unit13_relu1, (1, 1, 1, 1))
        stage2_unit13_conv2 = self.stage2_unit13_conv2(stage2_unit13_conv2_pad)
        stage2_unit13_bn3 = self.stage2_unit13_bn3(stage2_unit13_conv2)
        if stage2_unit13_bn3.requires_grad:
            stage2_unit13_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus15 = stage2_unit13_bn3 + plus14  # !!!!!
        stage3_unit1_bn1 = self.stage3_unit1_bn1(plus15)
        stage3_unit1_conv1sc = self.stage3_unit1_conv1sc(plus15)
        stage3_unit1_conv1_pad = F.pad(stage3_unit1_bn1, (1, 1, 1, 1))
        stage3_unit1_conv1 = self.stage3_unit1_conv1(stage3_unit1_conv1_pad)
        stage3_unit1_sc = self.stage3_unit1_sc(stage3_unit1_conv1sc)
        stage3_unit1_bn2 = self.stage3_unit1_bn2(stage3_unit1_conv1)
        stage3_unit1_relu1 = F.prelu(stage3_unit1_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit1_relu1']['weights']))
        stage3_unit1_conv2_pad = F.pad(stage3_unit1_relu1, (1, 1, 1, 1))
        stage3_unit1_conv2 = self.stage3_unit1_conv2(stage3_unit1_conv2_pad)
        stage3_unit1_bn3 = self.stage3_unit1_bn3(stage3_unit1_conv2)
        if stage3_unit1_bn3.requires_grad:
            stage3_unit1_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus16 = stage3_unit1_bn3 + stage3_unit1_sc  # !!!!
        stage3_unit2_bn1 = self.stage3_unit2_bn1(plus16)
        stage3_unit2_conv1_pad = F.pad(stage3_unit2_bn1, (1, 1, 1, 1))
        stage3_unit2_conv1 = self.stage3_unit2_conv1(stage3_unit2_conv1_pad)
        stage3_unit2_bn2 = self.stage3_unit2_bn2(stage3_unit2_conv1)
        stage3_unit2_relu1 = F.prelu(stage3_unit2_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit2_relu1']['weights']))
        stage3_unit2_conv2_pad = F.pad(stage3_unit2_relu1, (1, 1, 1, 1))
        stage3_unit2_conv2 = self.stage3_unit2_conv2(stage3_unit2_conv2_pad)
        stage3_unit2_bn3 = self.stage3_unit2_bn3(stage3_unit2_conv2)
        if stage3_unit2_bn3.requires_grad:
            stage3_unit2_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus17 = stage3_unit2_bn3 + plus16  # !!!!!
        stage3_unit3_bn1 = self.stage3_unit3_bn1(plus17)
        stage3_unit3_conv1_pad = F.pad(stage3_unit3_bn1, (1, 1, 1, 1))
        stage3_unit3_conv1 = self.stage3_unit3_conv1(stage3_unit3_conv1_pad)
        stage3_unit3_bn2 = self.stage3_unit3_bn2(stage3_unit3_conv1)
        stage3_unit3_relu1 = F.prelu(stage3_unit3_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit3_relu1']['weights']))
        stage3_unit3_conv2_pad = F.pad(stage3_unit3_relu1, (1, 1, 1, 1))
        stage3_unit3_conv2 = self.stage3_unit3_conv2(stage3_unit3_conv2_pad)
        stage3_unit3_bn3 = self.stage3_unit3_bn3(stage3_unit3_conv2)
        if stage3_unit3_bn3.requires_grad:
            stage3_unit3_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus18 = stage3_unit3_bn3 + plus17  # !!!!
        stage3_unit4_bn1 = self.stage3_unit4_bn1(plus18)
        stage3_unit4_conv1_pad = F.pad(stage3_unit4_bn1, (1, 1, 1, 1))
        stage3_unit4_conv1 = self.stage3_unit4_conv1(stage3_unit4_conv1_pad)
        stage3_unit4_bn2 = self.stage3_unit4_bn2(stage3_unit4_conv1)
        stage3_unit4_relu1 = F.prelu(stage3_unit4_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit4_relu1']['weights']))
        stage3_unit4_conv2_pad = F.pad(stage3_unit4_relu1, (1, 1, 1, 1))
        stage3_unit4_conv2 = self.stage3_unit4_conv2(stage3_unit4_conv2_pad)
        stage3_unit4_bn3 = self.stage3_unit4_bn3(stage3_unit4_conv2)
        if stage3_unit4_bn3.requires_grad:
            stage3_unit4_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus19 = stage3_unit4_bn3 + plus18  # !!!!!!
        stage3_unit5_bn1 = self.stage3_unit5_bn1(plus19)
        stage3_unit5_conv1_pad = F.pad(stage3_unit5_bn1, (1, 1, 1, 1))
        stage3_unit5_conv1 = self.stage3_unit5_conv1(stage3_unit5_conv1_pad)
        stage3_unit5_bn2 = self.stage3_unit5_bn2(stage3_unit5_conv1)
        stage3_unit5_relu1 = F.prelu(stage3_unit5_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit5_relu1']['weights']))
        stage3_unit5_conv2_pad = F.pad(stage3_unit5_relu1, (1, 1, 1, 1))
        stage3_unit5_conv2 = self.stage3_unit5_conv2(stage3_unit5_conv2_pad)
        stage3_unit5_bn3 = self.stage3_unit5_bn3(stage3_unit5_conv2)
        if stage3_unit5_bn3.requires_grad:
            stage3_unit5_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus20 = stage3_unit5_bn3 + plus19  # !!!!!
        stage3_unit6_bn1 = self.stage3_unit6_bn1(plus20)
        stage3_unit6_conv1_pad = F.pad(stage3_unit6_bn1, (1, 1, 1, 1))
        stage3_unit6_conv1 = self.stage3_unit6_conv1(stage3_unit6_conv1_pad)
        stage3_unit6_bn2 = self.stage3_unit6_bn2(stage3_unit6_conv1)
        stage3_unit6_relu1 = F.prelu(stage3_unit6_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit6_relu1']['weights']))
        stage3_unit6_conv2_pad = F.pad(stage3_unit6_relu1, (1, 1, 1, 1))
        stage3_unit6_conv2 = self.stage3_unit6_conv2(stage3_unit6_conv2_pad)
        stage3_unit6_bn3 = self.stage3_unit6_bn3(stage3_unit6_conv2)
        if stage3_unit6_bn3.requires_grad:
            stage3_unit6_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus21 = stage3_unit6_bn3 + plus20  # !!!!!
        stage3_unit7_bn1 = self.stage3_unit7_bn1(plus21)
        stage3_unit7_conv1_pad = F.pad(stage3_unit7_bn1, (1, 1, 1, 1))
        stage3_unit7_conv1 = self.stage3_unit7_conv1(stage3_unit7_conv1_pad)
        stage3_unit7_bn2 = self.stage3_unit7_bn2(stage3_unit7_conv1)
        stage3_unit7_relu1 = F.prelu(stage3_unit7_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit7_relu1']['weights']))
        stage3_unit7_conv2_pad = F.pad(stage3_unit7_relu1, (1, 1, 1, 1))
        stage3_unit7_conv2 = self.stage3_unit7_conv2(stage3_unit7_conv2_pad)
        stage3_unit7_bn3 = self.stage3_unit7_bn3(stage3_unit7_conv2)
        if stage3_unit7_bn3.requires_grad:
            stage3_unit7_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus22 = stage3_unit7_bn3 + plus21  # !!!!!
        stage3_unit8_bn1 = self.stage3_unit8_bn1(plus22)
        stage3_unit8_conv1_pad = F.pad(stage3_unit8_bn1, (1, 1, 1, 1))
        stage3_unit8_conv1 = self.stage3_unit8_conv1(stage3_unit8_conv1_pad)
        stage3_unit8_bn2 = self.stage3_unit8_bn2(stage3_unit8_conv1)
        stage3_unit8_relu1 = F.prelu(stage3_unit8_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit8_relu1']['weights']))
        stage3_unit8_conv2_pad = F.pad(stage3_unit8_relu1, (1, 1, 1, 1))
        stage3_unit8_conv2 = self.stage3_unit8_conv2(stage3_unit8_conv2_pad)
        stage3_unit8_bn3 = self.stage3_unit8_bn3(stage3_unit8_conv2)
        if stage3_unit8_bn3.requires_grad:
            stage3_unit8_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus23 = stage3_unit8_bn3 + plus22  # !!!!!!
        stage3_unit9_bn1 = self.stage3_unit9_bn1(plus23)
        stage3_unit9_conv1_pad = F.pad(stage3_unit9_bn1, (1, 1, 1, 1))
        stage3_unit9_conv1 = self.stage3_unit9_conv1(stage3_unit9_conv1_pad)
        stage3_unit9_bn2 = self.stage3_unit9_bn2(stage3_unit9_conv1)
        stage3_unit9_relu1 = F.prelu(stage3_unit9_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage3_unit9_relu1']['weights']))
        stage3_unit9_conv2_pad = F.pad(stage3_unit9_relu1, (1, 1, 1, 1))
        stage3_unit9_conv2 = self.stage3_unit9_conv2(stage3_unit9_conv2_pad)
        stage3_unit9_bn3 = self.stage3_unit9_bn3(stage3_unit9_conv2)
        if stage3_unit9_bn3.requires_grad:
            stage3_unit9_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus24 = stage3_unit9_bn3 + plus23  # !!!!!
        stage3_unit10_bn1 = self.stage3_unit10_bn1(plus24)
        stage3_unit10_conv1_pad = F.pad(stage3_unit10_bn1, (1, 1, 1, 1))
        stage3_unit10_conv1 = self.stage3_unit10_conv1(stage3_unit10_conv1_pad)
        stage3_unit10_bn2 = self.stage3_unit10_bn2(stage3_unit10_conv1)
        stage3_unit10_relu1 = F.prelu(stage3_unit10_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit10_relu1']['weights']))
        stage3_unit10_conv2_pad = F.pad(stage3_unit10_relu1, (1, 1, 1, 1))
        stage3_unit10_conv2 = self.stage3_unit10_conv2(stage3_unit10_conv2_pad)
        stage3_unit10_bn3 = self.stage3_unit10_bn3(stage3_unit10_conv2)
        if stage3_unit10_bn3.requires_grad:
            stage3_unit10_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus25 = stage3_unit10_bn3 + plus24  # !!!!
        stage3_unit11_bn1 = self.stage3_unit11_bn1(plus25)
        stage3_unit11_conv1_pad = F.pad(stage3_unit11_bn1, (1, 1, 1, 1))
        stage3_unit11_conv1 = self.stage3_unit11_conv1(stage3_unit11_conv1_pad)
        stage3_unit11_bn2 = self.stage3_unit11_bn2(stage3_unit11_conv1)
        stage3_unit11_relu1 = F.prelu(stage3_unit11_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit11_relu1']['weights']))
        stage3_unit11_conv2_pad = F.pad(stage3_unit11_relu1, (1, 1, 1, 1))
        stage3_unit11_conv2 = self.stage3_unit11_conv2(stage3_unit11_conv2_pad)
        stage3_unit11_bn3 = self.stage3_unit11_bn3(stage3_unit11_conv2)
        if stage3_unit11_bn3.requires_grad:
            stage3_unit11_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus26 = stage3_unit11_bn3 + plus25  # !!!!!
        stage3_unit12_bn1 = self.stage3_unit12_bn1(plus26)
        stage3_unit12_conv1_pad = F.pad(stage3_unit12_bn1, (1, 1, 1, 1))
        stage3_unit12_conv1 = self.stage3_unit12_conv1(stage3_unit12_conv1_pad)
        stage3_unit12_bn2 = self.stage3_unit12_bn2(stage3_unit12_conv1)
        stage3_unit12_relu1 = F.prelu(stage3_unit12_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit12_relu1']['weights']))
        stage3_unit12_conv2_pad = F.pad(stage3_unit12_relu1, (1, 1, 1, 1))
        stage3_unit12_conv2 = self.stage3_unit12_conv2(stage3_unit12_conv2_pad)
        stage3_unit12_bn3 = self.stage3_unit12_bn3(stage3_unit12_conv2)
        if stage3_unit12_bn3.requires_grad:
            stage3_unit12_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus27 = stage3_unit12_bn3 + plus26  # !!!!!
        stage3_unit13_bn1 = self.stage3_unit13_bn1(plus27)
        stage3_unit13_conv1_pad = F.pad(stage3_unit13_bn1, (1, 1, 1, 1))
        stage3_unit13_conv1 = self.stage3_unit13_conv1(stage3_unit13_conv1_pad)
        stage3_unit13_bn2 = self.stage3_unit13_bn2(stage3_unit13_conv1)
        stage3_unit13_relu1 = F.prelu(stage3_unit13_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit13_relu1']['weights']))
        stage3_unit13_conv2_pad = F.pad(stage3_unit13_relu1, (1, 1, 1, 1))
        stage3_unit13_conv2 = self.stage3_unit13_conv2(stage3_unit13_conv2_pad)
        stage3_unit13_bn3 = self.stage3_unit13_bn3(stage3_unit13_conv2)
        if stage3_unit13_bn3.requires_grad:
            stage3_unit13_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus28 = stage3_unit13_bn3 + plus27  # !!!!!
        stage3_unit14_bn1 = self.stage3_unit14_bn1(plus28)
        stage3_unit14_conv1_pad = F.pad(stage3_unit14_bn1, (1, 1, 1, 1))
        stage3_unit14_conv1 = self.stage3_unit14_conv1(stage3_unit14_conv1_pad)
        stage3_unit14_bn2 = self.stage3_unit14_bn2(stage3_unit14_conv1)
        stage3_unit14_relu1 = F.prelu(stage3_unit14_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit14_relu1']['weights']))
        stage3_unit14_conv2_pad = F.pad(stage3_unit14_relu1, (1, 1, 1, 1))
        stage3_unit14_conv2 = self.stage3_unit14_conv2(stage3_unit14_conv2_pad)
        stage3_unit14_bn3 = self.stage3_unit14_bn3(stage3_unit14_conv2)
        if stage3_unit14_bn3.requires_grad:
            stage3_unit14_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus29 = stage3_unit14_bn3 + plus28  # !!!!!
        stage3_unit15_bn1 = self.stage3_unit15_bn1(plus29)
        stage3_unit15_conv1_pad = F.pad(stage3_unit15_bn1, (1, 1, 1, 1))
        stage3_unit15_conv1 = self.stage3_unit15_conv1(stage3_unit15_conv1_pad)
        stage3_unit15_bn2 = self.stage3_unit15_bn2(stage3_unit15_conv1)
        stage3_unit15_relu1 = F.prelu(stage3_unit15_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit15_relu1']['weights']))
        stage3_unit15_conv2_pad = F.pad(stage3_unit15_relu1, (1, 1, 1, 1))
        stage3_unit15_conv2 = self.stage3_unit15_conv2(stage3_unit15_conv2_pad)
        stage3_unit15_bn3 = self.stage3_unit15_bn3(stage3_unit15_conv2)
        if stage3_unit15_bn3.requires_grad:
            stage3_unit15_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus30 = stage3_unit15_bn3 + plus29  # !!!!
        stage3_unit16_bn1 = self.stage3_unit16_bn1(plus30)
        stage3_unit16_conv1_pad = F.pad(stage3_unit16_bn1, (1, 1, 1, 1))
        stage3_unit16_conv1 = self.stage3_unit16_conv1(stage3_unit16_conv1_pad)
        stage3_unit16_bn2 = self.stage3_unit16_bn2(stage3_unit16_conv1)
        stage3_unit16_relu1 = F.prelu(stage3_unit16_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit16_relu1']['weights']))
        stage3_unit16_conv2_pad = F.pad(stage3_unit16_relu1, (1, 1, 1, 1))
        stage3_unit16_conv2 = self.stage3_unit16_conv2(stage3_unit16_conv2_pad)
        stage3_unit16_bn3 = self.stage3_unit16_bn3(stage3_unit16_conv2)
        if stage3_unit16_bn3.requires_grad:
            stage3_unit16_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus31 = stage3_unit16_bn3 + plus30  # !!!!!!
        stage3_unit17_bn1 = self.stage3_unit17_bn1(plus31)
        stage3_unit17_conv1_pad = F.pad(stage3_unit17_bn1, (1, 1, 1, 1))
        stage3_unit17_conv1 = self.stage3_unit17_conv1(stage3_unit17_conv1_pad)
        stage3_unit17_bn2 = self.stage3_unit17_bn2(stage3_unit17_conv1)
        stage3_unit17_relu1 = F.prelu(stage3_unit17_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit17_relu1']['weights']))
        stage3_unit17_conv2_pad = F.pad(stage3_unit17_relu1, (1, 1, 1, 1))
        stage3_unit17_conv2 = self.stage3_unit17_conv2(stage3_unit17_conv2_pad)
        stage3_unit17_bn3 = self.stage3_unit17_bn3(stage3_unit17_conv2)
        if stage3_unit17_bn3.requires_grad:
            stage3_unit17_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus32 = stage3_unit17_bn3 + plus31  # !!!!!!
        stage3_unit18_bn1 = self.stage3_unit18_bn1(plus32)
        stage3_unit18_conv1_pad = F.pad(stage3_unit18_bn1, (1, 1, 1, 1))
        stage3_unit18_conv1 = self.stage3_unit18_conv1(stage3_unit18_conv1_pad)
        stage3_unit18_bn2 = self.stage3_unit18_bn2(stage3_unit18_conv1)
        stage3_unit18_relu1 = F.prelu(stage3_unit18_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit18_relu1']['weights']))
        stage3_unit18_conv2_pad = F.pad(stage3_unit18_relu1, (1, 1, 1, 1))
        stage3_unit18_conv2 = self.stage3_unit18_conv2(stage3_unit18_conv2_pad)
        stage3_unit18_bn3 = self.stage3_unit18_bn3(stage3_unit18_conv2)
        if stage3_unit18_bn3.requires_grad:
            stage3_unit18_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus33 = stage3_unit18_bn3 + plus32  # !!!!!
        stage3_unit19_bn1 = self.stage3_unit19_bn1(plus33)
        stage3_unit19_conv1_pad = F.pad(stage3_unit19_bn1, (1, 1, 1, 1))
        stage3_unit19_conv1 = self.stage3_unit19_conv1(stage3_unit19_conv1_pad)
        stage3_unit19_bn2 = self.stage3_unit19_bn2(stage3_unit19_conv1)
        stage3_unit19_relu1 = F.prelu(stage3_unit19_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit19_relu1']['weights']))
        stage3_unit19_conv2_pad = F.pad(stage3_unit19_relu1, (1, 1, 1, 1))
        stage3_unit19_conv2 = self.stage3_unit19_conv2(stage3_unit19_conv2_pad)
        stage3_unit19_bn3 = self.stage3_unit19_bn3(stage3_unit19_conv2)
        if stage3_unit19_bn3.requires_grad:
            stage3_unit19_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus34 = stage3_unit19_bn3 + plus33  # !!!!
        stage3_unit20_bn1 = self.stage3_unit20_bn1(plus34)
        stage3_unit20_conv1_pad = F.pad(stage3_unit20_bn1, (1, 1, 1, 1))
        stage3_unit20_conv1 = self.stage3_unit20_conv1(stage3_unit20_conv1_pad)
        stage3_unit20_bn2 = self.stage3_unit20_bn2(stage3_unit20_conv1)
        stage3_unit20_relu1 = F.prelu(stage3_unit20_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit20_relu1']['weights']))
        stage3_unit20_conv2_pad = F.pad(stage3_unit20_relu1, (1, 1, 1, 1))
        stage3_unit20_conv2 = self.stage3_unit20_conv2(stage3_unit20_conv2_pad)
        stage3_unit20_bn3 = self.stage3_unit20_bn3(stage3_unit20_conv2)
        if stage3_unit20_bn3.requires_grad:
            stage3_unit20_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus35 = stage3_unit20_bn3 + plus34  # !!!!
        stage3_unit21_bn1 = self.stage3_unit21_bn1(plus35)
        stage3_unit21_conv1_pad = F.pad(stage3_unit21_bn1, (1, 1, 1, 1))
        stage3_unit21_conv1 = self.stage3_unit21_conv1(stage3_unit21_conv1_pad)
        stage3_unit21_bn2 = self.stage3_unit21_bn2(stage3_unit21_conv1)
        stage3_unit21_relu1 = F.prelu(stage3_unit21_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit21_relu1']['weights']))
        stage3_unit21_conv2_pad = F.pad(stage3_unit21_relu1, (1, 1, 1, 1))
        stage3_unit21_conv2 = self.stage3_unit21_conv2(stage3_unit21_conv2_pad)
        stage3_unit21_bn3 = self.stage3_unit21_bn3(stage3_unit21_conv2)
        if stage3_unit21_bn3.requires_grad:
            stage3_unit21_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus36 = stage3_unit21_bn3 + plus35  # !!!!
        stage3_unit22_bn1 = self.stage3_unit22_bn1(plus36)
        stage3_unit22_conv1_pad = F.pad(stage3_unit22_bn1, (1, 1, 1, 1))
        stage3_unit22_conv1 = self.stage3_unit22_conv1(stage3_unit22_conv1_pad)
        stage3_unit22_bn2 = self.stage3_unit22_bn2(stage3_unit22_conv1)
        stage3_unit22_relu1 = F.prelu(stage3_unit22_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit22_relu1']['weights']))
        stage3_unit22_conv2_pad = F.pad(stage3_unit22_relu1, (1, 1, 1, 1))
        stage3_unit22_conv2 = self.stage3_unit22_conv2(stage3_unit22_conv2_pad)
        stage3_unit22_bn3 = self.stage3_unit22_bn3(stage3_unit22_conv2)
        if stage3_unit22_bn3.requires_grad:
            stage3_unit22_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus37 = stage3_unit22_bn3 + plus36  # !!!!
        stage3_unit23_bn1 = self.stage3_unit23_bn1(plus37)
        stage3_unit23_conv1_pad = F.pad(stage3_unit23_bn1, (1, 1, 1, 1))
        stage3_unit23_conv1 = self.stage3_unit23_conv1(stage3_unit23_conv1_pad)
        stage3_unit23_bn2 = self.stage3_unit23_bn2(stage3_unit23_conv1)
        stage3_unit23_relu1 = F.prelu(stage3_unit23_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit23_relu1']['weights']))
        stage3_unit23_conv2_pad = F.pad(stage3_unit23_relu1, (1, 1, 1, 1))
        stage3_unit23_conv2 = self.stage3_unit23_conv2(stage3_unit23_conv2_pad)
        stage3_unit23_bn3 = self.stage3_unit23_bn3(stage3_unit23_conv2)
        if stage3_unit23_bn3.requires_grad:
            stage3_unit23_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus38 = stage3_unit23_bn3 + plus37  # !!!!
        stage3_unit24_bn1 = self.stage3_unit24_bn1(plus38)
        stage3_unit24_conv1_pad = F.pad(stage3_unit24_bn1, (1, 1, 1, 1))
        stage3_unit24_conv1 = self.stage3_unit24_conv1(stage3_unit24_conv1_pad)
        stage3_unit24_bn2 = self.stage3_unit24_bn2(stage3_unit24_conv1)
        stage3_unit24_relu1 = F.prelu(stage3_unit24_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit24_relu1']['weights']))
        stage3_unit24_conv2_pad = F.pad(stage3_unit24_relu1, (1, 1, 1, 1))
        stage3_unit24_conv2 = self.stage3_unit24_conv2(stage3_unit24_conv2_pad)
        stage3_unit24_bn3 = self.stage3_unit24_bn3(stage3_unit24_conv2)
        if stage3_unit24_bn3.requires_grad:
            stage3_unit24_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus39 = stage3_unit24_bn3 + plus38  # !!!!
        stage3_unit25_bn1 = self.stage3_unit25_bn1(plus39)
        stage3_unit25_conv1_pad = F.pad(stage3_unit25_bn1, (1, 1, 1, 1))
        stage3_unit25_conv1 = self.stage3_unit25_conv1(stage3_unit25_conv1_pad)
        stage3_unit25_bn2 = self.stage3_unit25_bn2(stage3_unit25_conv1)
        stage3_unit25_relu1 = F.prelu(stage3_unit25_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit25_relu1']['weights']))
        stage3_unit25_conv2_pad = F.pad(stage3_unit25_relu1, (1, 1, 1, 1))
        stage3_unit25_conv2 = self.stage3_unit25_conv2(stage3_unit25_conv2_pad)
        stage3_unit25_bn3 = self.stage3_unit25_bn3(stage3_unit25_conv2)
        if stage3_unit25_bn3.requires_grad:
            stage3_unit25_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus40 = stage3_unit25_bn3 + plus39  # !!!!
        stage3_unit26_bn1 = self.stage3_unit26_bn1(plus40)
        stage3_unit26_conv1_pad = F.pad(stage3_unit26_bn1, (1, 1, 1, 1))
        stage3_unit26_conv1 = self.stage3_unit26_conv1(stage3_unit26_conv1_pad)
        stage3_unit26_bn2 = self.stage3_unit26_bn2(stage3_unit26_conv1)
        stage3_unit26_relu1 = F.prelu(stage3_unit26_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit26_relu1']['weights']))
        stage3_unit26_conv2_pad = F.pad(stage3_unit26_relu1, (1, 1, 1, 1))
        stage3_unit26_conv2 = self.stage3_unit26_conv2(stage3_unit26_conv2_pad)
        stage3_unit26_bn3 = self.stage3_unit26_bn3(stage3_unit26_conv2)
        if stage3_unit26_bn3.requires_grad:
            stage3_unit26_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus41 = stage3_unit26_bn3 + plus40  # !!!!!
        stage3_unit27_bn1 = self.stage3_unit27_bn1(plus41)
        stage3_unit27_conv1_pad = F.pad(stage3_unit27_bn1, (1, 1, 1, 1))
        stage3_unit27_conv1 = self.stage3_unit27_conv1(stage3_unit27_conv1_pad)
        stage3_unit27_bn2 = self.stage3_unit27_bn2(stage3_unit27_conv1)
        stage3_unit27_relu1 = F.prelu(stage3_unit27_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit27_relu1']['weights']))
        stage3_unit27_conv2_pad = F.pad(stage3_unit27_relu1, (1, 1, 1, 1))
        stage3_unit27_conv2 = self.stage3_unit27_conv2(stage3_unit27_conv2_pad)
        stage3_unit27_bn3 = self.stage3_unit27_bn3(stage3_unit27_conv2)
        if stage3_unit27_bn3.requires_grad:
            stage3_unit27_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus42 = stage3_unit27_bn3 + plus41  # !!!!!
        stage3_unit28_bn1 = self.stage3_unit28_bn1(plus42)
        stage3_unit28_conv1_pad = F.pad(stage3_unit28_bn1, (1, 1, 1, 1))
        stage3_unit28_conv1 = self.stage3_unit28_conv1(stage3_unit28_conv1_pad)
        stage3_unit28_bn2 = self.stage3_unit28_bn2(stage3_unit28_conv1)
        stage3_unit28_relu1 = F.prelu(stage3_unit28_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit28_relu1']['weights']))
        stage3_unit28_conv2_pad = F.pad(stage3_unit28_relu1, (1, 1, 1, 1))
        stage3_unit28_conv2 = self.stage3_unit28_conv2(stage3_unit28_conv2_pad)
        stage3_unit28_bn3 = self.stage3_unit28_bn3(stage3_unit28_conv2)
        if stage3_unit28_bn3.requires_grad:
            stage3_unit28_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus43 = stage3_unit28_bn3 + plus42  # !!!!!
        stage3_unit29_bn1 = self.stage3_unit29_bn1(plus43)
        stage3_unit29_conv1_pad = F.pad(stage3_unit29_bn1, (1, 1, 1, 1))
        stage3_unit29_conv1 = self.stage3_unit29_conv1(stage3_unit29_conv1_pad)
        stage3_unit29_bn2 = self.stage3_unit29_bn2(stage3_unit29_conv1)
        stage3_unit29_relu1 = F.prelu(stage3_unit29_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit29_relu1']['weights']))
        stage3_unit29_conv2_pad = F.pad(stage3_unit29_relu1, (1, 1, 1, 1))
        stage3_unit29_conv2 = self.stage3_unit29_conv2(stage3_unit29_conv2_pad)
        stage3_unit29_bn3 = self.stage3_unit29_bn3(stage3_unit29_conv2)
        if stage3_unit29_bn3.requires_grad:
            stage3_unit29_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus44 = stage3_unit29_bn3 + plus43  # !!!!!
        stage3_unit30_bn1 = self.stage3_unit30_bn1(plus44)
        stage3_unit30_conv1_pad = F.pad(stage3_unit30_bn1, (1, 1, 1, 1))
        stage3_unit30_conv1 = self.stage3_unit30_conv1(stage3_unit30_conv1_pad)
        stage3_unit30_bn2 = self.stage3_unit30_bn2(stage3_unit30_conv1)
        stage3_unit30_relu1 = F.prelu(stage3_unit30_bn2,
                                      torch.cuda.FloatTensor(__weights_dict['stage3_unit30_relu1']['weights']))
        stage3_unit30_conv2_pad = F.pad(stage3_unit30_relu1, (1, 1, 1, 1))
        stage3_unit30_conv2 = self.stage3_unit30_conv2(stage3_unit30_conv2_pad)
        stage3_unit30_bn3 = self.stage3_unit30_bn3(stage3_unit30_conv2)
        if stage3_unit30_bn3.requires_grad:
            stage3_unit30_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus45 = stage3_unit30_bn3 + plus44  # !!!!!
        stage4_unit1_bn1 = self.stage4_unit1_bn1(plus45)
        stage4_unit1_conv1sc = self.stage4_unit1_conv1sc(plus45)
        stage4_unit1_conv1_pad = F.pad(stage4_unit1_bn1, (1, 1, 1, 1))
        stage4_unit1_conv1 = self.stage4_unit1_conv1(stage4_unit1_conv1_pad)
        stage4_unit1_sc = self.stage4_unit1_sc(stage4_unit1_conv1sc)
        stage4_unit1_bn2 = self.stage4_unit1_bn2(stage4_unit1_conv1)
        stage4_unit1_relu1 = F.prelu(stage4_unit1_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage4_unit1_relu1']['weights']))
        stage4_unit1_conv2_pad = F.pad(stage4_unit1_relu1, (1, 1, 1, 1))
        stage4_unit1_conv2 = self.stage4_unit1_conv2(stage4_unit1_conv2_pad)
        stage4_unit1_bn3 = self.stage4_unit1_bn3(stage4_unit1_conv2)
        if stage4_unit1_bn3.requires_grad:
            stage4_unit1_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus46 = stage4_unit1_bn3 + stage4_unit1_sc  # !!!!!!
        stage4_unit2_bn1 = self.stage4_unit2_bn1(plus46)
        stage4_unit2_conv1_pad = F.pad(stage4_unit2_bn1, (1, 1, 1, 1))
        stage4_unit2_conv1 = self.stage4_unit2_conv1(stage4_unit2_conv1_pad)
        stage4_unit2_bn2 = self.stage4_unit2_bn2(stage4_unit2_conv1)
        stage4_unit2_relu1 = F.prelu(stage4_unit2_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage4_unit2_relu1']['weights']))
        stage4_unit2_conv2_pad = F.pad(stage4_unit2_relu1, (1, 1, 1, 1))
        stage4_unit2_conv2 = self.stage4_unit2_conv2(stage4_unit2_conv2_pad)
        stage4_unit2_bn3 = self.stage4_unit2_bn3(stage4_unit2_conv2)
        if stage4_unit2_bn3.requires_grad:
            stage4_unit2_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus47 = stage4_unit2_bn3 + plus46  # !!!!!
        stage4_unit3_bn1 = self.stage4_unit3_bn1(plus47)
        stage4_unit3_conv1_pad = F.pad(stage4_unit3_bn1, (1, 1, 1, 1))
        stage4_unit3_conv1 = self.stage4_unit3_conv1(stage4_unit3_conv1_pad)
        stage4_unit3_bn2 = self.stage4_unit3_bn2(stage4_unit3_conv1)
        stage4_unit3_relu1 = F.prelu(stage4_unit3_bn2,
                                     torch.cuda.FloatTensor(__weights_dict['stage4_unit3_relu1']['weights']))
        stage4_unit3_conv2_pad = F.pad(stage4_unit3_relu1, (1, 1, 1, 1))
        stage4_unit3_conv2 = self.stage4_unit3_conv2(stage4_unit3_conv2_pad)
        stage4_unit3_bn3 = self.stage4_unit3_bn3(stage4_unit3_conv2)
        if stage4_unit3_bn3.requires_grad:
            stage4_unit3_bn3.register_hook(lambda grad: grad * X.rvs(1)[0])
        plus48 = stage4_unit3_bn3 + plus47  # !!!!!
        plus48 = self.AdaptiveAvgPool2d(plus48)
        bn1 = self.bn1(plus48)
        dropout0 = F.dropout(input=bn1, p=0.4000000059604645, training=self.training, inplace=True)
        pre_fc1 = self.pre_fc1(dropout0.view(dropout0.size(0), -1))
        fc1 = self.fc1(pre_fc1)
        return fc1

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.cuda.FloatTensor(__weights_dict[name]['bias']))
        return layer
