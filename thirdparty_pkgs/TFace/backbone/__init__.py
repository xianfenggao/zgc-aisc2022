from functools import partial

from .fbnets.fbnet_builder import get_fbnet_model
from .model_densenet import densenet121, densenet169, densenet201
from .model_efficientnet import (EfficientNetB0,
                                 EfficientNetB1)
from .model_ghostnet import GhostNet
from .model_irse import IR_18, IR_34, IR_50, IR_101, IR_152, IR_200
from .model_irse import IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200
from .model_mobilefacenet import MobileFaceNet
from .model_resnet import ResNet_50, ResNet_101, ResNet_152
from .model_swin import SwinTransformerFaceXZOO

_model_dict = {
    'ResNet_50': ResNet_50,
    'ResNet_101': ResNet_101,
    'ResNet_152': ResNet_152,
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_200': IR_200,
    'IR_SE_50': IR_SE_50,
    'IR_SE_101': IR_SE_101,
    'IR_SE_152': IR_SE_152,
    'IR_SE_200': IR_SE_200,
    'MobileFaceNet': MobileFaceNet,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'GhostNet': GhostNet,
    'fbnet_a': partial(get_fbnet_model, "fbnet_a"),
    'fbnet_b': partial(get_fbnet_model, "fbnet_b"),
    'fbnet_c': partial(get_fbnet_model, "fbnet_c"),

    'DenseNet121': densenet121,
    'DenseNet169': densenet169,
    'DenseNet201': densenet201,
    'SwinTransformer': SwinTransformerFaceXZOO,
}


def get_model(key):
    """ Get different backbone network by key,
        support ResNet50, ResNet_101, ResNet_152
        IR_18, IR_34, IR_50, IR_101, IR_152, IR_200,
        IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200,
        EfficientNetB0, EfficientNetB1.
        MobileFaceNet, FBNets.
    """
    if key in _model_dict.keys():
        return _model_dict[key]
    else:
        raise KeyError("not support model {}".format(key))
