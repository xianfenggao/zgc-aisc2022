from .FaceNet import FaceNet_casia, FaceNet_vggface2
from .Face_Robustness_Benchmark import FRB_ArcFace_IR_50, FRB_CosFace, FRB_SphereFace, FRB_ResNet50, FRB_MobileNet, \
    FRB_ShuffleNetV1
from .InsightFace.mxnet.InsightFace_Mxnet import MXNET_LResNet34E_IR, MXNET_LResNet50E_IR, MXNET_LResNet100E_IR
from .InsightFace.mxnet.InsightFace_Mxnet_SGM import MXNET_LResNet34E_IR_SGM, MXNET_LResNet50E_IR_SGM, \
    MXNET_LResNet100E_IR_SGM
from .InsightFace.pytorch.ArcFaceGlint360k import ArcFace_torch_ir50_glint360k, ArcFace_torch_ir100_glint360k
from .InsightFace.pytorch.Arcface_torch import ArcFace_torch_ir50_faceemore, ArcFace_torch_ir200_faceemore
from .TFace.TFace import *
from .TFace.advprop import *
from .TFace.splitbn import *
from .evoLVe_irse import evoLVe_IR_152, evoLVe_IR_50, evoLVe_IR_50_Asia

__all__ = [
    'model_name_dict',
    'getmodels',
    'getmodel',
    'BaseFaceModel'
]

model_name_dict = {

    'evoLVe_IR_152': evoLVe_IR_152,
    'evoLVe_IR_50': evoLVe_IR_50,
    'evoLVe_IR_50_Asia': evoLVe_IR_50_Asia,

    'FaceNet_casia': FaceNet_casia,
    'FaceNet_vggface2': FaceNet_vggface2,

    'MXNET_LResNet34E_IR': MXNET_LResNet34E_IR,
    'MXNET_LResNet50E_IR': MXNET_LResNet50E_IR,
    'MXNET_LResNet100E_IR': MXNET_LResNet100E_IR,

    'MXNET_LResNet34E_IR_SGM': MXNET_LResNet34E_IR_SGM,
    'MXNET_LResNet50E_IR_SGM': MXNET_LResNet50E_IR_SGM,
    'MXNET_LResNet100E_IR_SGM': MXNET_LResNet100E_IR_SGM,

    'FRB_ArcFace_IR_50': FRB_ArcFace_IR_50,
    'FRB_CosFace': FRB_CosFace,
    'FRB_MobileNet': FRB_MobileNet,
    'FRB_SphereFace': FRB_SphereFace,
    'FRB_ResNet50': FRB_ResNet50,
    'FRB_ShuffleNetV1': FRB_ShuffleNetV1,

    'ArcFace_torch_ir50_faceemore': ArcFace_torch_ir50_faceemore,
    'ArcFace_torch_ir200_faceemore': ArcFace_torch_ir200_faceemore,
    'ArcFace_torch_ir50_glint360k': ArcFace_torch_ir50_glint360k,
    'ArcFace_torch_ir100_glint360k': ArcFace_torch_ir100_glint360k,

    'TFace_IR_18_glintasia': TFace_IR_18_glintasia,
    'TFace_MobileFaceNet_face_emore': TFace_MobileFaceNet_face_emore,
    'TFace_MobileFaceNet_glintasia': TFace_MobileFaceNet_glintasia,

    'TFace_IR_SE_50_2data': TFace_IR_SE_50_2data,
    'TFace_IR_SE_50_5data': TFace_IR_SE_50_5data,
    'TFace_IR_SE_50_faceemore': TFace_IR_SE_50_faceemore,
    'TFace_IR_SE_50_faceglint': TFace_IR_SE_50_faceglint,
    'TFace_IR_SE_50_glintasia': TFace_IR_SE_50_glintasia,
    'TFace_IR_SE_50_umd': TFace_IR_SE_50_umd,
    'TFace_IR_SE_50_vggface2': TFace_IR_SE_50_vggface2,
    'TFace_IR_SE_50_webface': TFace_IR_SE_50_webface,

    'TFace_MobileFaceNet_Aug_faceemore': TFace_MobileFaceNet_Aug_faceemore,
    'TFace_MobileFaceNet_Aug_glintasia': TFace_MobileFaceNet_Aug_glintasia,
    'TFace_MobileFaceNet_Aug_webface': TFace_MobileFaceNet_Aug_webface,
    'TFace_MobileFaceNet_Aug_vggface2': TFace_MobileFaceNet_Aug_vggface2,
    'TFace_MobileFaceNet_Aug_umd': TFace_MobileFaceNet_Aug_umd,
    'TFace_MobileFaceNet_Aug_glint360k': TFace_MobileFaceNet_Aug_glint360k,

    'TFace_IR101_Aug_faceemore': TFace_IR101_Aug_faceemore,
    'TFace_IR101_Aug_glint360k': TFace_IR101_Aug_glint360k,

    'TFace_IR_SE_50_splitbn_2data': TFace_IR_SE_50_splitbn_2data,
    'TFace_IR_SE_50_splitbn_alldata': TFace_IR_SE_50_splitbn_alldata,
    'TFace_DenseNet201_splitbn_faceemore_glintasia': TFace_DenseNet201_splitbn_faceemore_glintasia,

    'TFace_IR_18_AdvProp_faceemore': TFace_IR_18_AdvProp_faceemore,
    'TFace_EfficientNetB0_AdvProp_faceemore': TFace_EfficientNetB0_AdvProp_faceemore,
    'TFace_IR_SE_50_withaug_faceemore': TFace_IR_SE_50_withaug_faceemore,

}


def getmodels(name_list, device='cuda'):
    res = []
    for model in name_list:
        res.append(getmodel(model, device))
    return res


def getmodel(model_name, device='cuda'):
    if model_name in model_name_dict:
        return model_name_dict[model_name](device=device)
    else:
        raise NotImplementedError(model_name)
