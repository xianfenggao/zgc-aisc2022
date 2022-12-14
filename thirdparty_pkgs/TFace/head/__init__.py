def get_head(key, dist_fc):
    """ Get different classification head functions by key, support NormFace CosFace, ArcFace, CurricularFace.
        If distfc is True, the weight is splited equally into all gpus and calculated in parallel
    """
    if dist_fc:
        from .distfc.arcface import ArcFace
        from .distfc.cosface import CosFace
        from .distfc.curricularface import CurricularFace
        from .distfc.normface import NormFace
        _head_dict = {
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'NormFace': NormFace
        }
    else:
        from .localfc.cosface import CosFace
        from .localfc.arcface import ArcFace
        from .localfc.curricularface import CurricularFace
        from .localfc.cifp import Cifp
        from .localfc.adaface import AdaFace
        _head_dict = {
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'Cifp': Cifp,
            'AdaFace': AdaFace,
        }
    if key in _head_dict.keys():
        return _head_dict[key]
    else:
        raise KeyError("not support head {}".format(key))
