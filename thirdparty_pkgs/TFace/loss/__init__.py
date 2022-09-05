from torch.nn import CrossEntropyLoss

from .ddl import DDL
from .dist_softmax import DistCrossEntropy
from .focal import FocalLoss
from .poly_focal import PolyFocalLoss

_loss_dict = {
    'Softmax': CrossEntropyLoss(),
    'DistCrossEntropy': DistCrossEntropy(),
    'FocalLoss': FocalLoss(),
    'DDL': DDL(),
    'PolyFocalLoss': PolyFocalLoss(),   # not work now
}


def get_loss(key):
    """ Get different training loss functions by key,
        support Softmax(distfc = False), DistCrossEntropy (distfc = True), FocalLoss, and DDL.
    """
    if key in _loss_dict.keys():
        return _loss_dict[key]
    else:
        raise KeyError("not support loss {}".format(key))
