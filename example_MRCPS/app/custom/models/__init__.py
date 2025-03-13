# from ..utils import transformsgpu, transformmasks

from .modelsupervised import SupModel
from .modelmrsupervised import MRSupModel

from .modelcps import CPSModel
from .modelcpsmix import CPSMixModel
from .modelmrcps import MRCPSModel
# from .modelmrtrimix import MRTRIMixModel
# from models.modelmrunimatch import MRUniMatchModel

from .modelmrcpsmix import MRCPSMixModel
# from .modelmrcpsmix_f import MRCPSMixModel_F

# from .modelmsccs import MSCCSModel
# from .modelmscs_d import MSCSDualModel
# from .modelmscs_uni import MSCSUniModel

# from .modelmt import MTModel
# from .modelcutmix import CutMixModel
# from .modelpsmt import PSMTModel
# from .modelunimatch import UniMatchModel

# from .modelgct import GCTModel
# from .modelu2pl import U2PLModel

# from .model_taki import UnetModel
# from .model_qwz import QWZModel

# from .model_mlc import MLCModel

mapping = {
    "sup" : SupModel,
    "mrsup" : MRSupModel,
    "cps" : CPSModel,
    "cpsmix" : CPSMixModel,
    "mrcps" : MRCPSModel,
    "mrcpsmix" : MRCPSMixModel,

    # 模型變形
    # "msccs" : MSCCSModel,
    # "mscsd" : MSCSDualModel,
    # "mscsuni" : MSCSUniModel,
    # "mrcpsmix_f" : MRCPSMixModel_F,
    # "mrtrimix" : MRTRIMixModel,
    # "mrunimatch" : MRUniMatchModel,

    # 模型比較
    # "mt" : MTModel,
    # "psmt" : PSMTModel,
    # "unimatch" : UniMatchModel,
    # "cutmix" : CutMixModel,
    # "gct" : GCTModel,
    # "u2pl" : U2PLModel,

    # 學長姐
    # "taki" : UnetModel,
    # "qwz" : QWZModel,

    # consistency
    # "mlc" : MLCModel
}


def modelfactory(modeltype: str):
    return mapping[modeltype]