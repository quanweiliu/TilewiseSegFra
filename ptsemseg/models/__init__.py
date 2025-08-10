import copy

from ptsemseg.models.Baseline.Resnet18_single_branch import CRFN_base18_single
from ptsemseg.models.Baseline.Resnet18_double_branch import CRFN_base18_double
from ptsemseg.models.Baseline.Resnet34_single_decoder1 import Resnet_base34_decoder1
from ptsemseg.models.Baseline.Resnet34_single_decoder2 import Resnet_base34_decoder2
from ptsemseg.models.Baseline.Resnet34_double_branch import Resnet_base34_double

from ptsemseg.models.ACNet.ACNet import ACNet
from ptsemseg.models.AsymFormer.AsymFormer import B0_T
from ptsemseg.models.CANet.CANet import CANet

from ptsemseg.models.CMANet.CMANet import CMAnet
from ptsemseg.models.CMFNet.CMFNet import CMFNet
from ptsemseg.models.CMGFNet.CMGFNet18 import CMGFNet18
from ptsemseg.models.CMGFNet.CMGFNet34 import CMGFNet34

from ptsemseg.models.DE_CCFNet.DE_CCFNet18 import DE_CCFNet18
from ptsemseg.models.DE_CCFNet.DE_CCFNet34 import DE_CCFNet34
from ptsemseg.models.DE_DCGCN.DE_DCGCN import DEDCGCNEE
from ptsemseg.models.FAFNet.fafnet_alignD import FAFNet

from ptsemseg.models.MCANet.mcanet import MCANet
from ptsemseg.models.MGFNet.MGFNet50 import MGFNet50
from ptsemseg.models.MGFNet.MGFNet101 import MGFNet101

from ptsemseg.models.PCGNet.PCGNet18 import PCGNet18
from ptsemseg.models.PCGNet.PCGNet34 import PCGNet34
from ptsemseg.models.PACSCNet.PACSCNet import PACSCNet
from ptsemseg.models.RDFNet.rdfnet50 import RDF50
from ptsemseg.models.RDFNet.rdfnet101 import RDF101

from ptsemseg.models.SFAFMA.SFAFMA50 import SFAFMA50
from ptsemseg.models.SFAFMA.SFAFMA101 import SFAFMA101
from ptsemseg.models.SFAFMA.SFAFMA152 import SFAFMA152
from ptsemseg.models.SOLCV7.solcv7 import SOLCV7



def get_model(model_dict, bands1, bands2, classes, input_size, classification="Multi"):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    print("model name", name)

    if name == "baseline18_single":
        model = model(bands1, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_single_decoder1":
        model = model(bands1, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_single_decoder2":
        model = model(bands1, n_classes=classes, classification=classification, **param_dict)

    elif name == "baseline18_double":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_double":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)

    elif name == "AsymFormer":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "ACNet":
        model = model(bands1, bands2, num_class=classes, classification=classification, **param_dict)
    elif name == "CMFNet":
        model = model(bands1, bands2, **param_dict)
    elif name == "CANet50":
        model = model(bands1, bands2, num_class=classes, classification=classification, backbone='ResNet-50', pcca5=True, **param_dict)
    elif name == "CANet101":
        model = model(bands1, bands2, num_class=classes, classification=classification, backbone='ResNet-101', pcca5=True, **param_dict)

    elif name == "CMANet":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "CMGFNet18":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "CMGFNet34":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)

    elif name == "DE_CCFNet18":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "DE_CCFNet34":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "DE_DCGCN":
        model = model(in_x=bands1, in_y=bands2, n_classes=classes, classification=classification)
    elif name == "FAFNet":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "MGFNet50":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)
    elif name == "MGFNet101":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)

    elif name == "SOLC":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)
    elif name == "SFAFMA50":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "SFAFMA101":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "PCGNet18":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "PCGNet34":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "PACSCNet50":
        model = model(bands1, bands2, num_classes=classes, classification=classification, ind=50, **param_dict)
    elif name == "PACSCNet101":
        model = model(bands1, bands2, num_classes=classes, classification=classification, ind=101, **param_dict)
    elif name == "RDFNet50":
        model = model(bands1, bands2, input_size=input_size, num_classes=classes, classification=classification, **param_dict)
    elif name == "RDFNet101":
        model = model(bands1, bands2, input_size=input_size, num_classes=classes, classification=classification, **param_dict)
    else:
        raise("you havn't set the model parameters")
    return model


def _get_model_instance(name):
    try:
        return {
            "baseline18_single": CRFN_base18_single,
            "baseline18_double": CRFN_base18_double,
            "baseline34_single_decoder1": Resnet_base34_decoder1,
            "baseline34_single_decoder2": Resnet_base34_decoder2,
            "baseline34_double": Resnet_base34_double,

            "ACNet": ACNet,
            "AsymFormer": B0_T,
            "CANet50": CANet,
            "CANet101": CANet,
            "CMANet": CMAnet,
            "CMFNet": CMFNet,
            "CMGFNet18": CMGFNet18,
            "CMGFNet34": CMGFNet34,
            "DE_CCFNet18": DE_CCFNet18,
            "DE_CCFNet34": DE_CCFNet34,
            "DE_DCGCN": DEDCGCNEE,
            "FAFNet": FAFNet,
            "MCANet": MCANet,
            "MGFNet50": MGFNet50,
            "MGFNet101": MGFNet101,
            "PCGNet18": PCGNet18,
            "PCGNet34": PCGNet34,
            "SFAFMA50": SFAFMA50,
            "SFAFMA101": SFAFMA101,
            "SFAFMA152": SFAFMA152,
            "SOLC": SOLCV7,
            "PACSCNet50": PACSCNet,
            "PACSCNet101": PACSCNet,
            "RDFNet50": RDF50,
            "RDFNet101": RDF101,
        }[name]
    except:
        raise("Model {} not available".format(name))


