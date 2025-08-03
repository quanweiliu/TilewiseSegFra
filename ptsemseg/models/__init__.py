import copy

from ptsemseg.models.Baseline.Resnet34_base_decoder1 import Resnet_base34
from ptsemseg.models.Baseline.Resnet18_single_branch import CRFN_base18


from ptsemseg.models.ACNet.ACNet import ACNet
from ptsemseg.models.AsymFormer.AsymFormer import B0_T
from ptsemseg.models.CANet.CANet import CANet

from ptsemseg.models.CMANet.CMANet import CMAnet
from ptsemseg.models.CMFNet.CMFNet import CMFNet
from ptsemseg.models.CMGFNet.CMGFNet18 import CMGFNet18
from ptsemseg.models.CMGFNet.CMGFNet34 import CMGFNet34

from ptsemseg.models.DE_CCFNet.DE_CCFNet_18 import DE_CCFNet_18
from ptsemseg.models.DE_CCFNet.DE_CCFNet_34 import DE_CCFNet_34
from ptsemseg.models.DE_DCGCN.DE_DCGCN import DEDCGCNEE
from ptsemseg.models.HAFNetE.HAFNetE import EfficientHAFNet

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



def get_model(model_dict, bands1, bands2, classes, input_size=256):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    print("model name", name)

    # "前两个是 resNet 50"
    if name == "baseline18":
        model = model(bands1, bands2, n_classes=classes, is_pretrained="ResNet18_Weights.DEFAULT", data='lidar', **param_dict)
    elif name == "baseline34":
        model = model(bands1, bands2, n_classes=classes, is_pretrained="ResNet34_Weights.DEFAULT", data='lidar',**param_dict)
    elif name == "AsymFormer":
        model = model(bands1, bands2, n_classes=classes)
    elif name == "ACNet":
        model = model(bands1, bands2, num_class=classes, pretrained=True, **param_dict)
    elif name == "CMFNet":
        model = model(bands1, bands2, **param_dict)
    elif name == "CANet":
        model = model(bands1, bands2, num_class=classes, backbone='ResNet-50', pretrained=True, pcca5=True, **param_dict)

    elif name == "CMANet":
        model = model(bands1, bands2, n_classes=classes, pretrained=True)
    elif name == "CMGFNet18":
        model = model(bands1, bands2, n_classes=classes, pretrained="ResNet18_Weights.DEFAULT", **param_dict)
    elif name == "CMGFNet34":
        model = model(bands1, bands2, n_classes=classes, pretrained="ResNet34_Weights.DEFAULT", **param_dict)

    elif name == "DE_CCFNet_18":
        model = model(bands1, bands2, n_classes=classes, is_pretrained="ResNet18_Weights.IMAGENET1K_V1", **param_dict)
    elif name == "DE_CCFNet_34":
        model = model(bands1, bands2, n_classes=classes, is_pretrained="ResNet34_Weights.IMAGENET1K_V1", **param_dict)
    elif name == "DE_DCGCN":
        model = model(in_x=bands1, in_y=bands2, n_classes=classes)
    elif name == "HAFNetE":
        model = model(bands1, bands2, n_classes=classes)

    elif name == "MGFNet50":
        model = model(bands1, bands2, num_classes=classes, **param_dict)
    elif name == "MGFNet101":
        model = model(bands1, bands2, num_classes=classes, **param_dict)

    elif name == "SOLC":
        model = model(bands1, bands2, num_classes=classes, **param_dict)
    elif name == "SFAFMA50":
        model = model(bands1, bands2, n_classes=classes)
    elif name == "SFAFMA101":
        model = model(bands1, bands2, n_classes=classes)
    elif name == "PCGNet":
        model = model(bands1, bands2, n_classes=classes, is_pretrained="ResNet34_Weights.IMAGENET1K_V1")

    elif name == "PACSCNet50":
        model = model(bands1, bands2, num_classes=classes, ind=50, **param_dict)
    elif name == "PACSCNet101":
        model = model(bands1, bands2, num_classes=classes, ind=101, **param_dict)
    elif name == "RDFNet50":
        model = model(bands1, bands2, input_size=input_size, num_classes=classes, pretained=False)
    elif name == "RDFNet101":
        model = model(bands1, bands2, input_size=input_size, num_classes=classes, pretained=False)
    else:
        raise("you havn't set the model parameters")
    return model


def _get_model_instance(name):
    try:
        return {
            "baseline18": CRFN_base18,
            "baseline34": Resnet_base34,

            "ACNet": ACNet,
            "AsymFormer": B0_T,
            "CANet": CANet,
            "CMANet": CMAnet,
            "CMFNet": CMFNet,
            "CMGFNet18": CMGFNet18,
            "CMGFNet34": CMGFNet34,
            "DE_CCFNet_18": DE_CCFNet_18,
            "DE_CCFNet_34": DE_CCFNet_34,
            "DE_DCGCN": DEDCGCNEE,
            "HAFNetE": EfficientHAFNet,
            "MCANet": MCANet,
            "MGFNet50": MGFNet50,
            "MGFNet101": MGFNet101,
            "PCGNet18": PCGNet18,
            "PCGNet34": PCGNet34,
            "SFAFMA50": SFAFMA50,
            "SFAFMA101": SFAFMA101,
            "SFAFMA152": SFAFMA152,
            "SOLC": SOLCV7,
            "PACSCNet": PACSCNet,
            "RDFNet50": RDF50,
            "RDFNet101": RDF101,
        }[name]
    except:
        raise("Model {} not available".format(name))


