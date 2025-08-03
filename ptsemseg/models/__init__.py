import copy

from ptsemseg.models.Baseline.Resnet34_base_decoder1 import Resnet_base34
from ptsemseg.models.Baseline.Resnet18_single_branch import CRFN_base18

from ptsemseg.models.DE_CCFNet.DE_CCFNet_18 import DE_CCFNet_18
from ptsemseg.models.DE_CCFNet.DE_CCFNet_34 import DE_CCFNet_34
from ptsemseg.models.DE_DCGCN.DE_DCGCN import DEDCGCNEE
from ptsemseg.models.HAFNetE.HAFNetE import EfficientHAFNet
from ptsemseg.models.PCGNet.PCGNet34 import PCGNet
from ptsemseg.models.SFAFMA.SFAFMA50 import SFAFMA50
from ptsemseg.models.SFAFMA.SFAFMA101 import SFAFMA101
from ptsemseg.models.SFAFMA.SFAFMA152 import SFAFMA152

from ptsemseg.models.ACNet.ACNet import ACNet
from ptsemseg.models.CANet.CANet import CANet
from ptsemseg.models.CMANet.CMANet import CMAnet
from ptsemseg.models.CMGFNet.CMGFNet18 import CMGFNet18
from ptsemseg.models.CMGFNet.CMGFNet34 import CMGFNet34

from ptsemseg.models.PACSCNet.PACSCNet import PACSCNet
from ptsemseg.models.CMFNet.CMFNet import CMFNet
from ptsemseg.models.AsymFormer.AsymFormer import B0_T
from ptsemseg.models.RDFNet.rdfnet50 import RDF


def get_model(model_dict, n_classes, version=None):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    print("model name", name)

    # "前两个是 resNet 50"
    if name == "baseline18":
        # model = model(n_classes=2, data='rgb', **param_dict)
        model = model(n_classes=2, is_pretrained="ResNet18_Weights.DEFAULT", data='lidar', **param_dict)
    elif name == "baseline34":
        # model = model(n_classes=2, data='lidar', **param_dict)
        model = model(n_classes=2, is_pretrained="ResNet34_Weights.DEFAULT", data='lidar',**param_dict)

    elif name == "DE_CCFNet_18":
        model = model(n_classes=1, is_pretrained="ResNet18_Weights.IMAGENET1K_V1", **param_dict)
    elif name == "DE_CCFNet_34":
        model = model(n_classes=1, is_pretrained="ResNet34_Weights.IMAGENET1K_V1", **param_dict)
    elif name == "DE_DCGCN":
        model = model(in_x=193, in_y=3, n_classes=2)
    elif name == "HAFNetE":
        model = model(n_classes=1)
    elif name == "SFAFMA50":
        model = model(n_classes=2)
    elif name == "SFAFMA101":
        model = model(n_classes=2)
    elif name == "PCGNet":
        model = model(n_classes=2, is_pretrained="ResNet34_Weights.IMAGENET1K_V1")


    elif name == "CANet":
        model = model(num_class=2, backbone='ResNet-50', pretrained=True, pcca5=True, **param_dict)
    elif name == "ACNet":
        model = model(num_class=1, pretrained=True, **param_dict)
    elif name == "CMANet":
        model = model(n_classes=1, pretrained=True)
    elif name == "CMGFNet18":
        model = model(n_classes=1, pretrained="ResNet18_Weights.DEFAULT", **param_dict)
    elif name == "CMGFNet34":
        model = model(n_classes=1, pretrained="ResNet34_Weights.DEFAULT", **param_dict)

    elif name == "AsymFormer":
        model = model(n_classes=1)
    elif name == "CMFNet":
        model = model(**param_dict)
    elif name == "CMFNet_U":
        model = model(**param_dict)
    elif name == "PACSCNet":
        model = model(num_classes=2, ind=50, **param_dict)
    elif name == "RDFNet":
        model = model(input_size=128, num_classes=1, pretained=False)
    else:
        raise("you havn't set the model parameters")
    return model


def _get_model_instance(name):
    try:
        return {
            "baseline18": CRFN_base18,
            "baseline34": Resnet_base34,

            "DE_CCFNet_18": DE_CCFNet_18,
            "DE_CCFNet_34": DE_CCFNet_34,
            "DE_DCGCN": DEDCGCNEE,
            "HAFNetE": EfficientHAFNet,
            "SFAFMA50": SFAFMA50,
            "SFAFMA101": SFAFMA101,
            "SFAFMA152": SFAFMA152,
            "PCGNet": PCGNet,

            "CANet": CANet,
            "ACNet": ACNet,
            "CMANet": CMAnet,
            "CMGFNet18": CMGFNet18,
            "CMGFNet34": CMGFNet34,
            "AsymFormer": B0_T,
            "CMFNet": CMFNet,
            "PACSCNet": PACSCNet,     
            "RDFNet": RDF,     
        }[name]
    except:
        raise("Model {} not available".format(name))


