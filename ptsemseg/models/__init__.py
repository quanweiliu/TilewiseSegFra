import copy

from ptsemseg.models.Baseline.Resnet18_single_branch import CRFN_base18_single
from ptsemseg.models.Baseline.Resnet18_double_branch import CRFN_base18_double
from ptsemseg.models.Baseline.Resnet34_single_decoder1 import Resnet_base34_decoder1
from ptsemseg.models.Baseline.Resnet34_single_decoder2 import Resnet_base34_decoder2
from ptsemseg.models.Baseline.Resnet34_double_branch import Resnet_base34_double
from ptsemseg.models.AMSUnet import AMSUnet
from ptsemseg.models.MANet import MANet
from ptsemseg.models.ABCNet.ABCNet import ABCNet

from ptsemseg.models.ACNet.ACNet import ACNet
from ptsemseg.models.AsymFormer.AsymFormer import B0_T
from ptsemseg.models.CANet.CANet import CANet

from ptsemseg.models.CMANet.CMANet import CMANet
from ptsemseg.models.CMFNet.CMFNet import CMFNet
from ptsemseg.models.CMGFNet.CMGFNet18 import CMGFNet18
from ptsemseg.models.CMGFNet.CMGFNet34 import CMGFNet34

from ptsemseg.models.DE_CCFNet.DE_CCFNet18 import DE_CCFNet18
from ptsemseg.models.DE_CCFNet.DE_CCFNet34 import DE_CCFNet34
from ptsemseg.models.DE_CCFNet.DE_CCFNet_3Branch import DE_CCFNet34_3B
from ptsemseg.models.DE_DCGCN.DE_DCGCN import DEDCGCNEE
from ptsemseg.models.FAFNet.fafnet_alignD import FAFNet50, FAFNet101

from ptsemseg.models.MCANet.mcanet import MCANet
from ptsemseg.models.MGFNet_Wei.MGFNet_Wei50 import MGFNet_Wei50
from ptsemseg.models.MGFNet_Wei.MGFNet_Wei101 import MGFNet_Wei101
from ptsemseg.models.MGFNet_Wu.MGFNet_Wu34 import MGFNet_Wu34
from ptsemseg.models.MGFNet_Wu.MGFNet_Wu50 import MGFNet_Wu50

from ptsemseg.models.PCGNet.PCGNet18 import PCGNet18
from ptsemseg.models.PCGNet.PCGNet34 import PCGNet34
from ptsemseg.models.PACSCNet.PACSCNet_ import PACSCNet

from ptsemseg.models.SFAFMA.SFAFMA50 import SFAFMA50
from ptsemseg.models.SFAFMA.SFAFMA101 import SFAFMA101
from ptsemseg.models.SFAFMA.SFAFMA152 import SFAFMA152
from ptsemseg.models.SOLCV7.solcv7 import SOLCV7

from ptsemseg.models.SAM_MLoRA.sam_adapter import build_sam_vit_b_adapter_linknet
from ptsemseg.models.SAM_MLoRA.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora
from ptsemseg.models.SAM_MLoRA.sam_lora96_96 import build_sam_vit_b_adapter_linknet_lora96_96
from ptsemseg.models.extend_sam.extend_sam import SemanticSam

def get_model(model_dict, bands1, bands2, classes, classification="Multi", image_size=[256, 256]):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    print("model name", name)

    if name == "baseline18_single":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_single_decoder1":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_single_decoder2":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "AMSUnet":
        model = model(bands1, bands2, num_classes=classes, classification=classification)
    elif name == "MANet":
        model = model(bands1, bands2, num_classes=classes, classification=classification)
    elif name == "ABCNet":
        model = model(bands1, bands2, n_classes=classes, classification=classification)

    elif name == "baseline18_double":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)
    elif name == "baseline34_double":
        model = model(bands1, bands2, n_classes=classes, classification=classification, **param_dict)

    elif name == "AsymFormer_b0":
        model = model(bands1, bands2, n_classes=classes, classification=classification)
    elif name == "ACNet":
        model = model(bands1, bands2, num_class=classes, classification=classification, **param_dict)
    elif name == "CMFNet":
        model = model(bands1, bands2, out_channels=classes, classification=classification, image_size=image_size)
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
        model = model(bands1, bands2, n_classes=classes, classification=classification, is_pretrained=True, **param_dict)
    elif name == "DE_CCFNet34":
        model = model(bands1, bands2, n_classes=classes, classification=classification, is_pretrained=True, **param_dict)
    elif name == "DE_CCFNet34_3B":
        model = model(bands1=3, bands2=2, bands3=12, n_classes=classes, classification=classification, is_pretrained=True, **param_dict)
    elif name == "DE_DCGCN":
        model = model(in_x=bands1, in_y=bands2, n_classes=classes, classification=classification)
    elif name == "FAFNet50":
        model = model(bands1, bands2, num_classes=classes, classification=classification, pretrained=True)
    elif name == "FAFNet101":
        model = model(bands1, bands2, num_classes=classes, classification=classification, pretrained=True)
    elif name == "MCANet":
        model = model(bands1, bands2, num_classes=classes, classification=classification)
    elif name == "MGFNet_Wei50":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)
    elif name == "MGFNet_Wei101":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)
    elif name == "MGFNet_Wu34":
        model = model(bands1, bands2, num_classes=classes, classification=classification, **param_dict)
    elif name == "MGFNet_Wu50":
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
        model = model(bands1, bands2, num_classes=classes, classification=classification, ind=50, pretrained=True)
    elif name == "PACSCNet101":
        model = model(bands1, bands2, num_classes=classes, classification=classification, ind=101, pretrained=True)

    elif name == "b_adapter_sam":
        freeze_strategy = param_dict.pop('freeze_strategy', 'all_except_adapter')
        freeze_until_block = param_dict.pop('freeze_until_block', 10)
        model, encoder_global_attn_indexes = model(
            checkpoint='/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains/sam_vit_b_01ec64.pth',
            n_classes=classes, 
            image_size=image_size[0],
            freeze_strategy=freeze_strategy,
            freeze_until_block=freeze_until_block)
    elif name == "b_adapter_sam_multi_lora32":
        freeze_strategy = param_dict.pop('freeze_strategy', 'all_except_adapter')
        freeze_until_block = param_dict.pop('freeze_until_block', 10)
        model, encoder_global_attn_indexes = model(
            checkpoint='/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains/sam_vit_b_01ec64.pth',
            n_classes=classes, 
            image_size=image_size[0],
            freeze_strategy=freeze_strategy,
            freeze_until_block=freeze_until_block)
    elif name == "b_adapter_sam_lora96_96":
        freeze_strategy = param_dict.pop('freeze_strategy', 'all_except_adapter')
        freeze_until_block = param_dict.pop('freeze_until_block', 10)
        model, encoder_global_attn_indexes = model(
            checkpoint='/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains/sam_vit_b_01ec64.pth',
            n_classes=classes, 
            image_size=image_size[0],
            freeze_strategy=freeze_strategy,
            freeze_until_block=freeze_until_block)
    elif name == "extend_sam_b":
        model = model(ckpt_path="/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains/sam_vit_b_01ec64.pth", 
                      class_num=6, 
                      model_type='vit_b',
                      input_size=256)
        
    elif name == "extend_sam_l":
        model = model(ckpt_path="/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/pretrains/sam_vit_l_0b3195.pth", 
                      class_num=6, 
                      model_type='vit_l',
                      input_size=256)
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
            "AMSUnet": AMSUnet,
            "MANet": MANet,
            "ABCNet": ABCNet,

            "ACNet": ACNet,
            "AsymFormer_b0": B0_T,
            # "AsymFormer_b1": B1_T,
            # "AsymFormer_b3": B3_T,
            # "AsymFormer_b5": B5_T,
            "CANet50": CANet,
            "CANet101": CANet,
            "CMANet": CMANet,
            "CMFNet": CMFNet,
            "CMGFNet18": CMGFNet18,
            "CMGFNet34": CMGFNet34,
            "DE_CCFNet18": DE_CCFNet18,
            "DE_CCFNet34": DE_CCFNet34,
            "DE_CCFNet34_3B": DE_CCFNet34_3B,
            "DE_DCGCN": DEDCGCNEE,
            "FAFNet50": FAFNet50,
            "FAFNet101": FAFNet101,
            "MCANet": MCANet,
            "MGFNet_Wei50": MGFNet_Wei50,
            "MGFNet_Wei101": MGFNet_Wei101,
            "MGFNet_Wu34": MGFNet_Wu34,
            "MGFNet_Wu50": MGFNet_Wu50,
            "PCGNet18": PCGNet18,
            "PCGNet34": PCGNet34,
            "SFAFMA50": SFAFMA50,
            "SFAFMA101": SFAFMA101,
            "SFAFMA152": SFAFMA152,
            "SOLC": SOLCV7,
            "PACSCNet50": PACSCNet,
            "PACSCNet101": PACSCNet,
            "b_adapter_sam": build_sam_vit_b_adapter_linknet,
            "b_adapter_sam_lora96_96": build_sam_vit_b_adapter_linknet_lora96_96,
            "b_adapter_sam_multi_lora32": build_sam_vit_b_adapter_linknet_multi_lora,
            "extend_sam_b": SemanticSam,
            "extend_sam_l": SemanticSam,
        }[name]
    except:
        raise("Model {} not available".format(name))


