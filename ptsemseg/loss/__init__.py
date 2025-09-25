import copy
import logging
import functools
import torch.nn.functional as F

from ptsemseg.loss.GapLoss import GapLoss, BinaryGapLoss, gaploss
from ptsemseg.loss.cldice import dice_cldice_loss, bce_cldice_loss
from ptsemseg.loss.loss import cross_entropy2d, new_dice_bce_loss, \
                                focal_bce_loss, dice_bce_loss_re, \
                                dice_bce_loss_re2, multi_loss, \
                                multi_loss2, multi_loss3
from ptsemseg.loss.loss import multiclass_ce_dice_loss
from ptsemseg.loss.loss import multiclass_multi_loss
from ptsemseg.loss.loss import focal_dice_loss
from ptsemseg.loss.loss import focal_loss
from ptsemseg.loss.loss import bce_loss
from ptsemseg.loss.loss import dice_loss
from ptsemseg.loss.loss import MSE
from ptsemseg.loss.abcLoss import abc_loss


logger = logging.getLogger('ptsemseg')

key2loss = {
            'cross_entropy2d': cross_entropy2d,
            'focal_dice_loss': focal_dice_loss,
            'focal_loss': focal_loss,
            'bce_loss': bce_loss,
            'MSE': MSE,
            'dice_loss':dice_loss,
            'new_dice_bce_loss': new_dice_bce_loss,
            'dice_cldice_loss':dice_cldice_loss,
            'bce_cldice_loss':bce_cldice_loss,
            'focal_bce_loss':focal_bce_loss,
            'dice_bce_loss_re':dice_bce_loss_re,
            'dice_bce_loss_re2':dice_bce_loss_re2,
            'multiclass_ce_dice_loss':multiclass_ce_dice_loss,
            'multiclass_multi_loss':multiclass_multi_loss,
            'gaploss':gaploss,
            'multi_loss':multi_loss,
            'multi_loss2':multi_loss2,
            'multi_loss3':multi_loss3,
            "abc_loss": abc_loss,
            }


def get_loss_function(cfg):
    if cfg['training']['loss'] is None:
        logger.info("Using default bce_loss")
        return bce_loss()

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k, v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)


# def get_loss_function2(cfg):
#     if cfg['training']['loss'] is None:
#         logger.info("Using default bce_loss")
#         return bce_loss()
#
#     else:
#         loss_dict = cfg['training']['loss']
#         loss_name = loss_dict['name']
#         loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}
#         loss_params['a']=cfg['a']
#         loss_params['b'] = cfg['b']
#
#
#         if loss_name not in key2loss:
#             raise NotImplementedError('Loss {} not implemented'.format(loss_name))
#
#         logger.info('Using {} with {} params'.format(loss_name,
#                                                      loss_params))
#         return functools.partial(key2loss[loss_name], **loss_params)
