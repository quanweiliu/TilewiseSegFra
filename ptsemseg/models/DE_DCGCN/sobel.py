import torch
import numpy as np
from tifffile import tifffile
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
# from torchsummaryX import summary


def edge_conv2d(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def edge_conv2d64(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 64, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 64, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 64, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 64, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 64, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 64, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 64, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 64, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def edge_conv2d128(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 128, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 128, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 128, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 128, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 128, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 128, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 128, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 128, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def edge_conv2d256(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 256, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 256, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 256, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 256, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 256, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 256, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 256, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 256, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out



# def edge_conv2d_2(im):

#     conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#     sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
#     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     sobel_kernel = np.repeat(sobel_kernel, 512, axis=1)
#     sobel_kernel = np.repeat(sobel_kernel, 512, axis=0)
#     conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
#     edge_detect = torch.abs(conv_op(Variable(im)))

#     conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#     sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
#     sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
#     sobel_kernel1 = np.repeat(sobel_kernel1, 512, axis=1)
#     sobel_kernel1 = np.repeat(sobel_kernel1, 512, axis=0)
#     conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
#     edge_detect1 = torch.abs(conv_op1(Variable(im)))

#     conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#     sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
#     sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
#     sobel_kernel2 = np.repeat(sobel_kernel2, 512, axis=1)
#     sobel_kernel2 = np.repeat(sobel_kernel2, 512, axis=0)
#     conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
#     edge_detect2 = torch.abs(conv_op2(Variable(im)))

#     conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#     sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
#     sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
#     sobel_kernel3 = np.repeat(sobel_kernel3, 512, axis=1)
#     sobel_kernel3 = np.repeat(sobel_kernel3, 512, axis=0)
#     conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
#     edge_detect3 = torch.abs(conv_op3(Variable(im)))
#     # print(conv_op.weight.size())
#     # print(conv_op, '\n')

#     sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

#     return sobel_out

def Gedge_map(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out




if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # x = torch.randn(1, 64,256,256, device=device)
    # y = torch.randn(1,64,256,256,device=device)
    path1=r'G:\buffalo_new\multi_data7\test\img\buffalo_1_9.tif'
    # path1=r'C:\Users\15162\Desktop\test.jpg'
    x = np.array(Image.open(path1),np.float32)
    y=Gedge_map(torch.tensor(x))
    a=1

    # y = torch.randn(4,2,512,512,device=device)
    # model =GCN(512)
    # model=SKBlock(64,M=2,G=64)
    # model=cfm(64)
    # summary(model.to(device), x)