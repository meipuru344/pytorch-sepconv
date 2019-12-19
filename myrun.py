# -*- coding: utf-8 -*-
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.utils.serialization

#####定数の定義#####
device = 'cpu'
arguments_strModel = 'lf'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.png'

#try:
	#from sepconv import sepconv # the custom separable convolution layer
#except:
sys.path.insert(0, './sepconv'); import sepconv # you should consider upgrading python
# end


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #####カスタムレイヤーの宣言#####
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
        )
        # end
        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
            # end
        #####ネットワーク層の初期化#####
        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

    # end
    #####順方向計算#####
    def forward(self, tensorFirst, tensorSecond):
        tensorJoin = torch.cat([ tensorFirst, tensorSecond ], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2
        print(self.moduleVertical1(tensorCombine).size())
        print(self.moduleHorizontal1(tensorCombine).size())
        #print(tensorCombine.size(0), tensorCombine.size(1),tensorCombine.size(2),tensorCombine.size(3))
        sepconv.FunctionSepconv().forward(self.modulePad(tensorFirst), self.moduleVertical1(tensorCombine), self.moduleHorizontal1(tensorCombine))
        #tensorDot2 = sepconv.FunctionSepconv().forward(self.modulePad(tensorSecond), self.moduleVertical2(tensorCombine), self.moduleHorizontal2(tensorCombine))
        return torch.zeros(1,1,388+124, 584+56)#1,1,512,640
	# end
# end

#####ネットワークの宣言#####
moduleNetwork = Network()
moduleNetwork.to(device)

#####画像の計算#####
def estimate(tensorFirst, tensorSecond):
    tensorOutput = torch.FloatTensor()

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)
    intPaddingLeft = int(math.floor(51 / 2.0))
    intPaddingTop = int(math.floor(51 / 2.0))
    intPaddingRight = int(math.floor(51 / 2.0))
    intPaddingBottom = int(math.floor(51 / 2.0))
    modulePaddingInput = torch.nn.Sequential()
    modulePaddingOutput = torch.nn.Sequential()

    if True:
        intPaddingWidth = intPaddingLeft + intWidth + intPaddingRight
        intPaddingHeight = intPaddingTop + intHeight + intPaddingBottom
        print('intpaddingwidth')
        print(intPaddingWidth)
        #####高さを128の倍数に変更
        if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
            intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7) # more than necessary
            print('intpaddingwidth')
            print(intPaddingWidth)
        # end

        if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
            intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7) # more than necessary
        # end

        intPaddingWidth = intPaddingWidth - (intPaddingLeft + intWidth + intPaddingRight)
        print('intpaddingwidth')
        print(intPaddingWidth)
        intPaddingHeight = intPaddingHeight - (intPaddingTop + intHeight + intPaddingBottom)
        print([ intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight ])
        modulePaddingInput = torch.nn.ReplicationPad2d(padding=[ intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight ])
        modulePaddingOutput = torch.nn.ReplicationPad2d(padding=[ 0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight ])
	# end

    if True:
        tensorFirst = tensorFirst.to(device)
        tensorSecond = tensorSecond.to(device)
        tensorOutput = tensorOutput.to(device)

        modulePaddingInput = modulePaddingInput.to(device)
        modulePaddingOutput = modulePaddingOutput.to(device)
# end

    if True:
        tensorPreprocessedFirst = modulePaddingInput(tensorFirst.view(1, 3, intHeight, intWidth))
        tensorPreprocessedSecond = modulePaddingInput(tensorSecond.view(1, 3, intHeight, intWidth))
        print(intWidth,intHeight)################################              moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)
        tensorOutput.resize_(3, intHeight, intWidth).copy_(modulePaddingOutput(moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond))[0])
	# end

    if True:
        tensorFirst = tensorFirst.to(device)
        tensorSecond = tensorSecond.to(device)
        tensorOutput = tensorOutput.to(device)
        # end

    return tensorOutput
# end


#####main関数的#####
if True:
    tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    print('picture roaded')
    tensorOutput = estimate(tensorFirst, tensorSecond)
    #####tensorOutput seems to be 3 x 1280 x 720
    PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)


print('process finished')
