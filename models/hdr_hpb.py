import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

from models.getFlow import getFlow
from models.hpb import  UN

from utils.utils import flow_to_image, warp, flow_to_image_torch


class hls(nn.Module):
    def __init__(self):
        super(hls, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)

        return low,high


class AttentionBlock(nn.Module):
    def __init__(self,nFeat):
        super(AttentionBlock, self).__init__()
        self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self,x1,x2):
        Fx_i = torch.cat((x1, x2), 1)
        Fx_A = self.relu(self.att11(Fx_i))
        Fx_A = self.att12(Fx_A)
        Fx_A = nn.functional.sigmoid(Fx_A)
        x = x1 * Fx_A

        return x

class HLFusion(nn.Module):
    def __init__(self, n_feats):
        super(HLFusion, self).__init__()

        self.alise1= nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)

    def forward(self, high,low):

        highfeat=high
        lowfeat = F.interpolate(low, size=high.size()[-2:], mode='bilinear', align_corners=True)

        out=self.alise2(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))

        return out

# Attention Guided HDR, AHDR-Net
class HDR_HPB(nn.Module):
    def __init__(self, nChannel, nFeat, wave):
        super(HDR_HPB, self).__init__()
        # nDenselayer 6    growthRate 32   nBlock 16   nFeat 64   nChannel 6  op_channel  64
        # nChannel = args.nChannel
        # nDenselayer = args.nDenselayer
        # nFeat = args.nFeat
        # growthRate = args.growthRate
        # self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)

        # 高低频分离
        self.hls1 = hls()
        self.hls2 = hls()
        self.hls3 = hls()

        # 低频获取光流信息
        self.flow1 = getFlow()
        self.flow3 = getFlow()

        # 注意力模块
        self.attentionh1 = AttentionBlock(nFeat)
        self.attentionh3 = AttentionBlock(nFeat)


        #对齐之后融合
        self.hlf1=HLFusion(nFeat)
        self.hlf3=HLFusion(nFeat)
        # F0

        self.hpb = UN(nFeat, wave)

        # feature fusion (GFF)
        self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)

        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2, x3, save_flow=False, flow_prefix=None):
        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))

        l1,h1 = self.hls1(F1_)
        l2,h2 = self.hls2(F2_)
        l3,h3 = self.hls3(F3_)

        l1_flow = self.flow1(l1,l2)
        l3_flow = self.flow3(l3,l2)

        # 保存光流图片
        if save_flow:
            os.makedirs('results/flow', exist_ok=True)
            # 只保存第一个batch
            l1_flow_np = l1_flow[0].detach().cpu().numpy().transpose(1, 2, 0)
            l3_flow_np = l3_flow[0].detach().cpu().numpy().transpose(1, 2, 0)
            l1_flow_img = flow_to_image_torch(l1_flow_np)
            l3_flow_img = flow_to_image_torch(l3_flow_np)
            prefix = flow_prefix if flow_prefix is not None else '0'
            cv2.imwrite(f'results/flow/{prefix}_l1_flow.png', l1_flow_img)
            cv2.imwrite(f'results/flow/{prefix}_l3_flow.png', l3_flow_img)

        l1_feat = warp(l1, l1_flow)
        l3_feat = warp(l3, l3_flow)

        h1_feat=self.attentionh1(h1,h2)
        h3_feat=self.attentionh3(h3,h2)



        F1_ = self.hlf1(h1_feat,l1_feat)
        F3_ = self.hlf3(h3_feat,l3_feat)


        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)

        F_hpb=self.hpb(F_0)

        FdLF = self.GFF_1x1(F_hpb)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)

        output = self.conv3(us)
        output = nn.functional.sigmoid(output)

        return output



# def model_structure(model):
#     blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
#     num_para = 0
#     type_size = 1  # 如果是浮点数就是4
#
#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#     print('-' * 90)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 90)
#
#
#
# if __name__ == '__main__':
#     x1 = torch.randn(1, 6, 256, 256)
#     x2 = torch.randn(1, 6, 256, 256)
#     x3 = torch.randn(1, 6, 256, 256)
#     model = HDR_HPB(6, 64, 'haar')
#     model_structure(model)
#     print(model(x1,x2,x3).shape)


def count_flops_params_multi_input(model, inputs, verbose=True):
    """
    通用计算多输入模型的参数量和 FLOPs
    - model: nn.Module
    - inputs: list of torch.Tensor [(B,C,H,W), ...]
    """
    hooks = []
    flops_dict = {}

    def conv_hook(self, input, output):
        batch_size, in_c, h, w = input[0].size()
        out_c, out_h, out_w = output.size()[1:]
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (in_c // self.groups)
        bias_ops = 1 if self.bias is not None else 0
        total_ops = (kernel_ops + bias_ops) * out_c * out_h * out_w * batch_size
        flops_dict[self] = total_ops

    def linear_hook(self, input, output):
        batch_size = input[0].size(0)
        in_features = self.in_features
        out_features = self.out_features
        bias_ops = 1 if self.bias is not None else 0
        total_ops = batch_size * (in_features + bias_ops) * out_features
        flops_dict[self] = total_ops

    # 注册 hook
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    # 前向一次
    device = next(model.parameters()).device
    inputs = [x.to(device) for x in inputs]
    with torch.no_grad():
        _ = model(*inputs)  # 多输入 forward

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算 FLOPs
    total_flops = sum(flops_dict.values())

    if verbose:
        print(f"Total Params: {total_params:,} ({total_params/1e6:.3f} M)")
        print(f"Total FLOPs: {total_flops:,} ({total_flops/1e9:.3f} G)")

    # 移除 hook
    for h in hooks:
        h.remove()

    return total_params, total_flops

# ===== 使用示例 =====
if __name__ == "__main__":
    model = HDR_HPB(6, 64, 'haar')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 你的输入
    B, C, H, W = 1, 6, 128, 128
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    x3 = torch.randn(B, C, H, W)

    count_flops_params_multi_input(model, [x1, x2, x3])