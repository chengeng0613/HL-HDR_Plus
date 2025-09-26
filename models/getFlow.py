import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import warp

div_size = 16
div_flow = 20.0


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False,
                         recompute_scale_factor=True)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(64, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1)
        )

    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(82, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, flow0):
        f0_warp = warp(f0, flow0)

        f_in = torch.cat([f0_warp, f1, flow0], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]

        return up_flow0


class getFlow(nn.Module):
    def __init__(self):
        super(getFlow, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward_flow_mask(self, img0_c, img1_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
        int(div_size * np.ceil(h * scale_factor / div_size)), int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])

        up_flow0_4 = self.decoder(f0_4, f1_4, up_flow0_5)
        up_flow0_3 = self.decoder(f0_3, f1_3, up_flow0_4)
        up_flow0_2 = self.decoder(f0_2, f1_2, up_flow0_3)
        up_flow0_1 = self.decoder(f0_1, f1_1, up_flow0_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h

        return up_flow0_1

    def forward(self, img0_c, img1_c, scale_factor=0.5, refine=True):
        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        flow = self.forward_flow_mask(img0_c, img1_c, scale_factor=scale_factor)

        return flow


# --- 你已有的模型定义部分省略重复，此处直接接续在模型代码之后 ---

if __name__ == "__main__":
    # 模拟两个输入图像：大小为 256x256，6 个通道（3 个为线性域，3 个为 LDR）
    B, C, H, W = 1, 6, 256, 256
    img0_c = torch.randn(B, C, H, W)
    img1_c = torch.randn(B, C, H, W)

    # 创建模型
    model = getFlow()

    # 前向推理
    with torch.no_grad():  # 关闭梯度计算，节省内存
        flow = model(img0_c, img1_c)

    # 输出结果形状
    print("输出光流形状:", flow.shape)  # 应该是 [1, 2, 256, 256]
