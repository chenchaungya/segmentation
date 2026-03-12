"""
Paper:      
Url:
Create by:Chuang Chen
Date:       2026/1/20
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, SegHead
#from .model_registry import register_model, aux_models, detail_head_models

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


#@register_model(aux_models, detail_head_models)
class DBBGNet(nn.Module):
    def __init__(self, num_class=19, n_channel=3, encoder_type='stdc2', head_channels=128, augment=True,
                    act_type='relu'):
        super().__init__()

        self.augment = augment

        repeat_times_hub = {'stdc1': [1,1,1], 'stdc2': [3,4,2]}
        if encoder_type not in repeat_times_hub.keys():
            raise ValueError('Unsupported encoder type.\n')
        repeat_times = repeat_times_hub[encoder_type]

        self.stage1 = ConvBNAct(n_channel, 32, 3, 2)
        self.stage2 = ConvBNAct(32, 64, 3, 2)
        self.stage3 = self._make_stage(64, 256, repeat_times[0], act_type)
        self.stage4 = self._make_stage(256, 512, repeat_times[1], act_type)
        self.stage5 = self._make_stage(512, 1024, repeat_times[2], act_type)
        self.eappm = EAPPM(1024, 96, 256)

        self.spatial_path = SpatialPath(n_channel, 128, act_type=act_type)


        self.bfam = BFAM(256+256, 128, act_type)


        # Prediction Head
        if self.augment:
            self.seg_head = segmenthead(planes * 2, head_channels, num_class)
            self.boundary_head = segmenthead(planes * 2, 64, 1)


    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]

        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x, is_training=False):
        size = x.size()[2:]

        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        if self.use_boundary_head:
            boundary = self.boundary_head(x3)

        x4 = self.stage4(x3)

        x5 = self.stage5(x4)
        x5 = self.eappm(x5)
        x5=np.concatenate(x5,x3)
        y=self.spatial_path(x)
        if self.use_auxseg_head:
            auxseg = self.seg_head(y)

        x4 = self.arm4(x4)
        x4 = self.conv4(x4)
        x4 += x5
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.bfam(y, x5)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)


        if self.use_boundary_head and self.use_auxseg_head and is_training:
            return x, (boundary, auxseg)
        else:
            return x


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=algc)

        return out

class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        if stride not in [1, 2]:
            raise ValueError(f'Unsupported stride: {stride}\n')

        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels//2, 1)
        self.block2 = ConvBNAct(out_channels//2, out_channels//4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels//4, out_channels//8, 3)
        self.block4 = ConvBNAct(out_channels//8, out_channels//8, 3)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        if self.stride == 2:
            x1 = self.pool(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x1, x2, x3, x4], dim=1)

class EAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                        BatchNorm(inplanes, momentum=bn_mom),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                        )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                        BatchNorm(inplanes, momentum=bn_mom),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                        )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                        BatchNorm(inplanes, momentum=bn_mom),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                        )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        BatchNorm(inplanes, momentum=bn_mom),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                        )

        self.scale0 = nn.Sequential(
                BatchNorm(inplanes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
            )

        self.scale_process = nn.Sequential(
                BatchNorm(branch_planes * 4, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
                nn.Conv2d(branch_planes * 4, branch_planes, kernel_size=1, bias=False),

            )

        self.compression = nn.Sequential(
                BatchNorm(branch_planes * 5, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
            )

        self.shortcut = nn.Sequential(
                BatchNorm(inplanes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
            )

        self.arm = AttentionRefinementModule(outplanes)

    def forward(self, x):
            width = x.shape[-1]
            height = x.shape[-2]
            scale_list = []

            x_ = self.scale0(x)

            scale_list.append(torch.cat([F.interpolate(self.scale1(x), size=[height, width],
                                            mode='bilinear', align_corners=algc), x_], dim=1))
            scale_list.append(torch.cat([F.interpolate(self.scale2(x), size=[height, width],
                                                       mode='bilinear', align_corners=algc), x_], dim=1))
            scale_list.append(torch.cat([F.interpolate(self.scale3(x), size=[height, width],
                                                       mode='bilinear', align_corners=algc), x_], dim=1))
            scale_list.append(torch.cat([F.interpolate(self.scale4(x), size=[height, width],
                                                       mode='bilinear', align_corners=algc), x_], dim=1))

            scale_out = self.scale_process(scale_list[0]+scale_list[1]+scale_list[2]+scale_list[3])

            out = self.compression(scale_out+x_)
            arm_out = self.arm(out)

            out = F.interpolate(arm_out+self.shortcut(x), size=[4*height, 4*width],
                                            mode='bilinear', align_corners=algc)
            return out



class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNAct(channels, channels, 1, act_type='sigmoid')

    def forward(self, x):
        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv(x_pool)
        x = x * x_pool

        return x
class BoundaryRefinementModule(nn.Module):

    def __init__(self, in_channels):
        super(BoundaryRefinementModule, self).__init__()
        # Conv3×3 + ReLU
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        # Conv3×3 + ReLU
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)

class SpatialPath(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 1, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(in_channels, out_channels, 3, 1, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 1, act_type=act_type),

        )

        self.brm = BoundaryRefinementModule(out_channels)
    def forward(self, x):

        return self.brm(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class BFAM(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, reduction=4):
        super().__init__()
        self.conv_semantic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, bias=False),
            BatchNorm(out_channels)
        )

        self.balance = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, y):
        semantic_x = self.conv_semantic(x)
        spatial_y = self.conv_spatial(y)
        z = torch.cat([semantic_x, spatial_y], dim=1)
        z = self.balance(z)
        n, c, h, w = z.size()
        x_h = self.pool_h(z)
        x_w = self.pool_w(z).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_1 = semantic_x *(1+a_w * a_h)
        out_2 = spatial_y * (1+a_w * a_h)
        out = self.conv3(out1+out_2)

        return out
class BFAM(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        inter = max(in_ch // 4, 32)
        # Semantic
        self.sem_gap = nn.AdaptiveAvgPool2d(1)
        self.sem_fc = nn.Sequential(
            nn.Conv2d(in_ch, inter, 1, bias=False), nn.BatchNorm2d(inter), nn.ReLU(),
            nn.Conv2d(inter, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch)
        )
        # Spatial
        self.spa_conv = nn.Sequential(
            nn.Conv2d(1, inter, 3, 1, 1, bias=False), nn.BatchNorm2d(inter), nn.ReLU(),
            nn.Conv2d(inter, 1, 3, 1, 1, bias=False), nn.BatchNorm2d(1)
        )
        # Fusion
        self.fusion = nn.Conv2d(in_ch, in_ch, 1, bias=False)

    def forward(self, x):

        sem = self.sem_gap(x)          # B,C,1,1
        sem = self.sem_fc(sem).sigmoid()

        spa = torch.mean(x, dim=1, keepdim=True)  # B,1,H,W
        spa = self.spa_conv(spa).sigmoid()
        sem_out = x * sem
        spa_out = x * spa

        out = self.fusion(sem_out + spa_out)
        return out


class BDGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BDGM, self).__init__()

        # Sobel X and Y
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)


        self.conv1x1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        if x.size(1) > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x


        edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)


        edges = torch.cat([edge_x, edge_y], dim=1)


        out = self.conv1x1(edges)


        out = self.upsample(out)

        return out