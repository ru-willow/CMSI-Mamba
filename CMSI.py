import torch
import torch.nn as nn
import torch.nn.functional as F
import causal_conv1d
from einops import repeat
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn, selective_scan_fn

# from dct_module import DeformableAttention, DeformableAttention2
# from g_module import Gms, PLMS_Loss, Gpan, PLPAN_Loss
# import tf
# import mobilenetV3 as Mb
# from DTM import Mamba
# from mamba_ssm.modules.mamba_simple import Mamba
# from DMT_2 import *
import math
from mamba_ssm import Mamba
from typing import Union
import numbers
from einops import rearrange

# from mmpretrain.models import build_2d_sincos_position_embedding
# device = torch.device("cuda: 2" if torch.cuda.is_available() else "cpu")


class PatchEmbed(
    nn.Module
):  #     img_size=64, patch_size=8, stride=8, in_chans=channel_pan, embed_dim=64
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,  # 1
        embed_dim=768,  # 64
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        self.img_size = img_size  # 64
        self.patch_size = patch_size  # 8
        # self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 1*64*64 -> 64*8*8

        x_out = x.flatten(2).transpose(
            1, 2
        )  # BCHW -> BNC  B*64*8*8 -> B*64(c)*64(h*w) -> B*64(h*w)*64(c) = b,l,d
        y_out = x.transpose(2, 3).flatten(2).transpose(1, 2)  # 按行展开
        x_out = self.norm(x_out)
        y_out = self.norm(y_out)
        return x_out, y_out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        # in_channels：输入张量的channels数，out_channels：输出通道数，kernel_size：卷积核大小，stride：步长大小，Padding：即所谓的图像填充
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 防止数据在进行Relu之前因为数据过大而导致网络性能的不稳定
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        out = F.relu(out)

        return out


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x):
        out = F.dropout(x, p=0.2, training=self.training)
        return out


class CMCI_Mamba(nn.Module):
    def __init__(self, dim, use_num=2):
        super(CMCI_Mamba, self).__init__()

        self.norm1 = LayerNorm(dim, "with_bias")
        self.norm2 = LayerNorm(dim, "with_bias")
        self.num = use_num

        self.Fusion_mb_a = nn.ModuleList([Mamba(d_model=64) for _ in range(self.num)])
        self.Fusion_mb_b = nn.ModuleList([Mamba(d_model=64) for _ in range(self.num)])

    def forward(self, Ms_feature, Pan_feature):
        # Ms_feature (B,C,N)
        # Pan_feature (B,C,N)
        # Ms_feature = self.norm1(Ms_feature)
        # Pan_feature = self.norm2(Pan_feature)
        B, C, L = Ms_feature.shape
        Ms_first_half = Ms_feature[:, : C // 2, :]
        Ms_second_half = Ms_feature[:, C // 2 :, :]
        Pan_first_half = Pan_feature[:, : C // 2, :]
        Pan_second_half = Pan_feature[:, C // 2 :, :]
        cross_fusion1 = torch.cat((Ms_first_half, Pan_second_half), dim=1)
        cross_fusion2 = torch.cat((Pan_first_half, Ms_second_half), dim=1)

        for layer in self.Fusion_mb_a:
            ms_resual = Ms_feature
            cross_fusion_a = torch.cat((cross_fusion1, cross_fusion2), dim=0)
            cross_fusion_a = layer(cross_fusion_a)  # B*128*64
            cross_fusion1, cross_fusion2 = torch.split(cross_fusion_a, B, dim=0)
            Ms_feature = F.relu((cross_fusion1 + cross_fusion2) / 2 + ms_resual)

        odd_indices = torch.arange(1, C, 2)  # 奇数通道索引
        even_indices = torch.arange(0, C, 2)  # 偶数通道索引

        Ms_feature_odd = Ms_feature[:, odd_indices, :]
        Pan_feature_even = Pan_feature[:, even_indices, :]
        new_channels = Ms_feature_odd.shape[1] + Pan_feature_even.shape[1]
        cross_fusion3 = torch.empty(
            B, new_channels, L, device="cuda", dtype=Pan_feature.dtype
        )
        cross_fusion3[:, ::2, :] = Pan_feature_even
        cross_fusion3[:, 1::2, :] = Ms_feature_odd
        Ms_feature_even = Ms_feature[:, even_indices, :]
        Pan_feature_odd = Pan_feature[:, odd_indices, :]
        cross_fusion4 = torch.empty(
            B, new_channels, L, device="cuda", dtype=Ms_feature.dtype
        )
        cross_fusion4[:, ::2, :] = Ms_feature_even
        cross_fusion4[:, 1::2, :] = Pan_feature_odd

        for layer in self.Fusion_mb_b:
            pan_resual = Pan_feature
            cross_fusion_b = torch.cat((cross_fusion3, cross_fusion4), dim=0)
            cross_fusion_b = layer(cross_fusion_b)  # B*128*64
            cross_fusion3, cross_fusion4 = torch.split(cross_fusion_b, B, dim=0)
            Pan_feature = F.relu((cross_fusion3 + cross_fusion4) / 2 + pan_resual)

        return Ms_feature, Pan_feature


class CrossSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fase_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        super(CrossSSM, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.d_conv = d_conv
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        # 输出: (d_inner,L_out)  L_out = ((L-K+2*padding) // stride) + 1  # kernel_size卷积核长度
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        # ↓ 这里的输出神经元是d_inner，不输出z，用另一个的z
        self.in_proj1 = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )
        self.conv1d1 = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj1 = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj1 = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.out_proj1 = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        # 初始化A矩阵，假设d_state=10,A[0] = [1,2...9,10] 将这个一维向量重复d_inner次得到A：形状(d_inner , d_state)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        # 将A设置为可学习的参数
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        A1 = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log1 = torch.log(A1)  # Keep A_log in fp32
        # 将A设置为可学习的参数
        self.A_log1 = nn.Parameter(A_log1)
        self.A_log1._no_weight_decay = True
        self.D1 = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D1._no_weight_decay = True

    def forward(self, x, x1):
        b, seqlen, d = x.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )  # 最后变成B，D，L是为了接下来的一维卷积
        x1 = rearrange(
            self.in_proj1.weight @ rearrange(x1, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        A = -torch.exp(self.A_log.float())
        A1 = -torch.exp(self.A_log1.float())
        L = xz.shape[-1]
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        # if xz1.stride(-1) != 1:
        #     xz1 = xz1.contiguous()
        conv1d_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")
        conv1d_weight1 = rearrange(self.conv1d1.weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        # x1, z1 = xz1.chunk(2, dim=1)
        conv1d_bias = (
            self.conv1d.bias.contiguous() if self.conv1d.bias is not None else None
        )
        conv1d_bias1 = (
            self.conv1d1.bias.contiguous() if self.conv1d1.bias is not None else None
        )
        conv1d_out = causal_conv1d_fn(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        conv1d_out1 = causal_conv1d_fn(
            x1, conv1d_weight1, conv1d_bias1, None, None, None, True
        )
        x_dbl = F.linear(
            rearrange(conv1d_out, "b d l -> (b l) d"), self.x_proj.weight
        )  # (bl d)
        x_dbl1 = F.linear(
            rearrange(conv1d_out1, "b d l -> (b l) d"), self.x_proj1.weight
        )
        delta = rearrange(
            self.dt_proj.weight @ x_dbl[:, : self.dt_rank].t(), "d (b l) -> b d l", l=L
        )
        delta1 = rearrange(
            self.dt_proj1.weight @ x_dbl1[:, : self.dt_rank].t(),
            "d (b l) -> b d l",
            l=L,
        )
        B = x_dbl[:, self.dt_rank : self.dt_rank + self.d_state]
        B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        B1 = x_dbl1[:, self.dt_rank : self.dt_rank + self.d_state]
        B1 = rearrange(B1, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        C = x_dbl[:, -self.d_state :]
        C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        C1 = x_dbl1[:, -self.d_state :]
        C1 = rearrange(C1, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        D = self.D.float().contiguous()
        D1 = self.D1.float().contiguous()
        out = selective_scan_fn(
            conv1d_out, delta, A, B, C, D, z, delta_bias=None, delta_softplus=True
        )
        out1 = selective_scan_fn(
            conv1d_out1, delta1, A1, B1, C1, D1, z, delta_bias=None, delta_softplus=True
        )
        result = out + out1
        output = F.linear(
            rearrange(result, "b d l -> b l d"),
            self.out_proj.weight,
            self.out_proj.bias,
        )

        return output


class FECI_Mamba(nn.Module):
    def __init__(self, dim, use_num):
        super(FECI_Mamba, self).__init__()

        self.norm1 = LayerNorm(dim, "with_bias")
        self.norm2 = LayerNorm(dim, "with_bias")
        self.norm3 = LayerNorm(dim, "with_bias")
        self.num = use_num

        self.Fusion_mb_a = nn.ModuleList(
            [CrossSSM(d_model=64) for _ in range(self.num)]
        )
        # self.Fusion_mb_2a = nn.ModuleList([Mamba(d_model=64) for _ in range(2)])
        self.Fusion_mb_b = nn.ModuleList(
            [CrossSSM(d_model=64) for _ in range(self.num)]
        )
        # self.Fusion_mb_2b = nn.ModuleList([Mamba(d_model=64, ) for _ in range(2)])
        self.cmci = CMCI_Mamba(dim, use_num=2)
        self.norm = nn.LayerNorm(64)

    def forward(self, cross_fusion_a, cross_fusion_b, cross_fusion):
        #
        # cross_fusion_a = self.norm1(cross_fusion_a)
        # cross_fusion_b = self.norm2(cross_fusion_b)
        cross_fusion = self.norm3(cross_fusion)

        channel_Inter_1a, channel_Inter_2a = self.cmci(cross_fusion_a, cross_fusion)
        channel_Inter_1b, channel_Inter_2b = self.cmci(cross_fusion_b, cross_fusion)

        for layer in self.Fusion_mb_a:
            a_resual = cross_fusion_a
            channel_inter_a = self.norm((channel_Inter_1a + channel_Inter_2a) / 2)
            channel_Inter_1a = layer(channel_inter_a, cross_fusion)
            cross_fusion_a = F.relu(channel_Inter_1a + a_resual)
        # for layer in self.Fusion_mb_2a:
        #     channel_Inter_2a = layer(channel_Inter_2a)
        # cross_fusion_a = F.relu((channel_Inter_1a + channel_Inter_2a) / 2 + cross_fusion_a)

        for layer in self.Fusion_mb_b:
            b_resual = cross_fusion_b
            channel_inter_b = (channel_Inter_1b + channel_Inter_2b) / 2
            channel_Inter_1b = layer(channel_inter_b, cross_fusion)
            cross_fusion_b = F.relu(channel_Inter_1b + b_resual)
        # for layer in self.Fusion_mb_2b:
        #     channel_Inter_2b = layer(channel_Inter_2b)
        # cross_fusion_b = F.relu((channel_Inter_1b + channel_Inter_2b) / 2 + cross_fusion_b)
        out = (cross_fusion_a + cross_fusion_b) / 2
        # out = torch.cat((cross_fusion_a, cross_fusion_b), dim=1)

        return out


class Net(nn.Module):
    def __init__(
        self,
        channel_ms,
        channel_pan,
        class_num,
        if_abs_pos_embed=True,
        pe_type="learnable",
    ):
        super(Net, self).__init__()

        self.hide_line = 64
        # self.cam = CAM()
        # self.fc11 = nn.Linear(1 * 1 * 1024, 64)
        self.fc2 = nn.Linear(self.hide_line * 1, class_num)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 128, self.hide_line)
        self.fc3 = nn.Linear(64, 128)
        self.proj_norm_ms = LayerNorm(64, "with_bias")
        self.proj_norm_pan = LayerNorm(64, "with_bias")
        self.if_abs_pos_embed = if_abs_pos_embed
        self.Ms_mb1 = nn.ModuleList([Mamba(d_model=64) for _ in range(2)])
        self.Pan_mb1 = nn.ModuleList([Mamba(d_model=64) for _ in range(2)])

        self.cmci = CMCI_Mamba(64, use_num=2)
        self.feci = FECI_Mamba(64, use_num=1)
        # self.feci1 = FECI_Mamba1(64, use_num=2)

        self.out_proj_norm = LayerNorm(self.hide_line, "with_bias")

        self.PE = PatchEmbed(
            img_size=64, patch_size=8, stride=8, in_chans=channel_pan, embed_dim=64
        )
        if pe_type == "learnable":
            self.channel_pos_embed = nn.Parameter(torch.zeros(1, 128 + 0, 64))
            self.patch_pos_embed = nn.Parameter(torch.zeros(1, 64 + 0, 64))
        # elif pe_type == 'sine':
        #     self.pos_embed = build_2d_sincos_position_embedding(
        #         patches_resolution=self.patch_resolution,
        #         embed_dims=self.embed_dims,
        #         temperature=10000,
        #         cls_token=False)
        self.Res1 = ResBlk(channel_ms, 64, stride=2)
        self.Res2 = ResBlk(64, 128, stride=1)
        self.Res3 = ResBlk(128, 256, stride=1)
        self.Res4 = ResBlk(256, 512, stride=1)
        self.Res5 = ResBlk(512, 128, stride=1)

        self.R1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2,2)
        )
        self.R2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2,2)
        )

        self.R3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            # nn.MaxPool2d(2,2)
        )

    def forward(self, x, y, label=None):
        # Pre-process Image Feature
        Ms = self.Res1(x)  # 4*16*16 -> 64*8*8
        Ms = self.Res2(Ms)  # 64*8*8 -> 128*8*8 128*8*8 = 128*64
        # Ms = self.Res3(Ms)
        # Ms = self.Res4(Ms)
        # Ms = self.Res5(Ms)
        Ms_feature = Ms.flatten(
            2
        )  # B*128*64 = B,L,D  每个时间步是一个通道，每个通道上8*8=64个 d_model=64
        Ms_forward = Ms_feature
        B, C, L = Ms_forward.shape
        rand_index = torch.randperm(C)
        Ms_shuffle = Ms[:, rand_index, :].flatten(2)  # B*128*64
        Ms_reverse = Ms_forward.flip([1]).flatten(2)  # B*128*64
        # if self.if_abs_pos_embed is not None:
        #     Ms_forward = Ms_forward + self.channel_pos_embed.to(device=x.device)
        for layer in self.Ms_mb1:
            ms_residual = Ms_feature
            ms_inputs = [Ms_forward, Ms_shuffle, Ms_reverse]
            ms_inputs = torch.cat(ms_inputs, dim=0)  # 3B*128*64
            ms_inputs = self.proj_norm_ms(ms_inputs)
            ms_inputs = layer(ms_inputs)
            Ms_forward, Ms_shuffle, Ms_reverse = torch.split(ms_inputs, B, dim=0)
            rand_index = torch.argsort(rand_index)
            Ms_shuffle = Ms_shuffle[:, rand_index]
            Ms_reverse = torch.flip(Ms_reverse, [1])
            Ms_feature = F.relu(
                (Ms_forward + Ms_shuffle + Ms_reverse) / 3 + ms_residual
            )  # B*128*64

        # y = self.R1(y)
        # y = self.R2(y)
        # y = self.R3(y)
        # y = self.R4(y)
        Pan_horizontal, Pan_vertical = self.PE(
            y
        )  # 1*64*64 -> 64*8*8  B*64*64 = B,L,D  每个时间步是一个像素，每个像素上64个通道
        Pan_feature = Pan_horizontal
        Pan_horizontal_reverse = Pan_horizontal.flip([1])  # B*64*64
        Pan_vertical_reverse = Pan_vertical.flip([1])  # B*64*64
        if self.if_abs_pos_embed is not None:
            Pan_horizontal = Pan_horizontal + self.patch_pos_embed.to(device="cuda")

        for layer in self.Pan_mb1:
            pan_residual = Pan_feature
            pan_inputs = [
                Pan_horizontal,
                Pan_vertical,
                Pan_horizontal_reverse,
                Pan_vertical_reverse,
            ]
            pan_inputs = torch.cat(pan_inputs, dim=0)
            pan_inputs = self.proj_norm_pan(pan_inputs)
            pan_inputs = layer(pan_inputs)
            (
                Pan_horizontal,
                Pan_vertical,
                Pan_horizontal_reverse,
                Pan_vertical_reverse,
            ) = torch.split(pan_inputs, B, dim=0)
            Pan_horizontal_reverse = torch.flip(Pan_horizontal_reverse, [1])

            Pan_feature = F.relu(
                (
                    Pan_horizontal
                    + Pan_vertical
                    + Pan_horizontal_reverse
                    + Pan_vertical_reverse
                )
                / 4
                + pan_residual
            )  # B*64*64
        Pan_feature = self.fc3(Pan_feature).transpose(1, 2)  # B*128*64

        cross_fusion_a, cross_fusion_b = self.cmci(Ms_feature, Pan_feature)  # B*128*64

        cross_fusion = cross_fusion_a + cross_fusion_b
        out = self.feci(cross_fusion_a, cross_fusion_b, cross_fusion)
        # out = self.feci1(cross_fusion_a, cross_fusion_b)
        out = out.contiguous().view(x.size(0), -1)
        # x1 = cross_fusion_a.contiguous().view(x.size(0), -1)
        # y1 = cross_fusion_b.contiguous().view(y.size(0), -1)
        # out = x1 + y1

        out = F.relu(self.out_proj_norm(self.fc1(out)))  # 8*8*128 -> 64
        out = self.dropout(out)
        output = self.fc2(out)  # 64 -> Classes_num
        # Pan_x = self.fc2(Pan_x)
        # MS_x = self.fc2(MS_x)

        # return x, Pan_x, MS_x
        if label != None:
            return output, self.get_loss(x, y, output, label)

        return output

    def get_loss(self, x, y, output, label):

        # weight_list = [20,60,20,20,40,20,5,15,15,5,10,25]
        # weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # weight_tensor = torch.tensor(weight_list).cuda()
        # criterion = InitLabelSmoothCE(weight=weight_tensor)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        # loss = 0.01*loss1 + 0.99*loss2
        # loss = loss2
        return loss



if __name__ == "__main__":

    pan = torch.randn(2, 1, 64, 64).to("cuda")
    ms = torch.randn(2, 4, 16, 16).to("cuda")
    # mshpan = torch.randn(2, 1, 64, 64)
    Net = Net(
        4,
        1,
        7,
    ).to("cuda")
    out_result = Net(ms, pan)
    
    print(out_result)
    print(out_result.shape)
