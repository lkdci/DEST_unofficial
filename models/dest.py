from typing import Union, Tuple, Optional, List
import math

import torch.nn as nn
import torch
from timm.models.layers import to_2tuple, trunc_normal_


class OverlapPatchEmbed(nn.Module):
    """
    DEST overlap patch embedding.
    diff with segformer:
        * BatchNorm instead of LayerNorm
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: Union[int, Tuple[int, int]] = 7,
                 stride: int = 4,
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride,
                      padding=(patch_size[0] // 2, patch_size[1] // 2), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.flatten_spatial = nn.Flatten(start_dim=2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = self.flatten_spatial(x)
        return x, H, W


class SimplifiedAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 qk_scale: Optional[float] = None,
                 sr_ratio: int = 1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} should be divided by num_heads {num_heads}."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.spatial_reduction = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio,
                          bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            self.flatten_spatial = nn.Flatten(start_dim=2)

        self.k = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)

        # TODO - with BN??
        self.proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, C, N = x.shape

        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)

        if self.sr_ratio > 1:
            x_ = x.reshape(B, C, H, W)
            x_ = self.spatial_reduction(x_)
            x_ = self.flatten_spatial(x_)
            k = self.k(x_).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 2, 3)
        else:
            k = self.k(x).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 2, 3)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.max(dim=-1, keepdims=False)[0]

        attn = attn.transpose(-1, -2)
        v = x.mean(dim=-1, keepdims=True).expand(B, C, self.num_heads).transpose(-1, -2)

        x = torch.matmul(attn, v).transpose(-1, -2)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_ratio: float):
        super().__init__()
        hidden_channels = int(in_channels * expansion_ratio)
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, C, N = x.size()
        x = x.reshape(B, C, H, W)
        x = self.dw_conv(x)
        x = torch.flatten(x, 2)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: float,
                 sr_ratio: int = 1,
                 qk_scale: Optional[float] = None):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(embed_dims)
        self.attn = SimplifiedAttention(embed_dims, num_heads=num_heads, sr_ratio=sr_ratio, qk_scale=qk_scale)
        self.norm2 = nn.BatchNorm1d(embed_dims)
        self.mix_ffn = MixFFN(embed_dims, embed_dims,  expansion_ratio=mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mix_ffn(self.norm2(x), H, W)
        return x


class EncoderStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int,
                 stride: int,
                 num_blocks: int,
                 num_heads: int,
                 mlp_ratio: float,
                 sr_ratio: int):
        super().__init__()
        self.path_embed = OverlapPatchEmbed(in_channels, out_channels, patch_size=patch_size, stride=stride)

        self.blocks = nn.ModuleList([
            Block(
                embed_dims=out_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x, H, W = self.path_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        return x, H, W


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 width_list: List[int],
                 patch_sizes: List[int],
                 strides_list: List[int],
                 num_blocks: List[int],
                 num_heads: List[int],
                 mlp_ratios: List[float],
                 sr_ratios: List[int],
                 is_out_feature_list: List[bool]):
        super().__init__()
        self.is_out_feature_list = is_out_feature_list
        self.width_list = width_list

        num_stages = len(width_list)
        self.stages = nn.ModuleList()

        for i in range(num_stages):
            self.stages.append(
                EncoderStage(
                    in_channels, out_channels=width_list[i], patch_size=patch_sizes[i], stride=strides_list[i],
                    num_blocks=num_blocks[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], sr_ratio=sr_ratios[i]
            ))
            in_channels = width_list[i]

    def encoder_out_channels(self) -> List[int]:
        """
        :return: num channels list of out feature maps.
        """
        return [ch for ch, is_out in zip(self.width_list, self.is_out_feature_list) if is_out]

    def forward(self, x) -> List[torch.Tensor]:
        B = x.size(0)
        out_list = []
        for i, stage in enumerate(self.stages):
            x, H, W = stage(x)
            x = x.reshape(B, -1, H, W)
            if self.is_out_feature_list[i]:
                out_list.append(x)
        return out_list


class UpFPNBlock(nn.Module):
    """
    Fuse features from the encoder. Upsample is done by bilinear upsample.
    """

    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.up_path = nn.Sequential(
            nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.skip_path = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
        )

    def forward(self, x, skip):
        x = self.up_path(x)
        skip = self.skip_path(skip)
        return x + skip


class DecoderFPN(nn.Module):
    def __init__(
        self,
        skip_channels_list: List[int],
    ):
        """
        """
        super().__init__()
        self.up_channels_list = skip_channels_list
        # Reverse order to up-bottom order, i.e [stage4_ch, stage3_ch, ... , stage1_ch]
        self.up_channels_list.reverse()
        # Remove last stage num_channels, as it is the input to the decoder.
        in_channels = self.up_channels_list.pop(0)

        self.up_stages = nn.ModuleList()
        for out_channels in self.up_channels_list:
            self.up_stages.append(UpFPNBlock(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, feats: List[torch.Tensor]):
        # Reverse order to up-bottom order, i.e [stage4_ch, stage3_ch, ... , stage1_ch]
        feats.reverse()
        # Remove last stage feature map, as it is the input to the decoder and not a skip connection.
        x = feats.pop(0)
        for up_stage, skip in zip(self.up_stages, feats):
            x = up_stage(x, skip)
        return x


class SegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.head(x)


class DepthHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.head(x)


class DepthNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 head_type: str,
                 in_channels: int,
                 width_list: List[int],
                 patch_sizes: List[int],
                 strides_list: List[int],
                 num_blocks: List[int],
                 num_heads: List[int],
                 mlp_ratios: List[float],
                 sr_ratios: List[int],
                 is_out_feature_list: List[bool]):
        super().__init__()
        self.encoder = Encoder(in_channels, width_list=width_list, patch_sizes=patch_sizes, strides_list=strides_list,
                               num_blocks=num_blocks, num_heads=num_heads, mlp_ratios=mlp_ratios, sr_ratios=sr_ratios,
                               is_out_feature_list=is_out_feature_list)
        self.decoder = DecoderFPN(skip_channels_list=self.encoder.encoder_out_channels())
        self.head = self.build_head(head_type=head_type, in_channels=self.encoder.encoder_out_channels()[0],
                                    num_classes=num_classes)

    def build_head(self, head_type: str, in_channels: int, num_classes: int):
        if head_type == "depth":
            return DepthHead(in_channels, num_classes)
        if head_type == "segmentation":
            return SegHead(in_channels, num_classes)
        raise ValueError(f"head_type: {head_type} is not supported.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class DEST_B0(DepthNet):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, head_type: str = "depth"):
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         head_type=head_type,
                         width_list=[32, 64, 128, 256],
                         patch_sizes=[7, 3, 3, 3],
                         strides_list=[4, 2, 2, 2],
                         num_blocks=[2, 2, 2, 2],
                         num_heads=[1, 2, 4, 8],
                         mlp_ratios=[4, 4, 4, 4],
                         sr_ratios=[8, 4, 2, 1],
                         is_out_feature_list=[True, True, True, True])


class B0Encoder(Encoder):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, head_type: str = "depth"):
        super().__init__(in_channels=in_channels,
                         width_list=[32, 64, 128, 256],
                         patch_sizes=[7, 3, 3, 3],
                         strides_list=[4, 2, 2, 2],
                         num_blocks=[2, 2, 2, 2],
                         num_heads=[1, 2, 4, 8],
                         mlp_ratios=[4, 4, 4, 4],
                         sr_ratios=[8, 4, 2, 1],
                         is_out_feature_list=[True, True, True, True])

if __name__ == '__main__':
    from utils.conversion_utils import onnx_simplify

    # m = OverlapPatchEmbed(embed_dim=64, patch_size=7, stride=4, in_channels=3)
    # path = "../checkpoints/overlap_path_embed.onnx"
    # x = torch.randn(1, 3, 512, 1024)

    # m = SimplifiedAttention(embed_dim=64, num_heads=2, spatial_reduction_ratio=4)
    # path = "../checkpoints/simplified_attention.onnx"
    # x = torch.randn(1, 64, 64 * 128)
    # x = (x, 128, 256)

    # m = MixFFN(64, 64, expansion_ratio=4)
    # path = "../checkpoints/mixffn.onnx"
    # x = torch.randn(1, 64, 64 * 128)
    # x = (x, 64, 128)

    # m = Block(64, 4, mlp_ratio=4, sr_ratio=4)
    # path = "../checkpoints/transformer_block.onnx"
    # x = torch.randn(1, 64, 64 * 128)
    # x = (x, 64, 128)

    # m = EncoderStage(32, 64, patch_size=3, stride=2, num_blocks=2, num_heads=4, mlp_ratio=4, sr_ratio=4)
    # path = "../checkpoints/transformer_stage.onnx"
    # x = torch.randn(1, 32, 64, 128)

    # m = B0Encoder()
    # path = "../checkpoints/b0_encoder.onnx"
    # x = torch.randn(1, 3, 32, 64)

    # m = DecoderFPN([32, 64, 128])
    # path = "../checkpoints/decoder_fpn.onnx"
    # x = [
    #     torch.randn(1, 32, 32, 32),
    #     torch.randn(1, 64, 16, 16),
    #     torch.randn(1, 128, 8, 8),
    # ]

    m = DEST_B0()
    path = "../checkpoints/dest-b0.onnx"
    x = torch.randn(1, 3, 512, 1024)

    # path = "../checkpoints/tmp.onnx"

    torch.onnx.export(m, x, path, opset_version=13)
    onnx_simplify(path, path)
