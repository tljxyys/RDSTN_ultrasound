import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as checkpoint
from argparse import Namespace
from typing import Optional

from models import register


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw] window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        scale_factor (float, optional): If not None, add a scale adaptive module. Default: None
    """

    def __init__(self, dim, num_heads, scale_factor, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.scale_factor = scale_factor
        self.scale_embed_network = nn.Sequential(*[
            nn.Linear(1, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        ])

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # scale embedding
        if self.scale_factor is not None:
            self.scale_factor = self.scale_factor.unsqueeze(1).unsqueeze(1).expand(B, H*W, 1)
            scale_embed = self.scale_embed_network(self.scale_factor)
            x = shortcut + self.drop_path(x) + scale_embed

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_lff (bool): If Ture, concatenate the first and last layer and add a local feature fusion. Default: True
    """

    def __init__(self, dim, depth, num_heads, window_size, scale_factor, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, use_lff=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                scale_factor=scale_factor if (i % 2 == 0) else None)
            for i in range(depth)])

        # Local Feature Fusion
        if use_lff:
            self.LFF = nn.Sequential(*[
                nn.Conv2d(self.dim * 2, self.dim, 1, padding=0, stride=1),
            ])
        else:
            self.LFF = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        assert self.depth % 2 == 0, "depth must be even number"
        B, _, _ = x.shape
        output = [x]
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        output.append(x)
        if self.LFF is not None:
            x = torch.cat(output, 2).view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.LFF(x).view(B, -1, H * W).permute(0, 2, 1).contiguous()

        return x, H, W


class LinearEmbed(nn.Module):
    """
    2D Image to Linear Embedding
    """

    def __init__(self, in_c=3, embed_dim=96, kSize=3, norm_layer=None, patch_size=None):
        super().__init__()
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if patch_size is not None:
            patch_size = (patch_size, patch_size)
            self.patch_size = patch_size
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.patch_size = None
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=kSize, padding=(kSize - 1) // 2, stride=1)

    def forward(self, x):
        _, _, H, W = x.shape
        # padding
        if self.patch_size is not None:
            pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
            if pad_input:
                # to pad the last 3 dimensions,
                # (W_left, W_right, H_top,H_bottom, C_front, C_back)
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, self.patch_size[0] - H % self.patch_size[0], 0, 0))
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class RDSTN(nn.Module):  # Residual Swin Transformer Network

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.in_chans = args.in_chans
        self.num_layers = len(args.depths)
        self.embed_dim = args.embed_dim
        self.out_dim = args.out_dim
        self.patch_norm = args.patch_norm
        self.H = args.input_size[0]
        self.W = args.input_size[1]
        self.scale_factor = args.scale_factor

        self.num_features = int(args.embed_dim)
        self.mlp_ratio = args.mlp_ratio
        self.use_lff = args.use_lff
        self.use_gff = args.use_gff

        self.embed = LinearEmbed(
            patch_size=args.patch_size, in_c=args.out_dim, embed_dim=args.embed_dim,
            norm_layer=args.norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=args.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, args.drop_path_rate, sum(args.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(args.embed_dim),
                                depth=args.depths[i_layer],
                                num_heads=args.num_heads[i_layer],
                                window_size=args.window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=args.qkv_bias,
                                drop=args.drop_rate,
                                attn_drop=args.attn_drop_rate,
                                drop_path=dpr[sum(args.depths[:i_layer]):sum(args.depths[:i_layer + 1])],
                                norm_layer=args.norm_layer,
                                use_checkpoint=args.use_checkpoint,
                                scale_factor=self.scale_factor,
                                use_lff=self.use_lff)
            self.layers.append(layers)

        self.norm = args.norm_layer(self.num_features)
        self.apply(_init_weights)

        self.mapping = nn.Sequential(*[
            nn.Conv2d(self.in_chans, self.out_dim, args.kSize, padding=(args.kSize - 1) // 2, stride=1)])
        self.channel_adjust_network = nn.Sequential(*[
            nn.Conv2d(self.embed_dim, self.out_dim, args.kSize, padding=(args.kSize - 1) // 2, stride=1)])
        # Global Feature Fusion
        if self.use_gff:
            self.GFF = nn.Sequential(*[
                nn.ConvTranspose2d(self.embed_dim * (self.num_layers + 1), self.out_dim, kernel_size=1),
            ])
        else:
            self.GFF = None

    def forward(self, x):
        x = self.mapping(x)
        tmp = x  # skip connection
        x, H, W = self.embed(x)  # x: [B, L, C]
        B, _, _ = x.shape
        x = self.pos_drop(x)
        RSTBs_out = []
        for layer in self.layers:
            RSTBs_out.append(x)
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # [B, L, C]
        RSTBs_out.append(x)

        # Global Feature Fusion
        if self.GFF is not None:
            x = torch.cat(RSTBs_out, 2)
            # [batch, height*width, channel] => [batch, channel, height, width]
            x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)
            return self.GFF(x) + tmp
        else:
            x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)
            x = self.channel_adjust_network(x)
            return x + tmp


@register('rdstn-baseline')
def make_RDSTN_baseline(scale_factor=None, input_size=(48, 48), patch_size=None, in_chans=3, out_dim=64, embed_dim=96,
                        depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=7, mlp_ratio=4., qkv_bias=True,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                        use_checkpoint=False, kSize=3, use_lff=True, use_gff=True):
    """
    Args:
        scale_factor (float): If not None, add a scale adaptive module. Default: None
        input_size (tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size. Default: None
        in_chans (int): Number of input image channels. Default: 3
        out_dim (int): Output channel dimension. Default: 64
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        kSize (int): Kernel size in the last convolution layer
        use_lff (bool): If Ture, concatenate the first and last layer and add a local feature fusion. Default: True
        use_gff (bool): If Ture, concatenate the first and last block and add a global feature fusion. Default: True
    """

    args = Namespace()
    args.scale_factor = scale_factor
    args.input_size = input_size
    args.out_dim = out_dim
    args.embed_dim = embed_dim
    args.depths = depths
    args.num_heads = num_heads

    args.patch_size = patch_size
    args.in_chans = in_chans
    args.window_size = window_size
    args.mlp_ratio = mlp_ratio
    args.qkv_bias = qkv_bias
    args.drop_rate = drop_rate
    args.attn_drop_rate = attn_drop_rate
    args.drop_path_rate = drop_path_rate
    args.norm_layer = norm_layer
    args.patch_norm = patch_norm
    args.use_checkpoint = use_checkpoint
    args.kSize = kSize

    args.use_lff = use_lff
    args.use_gff = use_gff

    return RDSTN(args)


@register('rdstn')
def make_RDSTN(scale_factor=None, input_size=(48, 48), patch_size=None, in_chans=3, out_dim=64, embed_dim=96,
                        depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), window_size=7,
                        mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                        drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                        use_checkpoint=False, kSize=3, use_lff=True, use_gff=True):
    """
    Args:
        scale_factor (float): If not None, add a scale adaptive module. Default: None
        input_size (tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size. Default: None
        in_chans (int): Number of input image channels. Default: 3
        out_dim (int): Output channel dimension. Default: 64
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        kSize (int): Kernel size in the last convolution layer
        use_lff (bool): If Ture, concatenate the first and last layer and add a local feature fusion. Default: True
        use_gff (bool): If Ture, concatenate the first and last layer and add a local feature fusion. Default: True
    """

    args = Namespace()
    args.scale_factor = scale_factor
    args.input_size = input_size
    args.out_dim = out_dim
    args.embed_dim = embed_dim
    args.depths = depths
    args.num_heads = num_heads

    args.patch_size = patch_size
    args.in_chans = in_chans
    args.window_size = window_size
    args.mlp_ratio = mlp_ratio
    args.qkv_bias = qkv_bias
    args.drop_rate = drop_rate
    args.attn_drop_rate = attn_drop_rate
    args.drop_path_rate = drop_path_rate
    args.norm_layer = norm_layer
    args.patch_norm = patch_norm
    args.use_checkpoint = use_checkpoint
    args.kSize = kSize

    args.use_lff = use_lff
    args.use_gff = use_gff

    return RDSTN(args)


if __name__ == '__main__':
    MyConvNet = make_RDSTN_baseline(use_gff=False, use_lff=False)
    input_ = torch.randn(1, 3, 48, 48)
    print("input size is {}".format(input_.shape))
    print("output size is {}".format(MyConvNet(input_).shape))
