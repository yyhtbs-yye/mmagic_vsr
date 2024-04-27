from mmengine.model.weight_init import trunc_normal_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange

from basic_modules import MSA, MMA, GLUAttention, DropPath

from .basic_utils import window_partition, window_reverse, get_window_size


class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA).

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self, dim, num_heads,
                 window_size=(6, 8, 8),
                 shift_size=(0, 0, 0),
                 mut_attn=True, mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)                                                                
        self.mma = MMA(dim, window_size=self.window_size, 
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GLUAttention(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def _forward(self, x, attn_mask):

        B, D, H, W, C = x.shape

        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # Layer Norm input ``x``
        x = self.norm1(x)                                                                                   

        # The ``tensor`` is padded to match multiples of the window size, ensuring 
        # that attention can be applied evenly across the data.

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape

        # Depending on the ``shift_size``, the tensor may be cyclically shifted to align 
        # different parts of the data under the attention mechanism, enhancing the model's 
        # ability to learn from various spatial relationships. [SWIN]

        if any(i > 0 for i in shift_size): # if exist any i in shift_size > 0
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            attn_mask = None

        # The shifted and padded tensor is then split into smaller blocks or windows.
        x_windows = window_partition(shifted_x, window_size)  # ``x_windows`` is token of patches of size [B*nW, Wd*Wh*Ww, C]
        
        # The attention mechanism is applied within these windows.
        # Either self-attention or a combination of mutual and self-attention
        attn_windows = self.mma(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # After processing through the attention mechanism, the tensor 
        # blocks are reassembled.
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # If there was a cyclic shift applied initially, it is reversed to 
        # bring the tensor back to its original alignment.
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove the Paddings (may contains something?)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)

        return x

    def forward(self, x, attn_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            attn_mask: Attention mask for cyclic shift.
        """

        x = x + self._forward(x, attn_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
