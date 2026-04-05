import torch
import torch.nn as nn
from mmengine.registry import MODELS
from .resnet import Resnet1D
import torch.nn.init as init

# Ref: https://github.com/snap-research/SnapMoGen/blob/main/model/cnn_networks.py#L69
@MODELS.register_module()
class EncoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.embed = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(AttnBlock(width))
        self.outproj = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        self.apply(init_weights)

    def forward(self, x, motion_length):
        x = self.embed(x)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            motion_length = motion_length // 2
            x = attn_block(x, motion_length)
        return self.outproj(x)

@MODELS.register_module()
class DecoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(AttnBlock(width))

        self.outproj = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )
        self.apply(init_weights)

    def forward(self, x, motion_length):
        x = self.embed(x)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            motion_length *= 2
            x = attn_block(x, motion_length)

        return self.outproj(x)
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(in_channels, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, m_lens):
        x = x.permute(0, 2, 1)
        key_mask = length_to_mask(m_lens, x.shape[1])

        attn_out, _ = self.attn_block(
            self.norm(x), self.norm(x), self.norm(x), key_padding_mask = ~key_mask
        )

        x = x + attn_out
        return x.permute(0, 2, 1)

class MultiInputIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m_lens=None):
        return x

def length_to_mask(lengths: torch.Tensor, max_length=None):
    '''
    - lengths: (B,)
    - return: (B, max_length)
    '''
    max_length = lengths.max() if max_length is None else max_length
    mask = torch.arange(max_length, device=lengths.device).expand( # type: ignore
        len(lengths), max_length).to(lengths.device) < lengths.unsqueeze(1)
    return mask

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
