import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict
import torch.nn as nn


@MODELS.register_module()
class RetargetTransformerPredMotion_no_smplvq(BaseModel):
    def __init__(self,
                 transformer_cfg: Dict,
                 smplx_vqvae_cfg: Dict,
                 n_embd: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = MODELS.build(transformer_cfg)
        self.motion_encoder = nn.Sequential(
            nn.Linear(512, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        self.smplx_vqvae = MODELS.build(smplx_vqvae_cfg)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(n_embd, n_embd, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(n_embd, n_embd, 3, 1, 1),
        )

        self.projector = nn.Linear(n_embd, 217)

    def forward_predict(self, cond_embd: torch.Tensor, cond_mask: torch.Tensor):
        feat = self.transformer(cond_embd, cond_mask)
        up_sample_feat = self.upsample(feat.permute(0, 2, 1)).permute(0, 2, 1)
        pred_motions = self.projector(up_sample_feat)
        new_mask = F.interpolate(cond_mask[None].float(), scale_factor=2, mode='nearest')[0]
        pred_motion_lengths = new_mask.sum(dim=1)

        _pred_motions = []
        for pred_motion, length in zip(pred_motions, pred_motion_lengths):
            _pred_motions.append(pred_motion[:length.int()])
        return pred_motions, pred_motion_lengths

    def forward(self,
                motion: torch.Tensor = None,
                smplx_motion: torch.Tensor = None,
                motion_length: torch.Tensor = None,
                mode='loss', **kwargs):
        B, T, _ = smplx_motion.shape
        smplx_motion_in = self.smplx_vqvae.preprocess(smplx_motion)
        smplx_motion_embd = self.smplx_vqvae.encoder(smplx_motion_in, motion_length=motion_length)
        smplx_motion_embd = smplx_motion_embd.permute(0, 2, 1)
        smplx_motion_embd = self.motion_encoder(smplx_motion_embd)
        masks = torch.arange(T // 2, device=smplx_motion_embd.device)[None].repeat(B, 1) < motion_length[:, None] // 2

        if mode == 'loss':
            return self.forward_loss(smplx_motion_embd, masks, motion)
        else:
            return self.forward_predict(smplx_motion_embd, masks)
