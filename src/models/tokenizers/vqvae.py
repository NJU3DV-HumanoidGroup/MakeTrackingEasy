import torch
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from .encdoc.encdoc_attn import length_to_mask
from torch.nn.utils.rnn import pad_sequence

@MODELS.register_module()
class VQVAE(BaseModel):
    def __init__(self, encoder_cfg, decoder_cfg, quantizer_cfg, loss_cfg=None, **kwargs):
        super().__init__(**kwargs)

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.quantizer = MODELS.build(quantizer_cfg)

        if loss_cfg:
            self.recons_loss = MODELS.build(loss_cfg)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward_loss(self, motion, motion_length, **kwargs): 
        # encoder
        x_in = self.preprocess(motion)
        x_encoder = self.encoder(x_in, motion_length=motion_length)

        motion_length = motion_length // (x_in.shape[2] // x_encoder.shape[2])

        # quantization
        x_quantized, commit_loss, perplexity, activate, code_indices = self.quantizer(
            x_encoder, motion_length=motion_length) # B, C, T'
        mask = length_to_mask(motion_length, max_length=x_quantized.shape[-1]).unsqueeze(1) # (B, 1, T')
        x_quantized = x_quantized * mask  # (B, C, T')

        # decoder
        x_decoder = self.decoder(x_quantized, motion_length=motion_length)
        pred_motion = self.postprocess(x_decoder) # (B, T, C)
        losses = self.recons_loss(pred_motion, motion, motion_length=motion_length, commit_loss=commit_loss)
        return losses

    def forward_predict(self, motion, motion_length, **kwargs):
        ## encoder
        x_in = self.preprocess(motion)
        x_encoder = self.encoder(x_in, motion_length=motion_length)
        motion_length = motion_length // (x_in.shape[2] // x_encoder.shape[2])

        x_quantized, commit_loss, perplexity, activate, code_indices = self.quantizer(
            x_encoder, motion_length=motion_length) # B, C, T'
        mask = length_to_mask(motion_length, max_length=x_quantized.shape[-1]).unsqueeze(1) # (B, 1, T')
        x_quantized = x_quantized * mask  # (B, C, T')

        ## decoder
        x_decoder = self.decoder(x_quantized, motion_length=motion_length)
        pred_motion = self.postprocess(x_decoder) # (B, T, C)
        return pred_motion, code_indices

    def forward(self, motion: torch.Tensor, mode: str='tensor', **kwargs): # type: ignore
        if mode == 'loss':
            return self.forward_loss(motion, **kwargs)
        elif mode == 'predict':
            return self.forward_predict(motion, **kwargs)
        else:
            raise NotImplementedError

    def encode(self, motion, motion_length):
        if isinstance(motion, list):
            motion = pad_sequence(motion, batch_first=True)

        x_in = self.preprocess(motion)
        x_encoder = self.encoder(x_in, motion_length=motion_length)
        code_indices_length = motion_length // (x_in.shape[2] // x_encoder.shape[2])

        code_indices = self.quantizer(x_encoder, motion_length=motion_length)[-1] # B, C, T'

        new_code_indices = []
        for code_indice, code_indice_length in zip(code_indices, code_indices_length):
            new_code_indices.append(code_indice[:code_indice_length])
        return new_code_indices

    def decode(self, x, motion_length=None):
        assert x.dim() == 2 # B, T
        # import ipdb; ipdb.set_trace()
        B, T = x.shape
        x_d = self.quantizer.dequantize(x) # B, T
        x_d = x_d.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        if motion_length is None:
            motion_length = torch.tensor(T).unsqueeze(0).to(x_d.device).repeat(B)
        x_out = self.decoder(x_d, motion_length)
        return self.postprocess(x_out)
