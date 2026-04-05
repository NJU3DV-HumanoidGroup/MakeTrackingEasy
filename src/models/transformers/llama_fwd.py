from mmengine.registry import MODELS
from .llama_ar import LLaMAHF_AR

@MODELS.register_module()
class LLaMAHF_Fwd(LLaMAHF_AR):
    def __init__(self, **kwargs) -> None:
        '''
        end_token_idx: vocab size - 2
        pad_token_idx: vocab size - 1
        '''
        super().__init__(**kwargs)
        del self.transformer.wte

    def forward(self, input_embd, masks): # type: ignore
        # import ipdb; ipdb.set_trace()
        B, T, C = input_embd.size()
        x = input_embd
        assert (T <= self.config.block_size), \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        for block in self.transformer.h: # type: ignore
            x, _ = block(x, masks)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)  # (b, t, vocab_size)
        return logits
