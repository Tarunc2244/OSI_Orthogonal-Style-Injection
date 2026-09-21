# osi/osi_core.py
import torch
import torch.nn as nn
from typing import Optional, Union, List
from utils.language_utils import LANGUAGE_CONFIGS, post_prune_generated_text
import config as cfg

class OSIConditioner(nn.Module):
    """Injects character style exactly once at the first decoding step."""
    def __init__(
        self,
        model,
        layer_idx: int,
        character_vector: torch.Tensor,
        style_axes: Union[torch.Tensor, List[torch.Tensor]],
        alpha: float = cfg.ALPHA,
    ):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.character_vector = character_vector
        self.style_axes = style_axes if isinstance(style_axes, list) else [style_axes]
        self.alpha = alpha
        self.applied = False

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        else:
            raise ValueError("Cannot locate transformer layers")

        self._orig_forward = self.layers[self.layer_idx].forward
        self._apply_hook()

    def _apply_hook(self):
        orig_forward = self._orig_forward
        style_axes = self.style_axes
        c_vec = self.character_vector
        alpha = self.alpha
        conditioner = self

        def hooked_forward(*args, **kwargs):
            hidden_states = args[0]
            if not conditioner.applied:
                print("🎯 OSI applied at first decoding step!")
                conditioner.applied = True
                h = hidden_states[:, -1, :]          # last token
                h_sem = h.clone()
                for axis in style_axes:
                    axis = axis.to(h.device, h.dtype)
                    coeff = torch.sum(h_sem * axis, dim=-1, keepdim=True)
                    h_sem = h_sem - coeff * axis
                c = c_vec.to(h.device, h.dtype)
                h_new = h_sem + alpha * c            # full OSI
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = h_new
                args = (hidden_states,) + args[1:]
            return orig_forward(*args, **kwargs)

        self.layers[self.layer_idx].forward = hooked_forward

    def restore(self):
        self.layers[self.layer_idx].forward = self._orig_forward


def translate_with_osi(
    text: str,
    character_vector: Optional[torch.Tensor],
    style_axes: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    tokenizer,
    model,
    language: str = 'bengali',
    alpha: float = cfg.ALPHA,
    max_new_tokens: int = cfg.MAX_NEW_TOKENS,
    condition_layer: int = cfg.CONDITION_LAYER_IDX,
) -> str:
    lang_name = LANGUAGE_CONFIGS[language]['name']
    messages = [
        {"role": "system", "content": "You are a helpful translation assistant."},
        {"role": "user", "content": f"Translate this English dialogue to {lang_name}: {text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if character_vector is None or style_axes is None:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        conditioner = OSIConditioner(
            model,
            condition_layer,
            character_vector,
            style_axes,
            alpha=alpha,
        )
        conditioner.applied = False
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        conditioner.restore()

    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    return post_prune_generated_text(raw, language)