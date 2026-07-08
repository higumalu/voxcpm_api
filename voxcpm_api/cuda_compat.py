"""Work around Windows page-file failures when loading large VoxCPM checkpoints."""

from __future__ import annotations

import gc
import os
import sys
from typing import Any, Callable


def patch_safetensors_cuda_loading() -> None:
    """Patch voxcpm loading. Call before and after ``import voxcpm``."""
    _patch_safetensors_load_file()
    _patch_voxcpm2_from_local()


def _load_without_mmap(filename: str, device: str, fallback: Callable[..., Any]):
    from safetensors.torch import load

    with open(filename, "rb") as fp:
        data = fp.read()
    state_dict = load(data)
    del data
    gc.collect()
    if str(device) == "cpu":
        return state_dict

    import torch

    moved: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        moved[key] = tensor.to(device)
    state_dict.clear()
    gc.collect()
    return moved


def _patch_safetensors_load_file() -> None:
    import safetensors.torch as st

    if getattr(st.load_file, "_voxcpm_api_patched", False):
        return

    original_load_file = st.load_file

    def load_file(filename, device="cpu"):
        if os.name == "nt" and os.path.getsize(filename) > 512 * 1024 * 1024:
            return _load_without_mmap(filename, device, original_load_file)
        return original_load_file(filename, device=device)

    load_file._voxcpm_api_patched = True  # type: ignore[attr-defined]
    st.load_file = load_file

    for module_name in ("voxcpm.model.voxcpm2", "voxcpm.model.voxcpm"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "load_file"):
            module.load_file = load_file


def _patch_voxcpm2_from_local() -> None:
    try:
        from voxcpm.model.voxcpm2 import VoxCPM2Model
    except ImportError:
        return

    if getattr(VoxCPM2Model.from_local, "_voxcpm_api_patched", False):
        return

    @classmethod
    def from_local(cls, path: str, optimize: bool = True, training: bool = False, lora_config: Any = None):
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file as st_load_file
        from transformers import LlamaTokenizerFast

        from voxcpm.model.utils import get_dtype
        from voxcpm.model.voxcpm2 import (
            SAFETENSORS_AVAILABLE,
            AudioVAEV2,
            VoxCPMConfig,
        )

        config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()

        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading model from safetensors: {safetensors_path}", file=sys.stderr)
            model_state_dict = _load_without_mmap(safetensors_path, "cuda:0", st_load_file)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}", file=sys.stderr)
            checkpoint = torch.load(pytorch_model_path, map_location="cuda:0", weights_only=True)
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}"
            )

        audiovae_safetensors_path = os.path.join(path, "audiovae.safetensors")
        audiovae_pth_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(audiovae_safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading AudioVAE from safetensors: {audiovae_safetensors_path}", file=sys.stderr)
            vae_state_dict = st_load_file(audiovae_safetensors_path, device="cpu")
        elif os.path.exists(audiovae_pth_path):
            print(f"Loading AudioVAE from pytorch: {audiovae_pth_path}", file=sys.stderr)
            checkpoint = torch.load(audiovae_pth_path, map_location="cpu", weights_only=True)
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"AudioVAE checkpoint not found. Expected either {audiovae_safetensors_path} or {audiovae_pth_path}"
            )

        for key, value in vae_state_dict.items():
            model_state_dict[f"audio_vae.{key}"] = value.to("cuda:0", dtype=torch.float32)
        del vae_state_dict
        gc.collect()

        with init_empty_weights():
            model = cls(config, tokenizer, audio_vae, lora_config)

        if training:
            for name, param in model.named_parameters():
                if "audio_vae" in name:
                    param.requires_grad = False
                    continue
                if lora_config is not None and "lora" not in name:
                    param.requires_grad = False
        else:
            model = model.to(get_dtype(model.config.dtype))

        model.load_state_dict(model_state_dict, strict=False, assign=True)
        del model_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        if training:
            return model
        return model.to(model.device).eval().optimize(disable=not optimize)

    from_local._voxcpm_api_patched = True  # type: ignore[attr-defined]
    VoxCPM2Model.from_local = from_local
