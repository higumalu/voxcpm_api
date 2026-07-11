"""Work around Windows / WSL2 Docker CUDA and checkpoint loading issues."""

from __future__ import annotations

import gc
import os
import sys
from typing import Any, Callable


def _is_wsl2() -> bool:
    if os.getenv("VOXCPM_WSL2_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        with open("/proc/version", encoding="utf-8") as fp:
            return "microsoft" in fp.read().lower()
    except OSError:
        return False


def _patch_cuda_mem_get_info() -> None:
    """Fix broken cgroup memory reporting on Windows Docker WSL2."""
    import subprocess

    import torch

    if getattr(torch.cuda.mem_get_info, "_voxcpm_api_patched", False):
        return

    _original_mem_get_info = torch.cuda.mem_get_info

    def mem_get_info(device=None):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device if device is not None else 0),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                free_mb, total_mb = map(float, result.stdout.strip().split(","))
                return int(free_mb * 1024 * 1024), int(total_mb * 1024 * 1024)
        except Exception:
            pass
        return _original_mem_get_info(device)

    mem_get_info._voxcpm_api_patched = True  # type: ignore[attr-defined]
    torch.cuda.mem_get_info = mem_get_info  # type: ignore[method-assign]


def _patch_cuda_device_properties() -> None:
    """Avoid WSL2 misreporting total VRAM to the PyTorch caching allocator."""
    import torch

    original = torch.cuda.get_device_properties

    if getattr(original, "_voxcpm_api_patched", False):
        return

    def get_device_properties(device=None):
        props = original(device)
        if not _is_wsl2():
            return props
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            if total_bytes > 0:
                props.total_memory = total_bytes  # type: ignore[attr-defined]
        except Exception:
            pass
        return props

    get_device_properties._voxcpm_api_patched = True  # type: ignore[attr-defined]
    torch.cuda.get_device_properties = get_device_properties  # type: ignore[method-assign]


def _move_model_to_device_except_vae(model: Any, device: str) -> None:
    import torch

    target = torch.device(device)
    for name, param in model.named_parameters():
        if name.startswith("audio_vae."):
            param.data = param.data.to("cpu")
        else:
            param.data = param.data.to(target)
    for name, buf in model.named_buffers():
        if name.startswith("audio_vae."):
            buf.data = buf.data.to("cpu")
        else:
            buf.data = buf.data.to(target)
    if target.type == "cuda":
        torch.cuda.empty_cache()


def patch_safetensors_cuda_loading() -> None:
    """Patch voxcpm loading. Call before and after ``import voxcpm``."""
    _patch_cuda_mem_get_info()
    _patch_cuda_device_properties()
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

    target = torch.device(device)
    moved: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        moved[key] = tensor.to(target)
    state_dict.clear()
    del state_dict
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return moved


def _load_checkpoint_state_dict(path: str, st_load_file: Callable[..., Any]) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Always stage weights on CPU first to avoid VRAM spikes during load.
    if os.name == "nt" or _is_wsl2():
        return _load_without_mmap(path, "cpu", st_load_file)
    return st_load_file(path, device="cpu")


def _patch_safetensors_load_file() -> None:
    import safetensors.torch as st

    if getattr(st.load_file, "_voxcpm_api_patched", False):
        return

    original_load_file = st.load_file

    def load_file(filename, device="cpu"):
        if os.path.getsize(filename) > 512 * 1024 * 1024 and (os.name == "nt" or _is_wsl2()):
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
            model_state_dict = _load_checkpoint_state_dict(safetensors_path, st_load_file)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}", file=sys.stderr)
            checkpoint = torch.load(pytorch_model_path, map_location="cpu", weights_only=True)
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

        model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("audio_vae.")}
        for key, value in vae_state_dict.items():
            model_state_dict[f"audio_vae.{key}"] = value
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

        if training:
            return model

        model = model.eval()
        if torch.cuda.is_available() and str(model.device) == "cuda":
            _move_model_to_device_except_vae(model, "cuda:0")
            print("Deployed LM on CUDA; AudioVAE kept on CPU for WSL2 VRAM headroom", file=sys.stderr)
        return model.optimize(disable=not optimize)

    from_local._voxcpm_api_patched = True  # type: ignore[attr-defined]
    VoxCPM2Model.from_local = from_local
