from __future__ import annotations

import shutil
from pathlib import Path

from .artifacts import validate_policy_directory


def merge_checkpoint(base, checkpoint, target, vlm):
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Final LoRA adapter missing: {checkpoint}")

    from peft import PeftModel
    from safetensors import safe_open
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    config.pretrained_path = str(base)
    config.use_peft = False
    policy = SmolVLAPolicy.from_pretrained(base, config=config, strict=False)
    merged = PeftModel.from_pretrained(
        policy, checkpoint, is_trainable=False
    ).merge_and_unload(safe_merge=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    merged.config.use_peft = False
    merged.config.pretrained_path = None
    merged.config.push_to_hub = False
    merged.config.repo_id = None
    merged.config.device = None
    merged.config.load_vlm_weights = False
    merged.config.vlm_model_name = str(vlm)
    merged.save_pretrained(target)
    for pattern in (
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ):
        for source in checkpoint.glob(pattern):
            shutil.copy2(source, target / source.name)
    with safe_open(target / "model.safetensors", framework="pt", device="cpu") as weights:
        if any("lora_" in key.lower() for key in weights.keys()):
            raise RuntimeError("LoRA weights remain in merged artifact")
    validate_policy_directory(target)
