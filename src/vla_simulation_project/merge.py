from __future__ import annotations
import shutil
from pathlib import Path
from .artifacts import validate_policy_directory

def merge_checkpoint(base: Path, checkpoint: Path, target: Path, vlm_name: str) -> None:
    adapter = checkpoint / 'adapter_model.safetensors'
    if not adapter.is_file():
        raise FileNotFoundError(f'train merge: final LoRA adapter missing: {adapter}')
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftModel
    from safetensors import safe_open
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = 'cpu'
    config.pretrained_path = str(base)
    config.use_peft = False
    policy = SmolVLAPolicy.from_pretrained(base, config=config, strict=False)
    merged = PeftModel.from_pretrained(policy, checkpoint, is_trainable=False).merge_and_unload(safe_merge=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    merged.config.use_peft = False
    merged.config.pretrained_path = None
    merged.config.push_to_hub = False
    merged.config.repo_id = None
    merged.config.device = None
    merged.config.load_vlm_weights = False
    merged.config.vlm_model_name = vlm_name
    merged.save_pretrained(target)
    for pattern in ('policy_preprocessor.json', 'policy_preprocessor*.safetensors', 'policy_postprocessor.json', 'policy_postprocessor*.safetensors'):
        for source in checkpoint.glob(pattern):
            shutil.copy2(source, target / source.name)
    weights = target / 'model.safetensors'
    with safe_open(weights, framework='pt', device='cpu') as content:
        if any(('lora_' in key.lower() for key in content.keys())):
            raise RuntimeError(f'LoRA adapter keys remain in merged model: {weights}')
    validate_policy_directory(target)
