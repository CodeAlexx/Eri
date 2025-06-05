"""
Model configuration management for SD Forms UI
"""

import json
from pathlib import Path
from typing import Dict, List

from ..utils.constants import CONFIG_FILE


class ModelConfig:
    """Manages model configurations and paths"""
    def __init__(self):
        self.config_path = Path(CONFIG_FILE)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "model_paths": [
                "~/stable-diffusion-webui/models/Stable-diffusion",
                "~/ComfyUI/models/checkpoints",
                "./models/checkpoints"
            ],
            "vae_paths": [
                "~/stable-diffusion-webui/models/VAE",
                "~/ComfyUI/models/vae",
                "./models/vae"
            ],
            "lora_paths": [
                "~/stable-diffusion-webui/models/Lora",
                "~/ComfyUI/models/loras",
                "./models/loras"
            ],
            "embedding_paths": [
                "~/stable-diffusion-webui/embeddings",
                "~/ComfyUI/models/embeddings",
                "./models/embeddings"  
            ],
            "controlnet_paths": [
                "~/ComfyUI/models/controlnet",
                "./models/controlnet"
            ],
            "upscaler_paths": [
                "~/stable-diffusion-webui/models/ESRGAN",
                "~/ComfyUI/models/upscale_models",
                "./models/upscalers"
            ],
            "clip_paths": [
                "~/ComfyUI/models/clip",
                "./models/clip"
            ],
            "recent_models": [],
            "favorite_models": [],
            "model_presets": {}
        }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def scan_models(self, paths: List[str], extensions: List[str]) -> List[Dict]:
        """Scan directories for model files"""
        models = []
        
        for path in paths:
            path = Path(path).expanduser()
            if not path.exists():
                continue
            
            for ext in extensions:
                for file in path.rglob(f"*{ext}"):
                    model_info = {
                        "name": file.stem,
                        "path": str(file),
                        "size": file.stat().st_size / (1024**3),  # GB
                        "modified": file.stat().st_mtime,
                        "type": ext,
                        "directory": str(file.parent)
                    }
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x["name"])