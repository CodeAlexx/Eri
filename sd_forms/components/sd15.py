"""
SD 1.5 Component for SD Forms
Optimized for Stable Diffusion 1.5 models with common enhancements
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import torch
import json
from PIL import Image

from .base import VisualComponent
from ..core import (
    Port, 
    PropertyDefinition, 
    PropertyType,
    ComponentStatus,
    PortType,
    PortDirection,
    create_port,
    create_property_definition
)
from ..utils.constants import DEVICE, DIFFUSERS_AVAILABLE

# SD 1.5 model presets
SD15_PRESETS = {
    'sd-v1-5': {
        'name': 'Stable Diffusion v1.5',
        'path': 'runwayml/stable-diffusion-v1-5',
        'optimal_size': (512, 512),
        'description': 'Official SD 1.5 base model'
    },
    'dreamshaper': {
        'name': 'DreamShaper',
        'path': '/home/alex/SwarmUI/Models/diffusion_models/dreamshaper_8.safetensors',
        'optimal_size': (512, 512),
        'description': 'High quality artistic model'
    },
    'realistic-vision': {
        'name': 'Realistic Vision',
        'path': '/home/alex/SwarmUI/Models/diffusion_models/realisticVisionV51.safetensors',
        'optimal_size': (512, 512),
        'description': 'Photorealistic generations'
    },
    'deliberate': {
        'name': 'Deliberate',
        'path': '/home/alex/SwarmUI/Models/diffusion_models/deliberate_v2.safetensors',
        'optimal_size': (512, 512),
        'description': 'Versatile artistic model'
    }
}

class SD15Component(VisualComponent):
    """Specialized SD 1.5 Component with optimized settings"""
    
    component_type = "sd15"
    display_name = "SD 1.5"
    category = "Models"
    icon = "🎨"
    
    # Define output ports only (this is a source component)
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("latents", PortType.LATENT, PortDirection.OUTPUT, optional=True),
        create_port("metadata", PortType.TEXT, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        create_property_definition(
            "model_preset", "Model Preset", PropertyType.CHOICE, "sd-v1-5", "Model",
            metadata={
                "choices": list(SD15_PRESETS.keys()),
                "descriptions": {k: v['description'] for k, v in SD15_PRESETS.items()}
            }
        ),
        create_property_definition(
            "custom_model_path", "Custom Model Path", PropertyType.FILE_PATH, "", "Model",
            metadata={
                "filter": "*.safetensors;*.ckpt;*.bin",
                "description": "Override preset with custom model"
            }
        ),
        PropertyDefinition(
            key="vae_path",
            display_name="VAE Path",
            type=PropertyType.FILE_PATH,
            default="",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.pt;*.bin",
                "description": "Custom VAE (optional)"
            }
        ),
        
        # LoRA Settings
        PropertyDefinition(
            key="loras",
            display_name="LoRA Models",
            type=PropertyType.COLLECTION,
            default=[],
            category="LoRA",
            metadata={
                "item_properties": {
                    "path": {"type": "file", "filter": "*.safetensors"},
                    "strength": {"type": "float", "min": -2.0, "max": 2.0, "default": 1.0}
                }
            }
        ),
        
        # Prompt Settings
        PropertyDefinition(
            key="prompt",
            display_name="Prompt",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompt",
            metadata={
                "placeholder": "Describe what you want to generate...",
                "syntax": "sd-prompt"
            }
        ),
        PropertyDefinition(
            key="negative_prompt",
            display_name="Negative Prompt",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompt",
            metadata={
                "placeholder": "What to avoid...",
                "syntax": "sd-prompt"
            }
        ),
        PropertyDefinition(
            key="prompt_strength",
            display_name="Prompt Strength",
            type=PropertyType.FLOAT,
            default=7.5,
            category="Prompt",
            metadata={"min": 1.0, "max": 30.0, "step": 0.5}
        ),
        
        # Generation Settings
        PropertyDefinition(
            key="width",
            display_name="Width",
            type=PropertyType.INTEGER,
            default=512,
            category="Generation",
            metadata={
                "min": 256, 
                "max": 1024, 
                "step": 64,
                "description": "SD 1.5 optimal: 512"
            }
        ),
        PropertyDefinition(
            key="height",
            display_name="Height",
            type=PropertyType.INTEGER,
            default=512,
            category="Generation",
            metadata={
                "min": 256, 
                "max": 1024, 
                "step": 64,
                "description": "SD 1.5 optimal: 512"
            }
        ),
        PropertyDefinition(
            key="steps",
            display_name="Steps",
            type=PropertyType.INTEGER,
            default=20,
            category="Generation",
            metadata={"min": 1, "max": 150}
        ),
        PropertyDefinition(
            key="sampler",
            display_name="Sampler",
            type=PropertyType.ENUM,
            default="DPMSolverMultistep",
            category="Generation",
            metadata={
                "values": ["DPMSolverMultistep", "DDIM", "Euler", "Euler a", "DPM++ 2M", "DPM++ 2M Karras", "PNDM", "HeunDiscrete"],
                "descriptions": {
                    "DPMSolverMultistep": "Fast, high quality (recommended)",
                    "Euler a": "Creative, artistic results",
                    "DDIM": "Fast, deterministic",
                    "DPM++ 2M Karras": "High quality, slower"
                }
            }
        ),
        PropertyDefinition(
            key="batch_size",
            display_name="Batch Size",
            type=PropertyType.INTEGER,
            default=1,
            category="Generation",
            metadata={"min": 1, "max": 8}
        ),
        PropertyDefinition(
            key="seed",
            display_name="Seed",
            type=PropertyType.INTEGER,
            default=-1,
            category="Generation",
            metadata={"description": "-1 for random"}
        ),
        
        # Advanced Settings
        PropertyDefinition(
            key="clip_skip",
            display_name="CLIP Skip",
            type=PropertyType.INTEGER,
            default=1,
            category="Advanced",
            metadata={
                "min": 1, 
                "max": 4,
                "description": "2 for anime models"
            }
        ),
        PropertyDefinition(
            key="enable_freeu",
            display_name="Enable FreeU",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Advanced",
            metadata={"description": "Quality enhancement technique"}
        ),
        PropertyDefinition(
            key="enable_tiling",
            display_name="Enable Tiling",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Advanced",
            metadata={"description": "For seamless textures"}
        ),
        PropertyDefinition(
            key="safety_checker",
            display_name="Safety Checker",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Advanced"
        ),
    ]
    
    def __init__(self, id: str = None):
        super().__init__(id)
        self.pipeline = None
        self.model_info = None
    
    def load_model(self):
        """Load SD 1.5 model and components"""
        try:
            if not DIFFUSERS_AVAILABLE:
                raise ImportError("Diffusers library not available")
            
            from diffusers import (
                StableDiffusionPipeline, 
                DPMSolverMultistepScheduler,
                DDIMScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                PNDMScheduler,
                HeunDiscreteScheduler,
                DPMSolverSinglestepScheduler
            )
            
            # Determine model path
            custom_path = self.properties.get("custom_model_path")
            if custom_path and Path(custom_path).exists():
                model_path = custom_path
                print(f"Loading custom model: {custom_path}")
            else:
                preset = self.properties.get("model_preset", "sd-v1-5")
                preset_info = SD15_PRESETS.get(preset, SD15_PRESETS["sd-v1-5"])
                model_path = preset_info["path"]
                self.model_info = preset_info
                print(f"Loading preset: {preset_info['name']}")
            
            # Load pipeline
            if model_path.startswith("runwayml/") or "/" in model_path and not Path(model_path).exists():
                # HuggingFace model
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    safety_checker=None if not self.properties.get("safety_checker") else "default",
                    requires_safety_checker=self.properties.get("safety_checker", False)
                )
            else:
                # Local model
                self.pipeline = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Load custom VAE if specified
            vae_path = self.properties.get("vae_path")
            if vae_path and Path(vae_path).exists():
                print(f"Loading custom VAE: {vae_path}")
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                )
                self.pipeline.vae = vae
            
            # Move to device
            self.pipeline = self.pipeline.to(DEVICE)
            
            # Enable optimizations
            if self.properties.get("enable_freeu"):
                self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
            
            if self.properties.get("enable_tiling"):
                self.pipeline.vae.enable_tiling()
            
            # Apply LoRAs
            for lora in self.properties.get("loras", []):
                if lora.get("path") and Path(lora["path"]).exists():
                    print(f"Loading LoRA: {lora['path']} @ {lora.get('strength', 1.0)}")
                    self.pipeline.load_lora_weights(
                        lora["path"],
                        weight_name=Path(lora["path"]).name,
                        adapter_name=Path(lora["path"]).stem
                    )
                    self.pipeline.set_adapters(
                        [Path(lora["path"]).stem],
                        adapter_weights=[lora.get("strength", 1.0)]
                    )
            
            print("SD 1.5 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading SD 1.5 model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_scheduler(self, name: str):
        """Get scheduler by name"""
        from diffusers import (
            DPMSolverMultistepScheduler,
            DDIMScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
            HeunDiscreteScheduler
        )
        
        schedulers = {
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "DDIM": DDIMScheduler,
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,  # Will configure algorithm
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,  # Will configure with Karras
            "PNDM": PNDMScheduler,
            "HeunDiscrete": HeunDiscreteScheduler
        }
        
        scheduler_class = schedulers.get(name, DPMSolverMultistepScheduler)
        config = self.pipeline.scheduler.config
        
        if "DPM++ 2M" in name:
            return DPMSolverMultistepScheduler.from_config(
                config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas="Karras" in name
            )
        
        return scheduler_class.from_config(config)
    
    async def process(self, context) -> bool:
        """Generate images using SD 1.5"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Load model if not already loaded
            if self.pipeline is None:
                if not self.load_model():
                    raise ValueError("Failed to load SD 1.5 model")
            
            # Set scheduler
            sampler_name = self.properties.get("sampler", "DPMSolverMultistep")
            self.pipeline.scheduler = self.get_scheduler(sampler_name)
            
            # CLIP skip
            clip_skip = self.properties.get("clip_skip", 1)
            if clip_skip > 1:
                self.pipeline.text_encoder.config.num_hidden_layers -= (clip_skip - 1)
            
            # Prepare generation parameters
            prompt = self.properties.get("prompt", "")
            negative_prompt = self.properties.get("negative_prompt", "")
            
            gen_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": self.properties.get("steps", 20),
                "guidance_scale": self.properties.get("prompt_strength", 7.5),
                "width": self.properties.get("width", 512),
                "height": self.properties.get("height", 512),
                "num_images_per_prompt": self.properties.get("batch_size", 1),
            }
            
            # Handle seed
            seed = self.properties.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            
            gen_params["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)
            
            # Add progress callback
            if hasattr(context, 'preview_callback'):
                def callback(step, timestep, latents):
                    if step % 5 == 0:  # Preview every 5 steps
                        progress = int((step / gen_params["num_inference_steps"]) * 100)
                        # Decode latents for preview
                        with torch.no_grad():
                            preview = self.pipeline.vae.decode(
                                latents / self.pipeline.vae.config.scaling_factor
                            ).sample
                            preview = (preview / 2 + 0.5).clamp(0, 1)
                            preview = preview.cpu().permute(0, 2, 3, 1).numpy()
                            preview_img = Image.fromarray((preview[0] * 255).astype("uint8"))
                            preview_img = preview_img.resize((256, 256))  # Small preview
                            context.preview_callback(preview_img, progress, step)
                
                gen_params["callback"] = callback
                gen_params["callback_steps"] = 1
            
            # Generate
            print(f"Generating with SD 1.5: {sampler_name}, {gen_params['num_inference_steps']} steps")
            result = self.pipeline(**gen_params)
            
            # Set outputs
            self.set_output_data("images", result.images)
            if hasattr(result, "latents"):
                self.set_output_data("latents", result.latents)
            
            # Metadata
            metadata = {
                "model": self.model_info.get("name", "SD 1.5") if self.model_info else "Custom SD 1.5",
                "sampler": sampler_name,
                "steps": gen_params["num_inference_steps"],
                "cfg_scale": gen_params["guidance_scale"],
                "size": f"{gen_params['width']}x{gen_params['height']}",
                "seed": seed,
                "clip_skip": clip_skip,
                "loras": [{"path": l["path"], "strength": l.get("strength", 1.0)} 
                         for l in self.properties.get("loras", [])]
            }
            self.set_output_data("metadata", metadata)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SD15Component: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("SD 1.5 model unloaded")
