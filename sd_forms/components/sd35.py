"""
SD3.5 Model Component for SD Forms
Supports SD3.5 Large, Medium, and Large Turbo variants with integrated LoRAs
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import torch
from PIL import Image
import json
import gc

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

class SD35Component(VisualComponent):
    """SD3.5 Model Component with all variants and integrated LoRAs"""
    
    component_type = "sd35"
    display_name = "SD3.5 Model"
    category = "Models"
    icon = "🚀"
    
    # Define output ports
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT),
        create_port("model_info", PortType.TEXT, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Selection
        PropertyDefinition(
            key="model_variant",
            display_name="Model Variant", 
            type=PropertyType.ENUM,
            default="sd3.5-large",
            category="Model",
            metadata={
                "values": ["sd3.5-large", "sd3.5-large-turbo", "sd3.5-medium", "custom"],
                "description": "Select SD3.5 model variant"
            }
        ),
        PropertyDefinition(
            key="model_path",
            display_name="Model Path",
            type=PropertyType.FILE_PICKER,
            default="",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.ckpt;*.bin",
                "description": "Path to model file (for custom variant)"
            },
            depends_on=["model_variant"]
        ),
        PropertyDefinition(
            key="t5_variant",
            display_name="T5 Text Encoder",
            type=PropertyType.ENUM,
            default="t5-xxl",
            category="Model",
            metadata={
                "values": ["t5-xxl", "t5-large", "t5-base", "none"],
                "description": "T5 text encoder size (none to disable)"
            }
        ),
        PropertyDefinition(
            key="custom_t5_path",
            display_name="Custom T5 Path",
            type=PropertyType.FILE_PICKER,
            default="",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.bin",
                "description": "Path to custom T5 encoder"
            }
        ),
        
        # Text Encoding
        PropertyDefinition(
            key="prompt",
            display_name="Prompt",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompting",
            metadata={
                "placeholder": "Describe your image...",
                "syntax": "sd3-prompt"
            }
        ),
        PropertyDefinition(
            key="negative_prompt",
            display_name="Negative Prompt",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompting"
        ),
        PropertyDefinition(
            key="prompt_3",
            display_name="T5 Prompt",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompting",
            metadata={
                "description": "SD3.5 T5 text encoder prompt (optional)",
                "placeholder": "Leave empty to use main prompt"
            }
        ),
        PropertyDefinition(
            key="negative_prompt_3",
            display_name="T5 Negative",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Prompting",
            metadata={
                "description": "SD3.5 T5 negative prompt"
            }
        ),
        PropertyDefinition(
            key="max_sequence_length",
            display_name="Max Sequence Length",
            type=PropertyType.INTEGER,
            default=512,
            category="Prompting",
            metadata={
                "min": 77,
                "max": 512,
                "description": "Maximum token length for T5 encoder"
            }
        ),
        
        # Model-Specific Settings
        PropertyDefinition(
            key="guidance_scale",
            display_name="Guidance Scale",
            type=PropertyType.FLOAT,
            default=7.0,
            category="Generation",
            metadata={
                "min": 0.0, 
                "max": 20.0, 
                "step": 0.5,
                "sd35_large_default": 7.0,
                "sd35_turbo_default": 0.0,
                "sd35_medium_default": 5.0
            }
        ),
        PropertyDefinition(
            key="num_inference_steps",
            display_name="Steps",
            type=PropertyType.INTEGER,
            default=28,
            category="Generation",
            metadata={
                "min": 1,
                "max": 100,
                "sd35_large_default": 28,
                "sd35_turbo_default": 4,
                "sd35_medium_default": 20
            }
        ),
        PropertyDefinition(
            key="shift",
            display_name="Shift (Timestep)",
            type=PropertyType.FLOAT,
            default=3.0,
            category="Generation",
            metadata={
                "min": 0.0,
                "max": 10.0,
                "step": 0.5,
                "description": "SD3.5 specific timestep shift parameter"
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
                "item_schema": {
                    "path": {
                        "type": PropertyType.FILE_PICKER,
                        "filter": "*.safetensors;*.bin",
                        "display_name": "LoRA Path"
                    },
                    "strength": {
                        "type": PropertyType.FLOAT,
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display_name": "Strength"
                    },
                    "enabled": {
                        "type": PropertyType.BOOLEAN,
                        "default": True,
                        "display_name": "Enabled"
                    }
                },
                "description": "Add multiple LoRA models with individual strengths"
            }
        ),
        PropertyDefinition(
            key="lora_merge_strategy",
            display_name="LoRA Merge Strategy",
            type=PropertyType.ENUM,
            default="sequential",
            category="LoRA",
            metadata={
                "values": ["sequential", "weighted", "concatenate"],
                "description": "How to apply multiple LoRAs"
            }
        ),
        
        # Advanced Settings
        PropertyDefinition(
            key="scheduler",
            display_name="Scheduler",
            type=PropertyType.ENUM,
            default="FlowMatchEulerDiscrete",
            category="Advanced",
            metadata={
                "values": [
                    "FlowMatchEulerDiscrete",
                    "FlowMatchHeun",
                    "DPMSolverMultistep",
                    "DDIM",
                    "Euler",
                    "EulerAncestral"
                ],
                "description": "Sampling scheduler (SD3.5 uses Flow Matching)"
            }
        ),
        PropertyDefinition(
            key="attention_mode",
            display_name="Attention Mode",
            type=PropertyType.ENUM,
            default="sdpa",
            category="Advanced",
            metadata={
                "values": ["sdpa", "xformers", "flash_attention_2", "default"],
                "description": "Attention implementation"
            }
        ),
        PropertyDefinition(
            key="clip_skip",
            display_name="CLIP Skip",
            type=PropertyType.INTEGER,
            default=0,
            category="Advanced",
            metadata={
                "min": 0,
                "max": 4,
                "description": "Skip CLIP layers (0 = disabled)"
            }
        ),
        
        # Memory Optimization
        PropertyDefinition(
            key="enable_cpu_offload",
            display_name="CPU Offload",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Memory",
            metadata={
                "description": "Offload model to CPU when not in use"
            }
        ),
        PropertyDefinition(
            key="enable_vae_slicing",
            display_name="VAE Slicing",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Memory",
            metadata={
                "description": "Enable VAE slicing for lower VRAM usage"
            }
        ),
        PropertyDefinition(
            key="enable_vae_tiling",
            display_name="VAE Tiling",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Memory",
            metadata={
                "description": "Enable VAE tiling for very large images"
            }
        ),
        PropertyDefinition(
            key="enable_t5_cpu_offload",
            display_name="T5 CPU Offload",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Memory",
            metadata={
                "description": "Keep T5 encoder on CPU to save VRAM"
            }
        ),
    ]
    
    def __init__(self, id: str = None):
        super().__init__(id)
        self.pipeline = None
        self.text_encoder_3 = None  # T5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._user_modified_properties = set()
        self.loaded_loras = []
        self.variant_configs = {
            "sd3.5-large": {
                "repo_id": "stabilityai/stable-diffusion-3.5-large",
                "steps": 28,
                "guidance": 7.0,
                "shift": 3.0
            },
            "sd3.5-large-turbo": {
                "repo_id": "stabilityai/stable-diffusion-3.5-large-turbo",
                "steps": 4,
                "guidance": 0.0,
                "shift": 3.0
            },
            "sd3.5-medium": {
                "repo_id": "stabilityai/stable-diffusion-3.5-medium",
                "steps": 20,
                "guidance": 5.0,
                "shift": 3.0
            }
        }
    
    def _get_model_paths(self) -> Dict[str, str]:
        """Get model paths based on variant"""
        variant = self.properties.get("model_variant", "sd3.5-large")
        
        # Standard model locations
        base_paths = [
            Path.home() / "models" / "sd3",
            Path("/home/alex/SwarmUI/Models/Stable-Diffusion"),
            Path("./models/sd3"),
            Path.home() / ".cache" / "huggingface" / "hub"
        ]
        
        model_files = {
            "sd3.5-large": ["sd3.5_large.safetensors", "sd3_5_large.safetensors"],
            "sd3.5-large-turbo": ["sd3.5_large_turbo.safetensors", "sd3_5_large_turbo.safetensors"],
            "sd3.5-medium": ["sd3.5_medium.safetensors", "sd3_5_medium.safetensors"]
        }
        
        # Search for model files
        if variant != "custom":
            for base_path in base_paths:
                if not base_path.exists():
                    continue
                    
                for filename in model_files.get(variant, []):
                    full_path = base_path / filename
                    if full_path.exists():
                        return {"model": str(full_path)}
        
        # Custom path
        custom_path = self.properties.get("model_path", "")
        if custom_path and Path(custom_path).exists():
            return {"model": custom_path}
            
        return {}
    
    def _load_t5_encoder(self):
        """Load T5 text encoder"""
        try:
            t5_variant = self.properties.get("t5_variant", "t5-xxl")
            
            if t5_variant == "none":
                print("T5 encoder disabled")
                return
            
            custom_t5_path = self.properties.get("custom_t5_path", "")
            
            if custom_t5_path and Path(custom_t5_path).exists():
                print(f"Loading custom T5 encoder from: {custom_t5_path}")
                from transformers import T5EncoderModel
                
                self.text_encoder_3 = T5EncoderModel.from_single_file(
                    custom_t5_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
            else:
                # Load standard T5 variants
                t5_models = {
                    "t5-xxl": "stabilityai/stable-diffusion-3.5-large",  # Includes T5-XXL
                    "t5-large": "google/t5-v1_1-large",
                    "t5-base": "google/t5-v1_1-base"
                }
                
                model_id = t5_models.get(t5_variant)
                if model_id:
                    print(f"Loading T5 encoder: {t5_variant}")
                    
                    # For SD3.5 models, T5 is included
                    if t5_variant == "t5-xxl" and hasattr(self.pipeline, 'text_encoder_3'):
                        # T5 already loaded with pipeline
                        self.text_encoder_3 = self.pipeline.text_encoder_3
                    else:
                        from transformers import T5EncoderModel
                        self.text_encoder_3 = T5EncoderModel.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                        )
            
            # Apply T5 CPU offload if enabled
            if self.text_encoder_3 and self.properties.get("enable_t5_cpu_offload", True):
                print("Moving T5 encoder to CPU for memory optimization")
                self.text_encoder_3 = self.text_encoder_3.to("cpu")
                
        except Exception as e:
            print(f"Error loading T5 encoder: {e}")
    
    def _load_loras(self):
        """Load and apply LoRA models"""
        loras = self.properties.get("loras", [])
        if not loras or not self.pipeline:
            return
        
        try:
            loaded_count = 0
            strategy = self.properties.get("lora_merge_strategy", "sequential")
            self.loaded_loras = []
            
            adapter_names = []
            adapter_weights = []
            
            for i, lora in enumerate(loras):
                if not lora.get("enabled", True):
                    continue
                    
                lora_path = lora.get("path", "")
                if not lora_path or not Path(lora_path).exists():
                    print(f"LoRA path not found: {lora_path}")
                    continue
                
                strength = lora.get("strength", 1.0)
                adapter_name = f"lora_{i}"
                
                try:
                    print(f"Loading LoRA: {Path(lora_path).name} (strength: {strength})")
                    
                    if hasattr(self.pipeline, 'load_lora_weights'):
                        # SD3.5 LoRA loading
                        self.pipeline.load_lora_weights(
                            lora_path,
                            adapter_name=adapter_name,
                            # SD3.5 specific: LoRAs might target different components
                            weight_name=None  # Auto-detect
                        )
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(strength)
                        loaded_count += 1
                        
                        self.loaded_loras.append({
                            "path": lora_path,
                            "name": Path(lora_path).name,
                            "strength": strength,
                            "adapter_name": adapter_name
                        })
                    else:
                        print("Pipeline doesn't support LoRA loading")
                        break
                        
                except Exception as e:
                    print(f"Error loading LoRA {lora_path}: {e}")
                    continue
            
            # Apply LoRAs based on strategy
            if loaded_count > 0 and hasattr(self.pipeline, 'set_adapters'):
                if strategy == "sequential":
                    for name, weight in zip(adapter_names, adapter_weights):
                        self.pipeline.set_adapters([name], adapter_weights=[weight])
                elif strategy == "weighted":
                    self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                elif strategy == "concatenate":
                    equal_weights = [1.0 / len(adapter_names)] * len(adapter_names)
                    self.pipeline.set_adapters(adapter_names, adapter_weights=equal_weights)
                
                print(f"Loaded {loaded_count} LoRA(s) with strategy: {strategy}")
            
        except Exception as e:
            print(f"Error in LoRA loading: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_variant_defaults(self):
        """Apply model variant specific defaults"""
        variant = self.properties.get("model_variant", "sd3.5-large")
        config = self.variant_configs.get(variant, {})
        
        if "num_inference_steps" not in self._user_modified_properties:
            self.properties["num_inference_steps"] = config.get("steps", 28)
            
        if "guidance_scale" not in self._user_modified_properties:
            self.properties["guidance_scale"] = config.get("guidance", 7.0)
            
        if "shift" not in self._user_modified_properties:
            self.properties["shift"] = config.get("shift", 3.0)
    
    def _load_model(self, model_path: str = None):
        """Load SD3.5 model"""
        try:
            from diffusers import StableDiffusion3Pipeline, AutoPipelineForText2Image
            
            variant = self.properties.get("model_variant", "sd3.5-large")
            
            if model_path and Path(model_path).exists():
                # Load from local file
                print(f"Loading SD3.5 model from: {model_path}")
                
                if model_path.endswith('.safetensors'):
                    # SD3.5 single file loading
                    self.pipeline = StableDiffusion3Pipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                    )
                else:
                    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                    )
            else:
                # Load from HuggingFace
                config = self.variant_configs.get(variant, {})
                print(f"Loading SD3.5 {variant} from HuggingFace: {config.get('repo_id')}")
                
                if variant == "sd3.5-large-turbo":
                    # Use AutoPipeline for turbo
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        config.get("repo_id"),
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        use_auth_token=True  # SD3.5 might need auth
                    )
                else:
                    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                        config.get("repo_id"),
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        use_auth_token=True  # SD3.5 might need auth
                    )
            
            # Set attention mode
            attention_mode = self.properties.get("attention_mode", "sdpa")
            if hasattr(self.pipeline, 'set_attn_processor'):
                if attention_mode == "xformers":
                    self.pipeline.enable_xformers_memory_efficient_attention()
                elif attention_mode == "flash_attention_2":
                    # This would need proper Flash Attention 2 setup
                    print("Flash Attention 2 selected but implementation needed")
            
            # Apply memory optimizations
            if self.properties.get("enable_cpu_offload", False):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            if self.properties.get("enable_vae_slicing", False):
                self.pipeline.enable_vae_slicing()
                    
            if self.properties.get("enable_vae_tiling", False):
                self.pipeline.enable_vae_tiling()
            
            # Configure scheduler
            scheduler_name = self.properties.get("scheduler", "FlowMatchEulerDiscrete")
            self._configure_scheduler(scheduler_name)
            
            return True
            
        except Exception as e:
            print(f"Error loading SD3.5 model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _configure_scheduler(self, scheduler_name: str):
        """Configure the scheduler for the pipeline"""
        try:
            from diffusers import (
                FlowMatchEulerDiscreteScheduler,
                FlowMatchHeunDiscreteScheduler,
                DPMSolverMultistepScheduler,
                DDIMScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler
            )
            
            scheduler_map = {
                "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
                "FlowMatchHeun": FlowMatchHeunDiscreteScheduler,
                "DPMSolverMultistep": DPMSolverMultistepScheduler,
                "DDIM": DDIMScheduler,
                "Euler": EulerDiscreteScheduler,
                "EulerAncestral": EulerAncestralDiscreteScheduler
            }
            
            scheduler_class = scheduler_map.get(scheduler_name)
            if scheduler_class and self.pipeline:
                # SD3.5 uses shift parameter
                config = self.pipeline.scheduler.config
                config["shift"] = self.properties.get("shift", 3.0)
                
                self.pipeline.scheduler = scheduler_class.from_config(config)
                    
        except Exception as e:
            print(f"Error configuring scheduler: {e}")
    
    def set_property(self, key: str, value: Any):
        """Override to track user modifications"""
        if hasattr(super(), 'set_property'):
            super().set_property(key, value)
        else:
            self.properties[key] = value
        self._user_modified_properties.add(key)
    
    async def process(self, context) -> bool:
        """Process SD3.5 model loading and conditioning"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Apply variant-specific defaults
            self._apply_variant_defaults()
            
            # Load model if not already loaded
            if self.pipeline is None:
                paths = self._get_model_paths()
                model_path = paths.get("model") if paths else None
                
                if not self._load_model(model_path):
                    raise ValueError("Failed to load SD3.5 model")
            
            # Load T5 encoder if needed
            self._load_t5_encoder()
            
            # Load LoRAs
            self._load_loras()
            
            # Prepare conditioning
            prompt = self.properties.get("prompt", "")
            negative_prompt = self.properties.get("negative_prompt", "")
            prompt_3 = self.properties.get("prompt_3", "") or prompt
            negative_prompt_3 = self.properties.get("negative_prompt_3", "") or negative_prompt
            
            if not prompt:
                prompt = "A high quality image"
                prompt_3 = prompt
            
            # Set max sequence length for T5
            max_length = self.properties.get("max_sequence_length", 512)
            if hasattr(self.pipeline, 'tokenizer_3'):
                self.pipeline.tokenizer_3.model_max_length = max_length
            
            # Apply CLIP skip if specified
            clip_skip = self.properties.get("clip_skip", 0)
            if clip_skip > 0:
                print(f"CLIP skip: {clip_skip} layers")
            
            # Set outputs
            self.set_output_data("pipeline", self.pipeline)
            
            conditioning = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prompt_3": prompt_3,
                "negative_prompt_3": negative_prompt_3,
                "guidance_scale": self.properties.get("guidance_scale", 7.0),
                "num_inference_steps": self.properties.get("num_inference_steps", 28),
                "max_sequence_length": max_length
            }
            self.set_output_data("conditioning", conditioning)
            
            # Model info
            model_info = {
                "variant": self.properties.get("model_variant", "sd3.5-large"),
                "device": str(self.device),
                "dtype": str(self.pipeline.dtype) if hasattr(self.pipeline, 'dtype') else "float16",
                "scheduler": self.properties.get("scheduler", "FlowMatchEulerDiscrete"),
                "shift": self.properties.get("shift", 3.0),
                "t5_variant": self.properties.get("t5_variant", "t5-xxl"),
                "attention_mode": self.properties.get("attention_mode", "sdpa"),
                "memory_optimizations": {
                    "cpu_offload": self.properties.get("enable_cpu_offload", False),
                    "vae_slicing": self.properties.get("enable_vae_slicing", False),
                    "vae_tiling": self.properties.get("enable_vae_tiling", False),
                    "t5_cpu_offload": self.properties.get("enable_t5_cpu_offload", True)
                },
                "loras": [
                    {
                        "path": lora.get("path", ""),
                        "name": lora.get("name", Path(lora.get("path", "unknown")).name),
                        "strength": lora.get("strength", 1.0)
                    }
                    for lora in self.loaded_loras
                ]
            }
            
            self.set_output_data("model_info", model_info)
            
            print(f"SD3.5 {self.properties.get('model_variant')} loaded successfully")
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SD3.5 component: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self):
        """Unload model to free memory"""
        # Unload LoRAs first
        if self.pipeline and hasattr(self.pipeline, 'unload_lora_weights'):
            try:
                self.pipeline.unload_lora_weights()
            except:
                pass
        
        self.loaded_loras = []
        
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            
        if self.text_encoder_3:
            del self.text_encoder_3
            self.text_encoder_3 = None
        
        # Clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("SD3.5 model unloaded")
        if self.loaded_loras:
            print(f"Unloaded {len(self.loaded_loras)} LoRA(s)")
