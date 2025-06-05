"""
SD 1.5 Component - Simplified
"""

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)

class SD15Component(VisualComponent):
    """SD 1.5 Model Component"""
    
    component_type = "sd15"
    display_name = "SD 1.5"
    category = "Models"
    icon = "🎨"
    
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT),
    ]
    
    property_definitions = [
        create_property_definition("model_path", "Model Path", PropertyType.FILE_PATH, 
                                 "/home/alex/SwarmUI/Models/Stable-Diffusion/bigLust_v15.safetensors", "Model"),
        create_property_definition("prompt", "Prompt", PropertyType.TEXT, "a beautiful landscape", "Generation"),
        create_property_definition("negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Generation"),
        create_property_definition("steps", "Steps", PropertyType.INTEGER, 20, "Generation"),
        create_property_definition("width", "Width", PropertyType.INTEGER, 512, "Generation"),
        create_property_definition("height", "Height", PropertyType.INTEGER, 512, "Generation"),
        create_property_definition("loras", "LoRA Collection", PropertyType.STRING, "", "LoRAs",
                                 metadata={"editor_type": "lora_collection"}),
    ]
    
    def __init__(self, component_id=None, comp_type="SD 1.5", position=None, scene=None):
        super().__init__(comp_type, position or (0, 0), scene, component_id)
    
    async def process(self, context):
        """Load and prepare SD 1.5 pipeline"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            print(f"🔧 Loading Stable Diffusion 1.5")
            
            # Check if we have diffusers available
            try:
                from diffusers import StableDiffusionPipeline
                import torch
                
                # Get model path from properties
                model_path = self.properties.get("model_path", "/home/alex/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion/sd-v1-5-pruned-emaonly.safetensors")
                
                # Check if model file exists
                from pathlib import Path
                if not Path(model_path).exists():
                    print(f"❌ SD 1.5 model file not found: {model_path}")
                    self.set_status(ComponentStatus.ERROR)
                    return False
                
                print(f"🔧 Loading local SD 1.5: {model_path}")
                
                # Load the pipeline from local file
                pipeline = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                # Move to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    pipeline = pipeline.to(device)
                
                print(f"✅ SD 1.5 loaded successfully on {device}")
                
                # Set outputs compatible with existing sampler component
                self.set_output_data("pipeline", pipeline)
                self.set_output_data("conditioning", {
                    "prompt": self.properties.get("prompt", "a beautiful landscape"),
                    "negative_prompt": self.properties.get("negative_prompt", ""),
                    "model_type": "sd15",
                    "width": self.properties.get("width", 512),
                    "height": self.properties.get("height", 512),
                    "steps": self.properties.get("steps", 20)
                })
                
                # Store pipeline for direct use
                self.pipeline = pipeline
                
                # Load LoRAs if specified
                self._load_loras()
                
            except ImportError:
                print("❌ Diffusers not available, cannot load SD 1.5")
                self.set_status(ComponentStatus.ERROR)
                return False
            except Exception as model_error:
                print(f"❌ Failed to load SD 1.5 model: {model_error}")
                self.set_status(ComponentStatus.ERROR)
                return False
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SD15Component: {e}")
            return False
    
    def _load_loras(self):
        """Load and apply LoRA models for SD 1.5"""
        loras_str = self.properties.get("loras", "")
        if not loras_str or not self.pipeline:
            return
        
        try:
            import json
            from pathlib import Path
            
            # Parse LoRA collection
            if isinstance(loras_str, str):
                if loras_str.strip():
                    loras = json.loads(loras_str)
                else:
                    loras = []
            else:
                loras = loras_str
            
            loaded_count = 0
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
                        self.pipeline.load_lora_weights(
                            lora_path,
                            adapter_name=adapter_name
                        )
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(strength)
                        loaded_count += 1
                    else:
                        print("Pipeline doesn't support LoRA loading")
                        break
                        
                except Exception as e:
                    print(f"Error loading LoRA {lora_path}: {e}")
                    continue
            
            # Apply LoRAs
            if loaded_count > 0 and hasattr(self.pipeline, 'set_adapters'):
                self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                print(f"✅ Loaded {loaded_count} LoRAs for SD 1.5")
            
        except Exception as e:
            print(f"Error loading LoRAs: {e}")