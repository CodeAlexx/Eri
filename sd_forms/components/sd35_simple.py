"""
SD3.5 Component - Simplified
"""

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)

class SD35Component(VisualComponent):
    """SD3.5 Model Component"""
    
    component_type = "sd35"
    display_name = "SD3.5"
    category = "Models"
    icon = "🚀"
    
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT),
        create_port("model_info", PortType.TEXT, PortDirection.OUTPUT),
    ]
    
    property_definitions = [
        create_property_definition("model_path", "Model Path", PropertyType.FILE_PATH, 
                                 "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors", "Model"),
        create_property_definition("model_variant", "Model Variant", PropertyType.CHOICE, "sd3.5-large", "Model",
                                 metadata={"choices": ["sd3.5-large", "sd3.5-large-turbo", "sd3.5-medium"]}),
        create_property_definition("prompt", "Prompt", PropertyType.TEXT, "", "Generation"),
        create_property_definition("negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Generation"),
        create_property_definition("num_inference_steps", "Steps", PropertyType.INTEGER, 28, "Generation"),
        create_property_definition("guidance_scale", "Guidance Scale", PropertyType.FLOAT, 7.0, "Generation"),
        create_property_definition("loras", "LoRA Collection", PropertyType.STRING, "", "LoRAs",
                                 metadata={"editor_type": "lora_collection"}),
    ]
    
    def __init__(self, component_id=None, comp_type="SD3.5", position=None, scene=None):
        super().__init__(comp_type, position or (0, 0), scene, component_id)
    
    async def process(self, context):
        """Process SD3.5 model loading"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            variant = self.properties.get("model_variant", "sd3.5-large")
            print(f"🔧 Loading SD3.5 {variant}")
            
            # Check if we have diffusers available
            try:
                from diffusers import StableDiffusion3Pipeline
                import torch
                
                # Get model path from properties
                model_path = self.properties.get("model_path", "/home/alex/SwarmUI/Models/Stable-Diffusion/sd3.5_large.safetensors")
                
                # Check if model file exists
                from pathlib import Path
                if not Path(model_path).exists():
                    print(f"❌ SD3.5 model file not found: {model_path}")
                    self.set_status(ComponentStatus.ERROR)
                    return False
                
                print(f"🔧 Loading local SD3.5: {model_path}")
                
                # Load the pipeline from local file
                pipeline = StableDiffusion3Pipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16
                )
                
                # Move to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda" and torch.cuda.is_available():
                    pipeline = pipeline.to(device)
                
                print(f"✅ SD3.5 {variant} loaded successfully on {device}")
                
                # Set outputs
                self.set_output_data("pipeline", pipeline)
                self.set_output_data("conditioning", {
                    "prompt": self.properties.get("prompt", ""),
                    "negative_prompt": self.properties.get("negative_prompt", ""),
                    "guidance_scale": self.properties.get("guidance_scale", 7.0),
                    "num_inference_steps": self.properties.get("num_inference_steps", 28),
                    "model_type": "sd35",
                    "model_variant": variant
                })
                self.set_output_data("model_info", f"SD3.5 {variant} loaded on {device}")
                
            except ImportError:
                print("❌ Diffusers not available, cannot load SD3.5")
                self.set_status(ComponentStatus.ERROR)
                return False
            except Exception as model_error:
                print(f"❌ Failed to load SD3.5 model: {model_error}")
                print("💡 Note: SD3.5 models require diffusers>=0.30.0 and may need authentication")
                self.set_status(ComponentStatus.ERROR)
                return False
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SD35Component: {e}")
            return False
