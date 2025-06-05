"""
Text encoder components for CLIP and T5-XXL
"""

from typing import Dict, Any, List
from ..core import Component, Port, PropertyDefinition, PropertyType, ComponentStatus
from ..core.process_context import ProcessContext


class CLIPComponent(Component):
    """CLIP text encoder component for Stable Diffusion"""
    
    component_type = "clip"
    display_name = "CLIP Text Encoder"
    category = "Processing"
    
    input_ports = [
        Port("text", "text", "INPUT")
    ]
    
    output_ports = [
        Port("conditioning", "conditioning", "OUTPUT")
    ]
    
    property_definitions = [
        PropertyDefinition(
            "clip_path", "CLIP Model Path", PropertyType.STRING,
            default="/home/alex/SwarmUI/Models/clip/clip_l.safetensors",
            category="Model",
            metadata={"editor_type": "file_picker", "file_types": [".safetensors", ".bin"]}
        ),
        PropertyDefinition(
            "clip_type", "CLIP Type", PropertyType.STRING,
            default="clip_l",
            category="Model",
            metadata={"editor_type": "choice", "choices": ["clip_l", "clip_g"]}
        ),
    ]
    
    async def process(self, context: ProcessContext) -> bool:
        """Process text through CLIP encoder"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get input text (from prompt or other source)
            text_input = self.get_input_data("text")
            if text_input is None:
                # If no direct text input, use the prompt from properties
                text_input = self.get_property("prompt", "")
            
            # Load CLIP model if needed
            clip_path = self.get_property("clip_path")
            clip_type = self.get_property("clip_type", "clip_l")
            
            # For now, just pass through the text
            # In a real implementation, this would encode the text through CLIP
            conditioning = {
                "text": text_input,
                "clip_path": clip_path,
                "clip_type": clip_type
            }
            
            # Set output
            self.set_output_data("conditioning", conditioning)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            print(f"CLIP processing error: {e}")
            self.set_status(ComponentStatus.ERROR)
            return False


class T5XXLComponent(Component):
    """T5-XXL text encoder component for Flux models"""
    
    component_type = "t5xxl"
    display_name = "T5-XXL Text Encoder"
    category = "Processing"
    
    input_ports = [
        Port("text", "text", "INPUT")
    ]
    
    output_ports = [
        Port("conditioning", "conditioning", "OUTPUT")
    ]
    
    property_definitions = [
        PropertyDefinition(
            "clip_path", "T5-XXL Model Path", PropertyType.STRING,
            default="/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors",
            category="Model",
            metadata={"editor_type": "file_picker", "file_types": [".safetensors", ".gguf"]}
        ),
        PropertyDefinition(
            "clip_type", "T5 Type", PropertyType.STRING,
            default="t5xxl",
            category="Model",
            metadata={"editor_type": "choice", "choices": ["t5xxl", "t5_v1_1_xxl"]}
        ),
    ]
    
    async def process(self, context: ProcessContext) -> bool:
        """Process text through T5-XXL encoder"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get input text
            text_input = self.get_input_data("text")
            if text_input is None:
                text_input = self.get_property("prompt", "")
            
            # Load T5-XXL model if needed
            clip_path = self.get_property("clip_path")
            clip_type = self.get_property("clip_type", "t5xxl")
            
            # For now, just pass through the text
            # In a real implementation, this would encode the text through T5-XXL
            conditioning = {
                "text": text_input,
                "clip_path": clip_path,
                "clip_type": clip_type
            }
            
            # Set output
            self.set_output_data("conditioning", conditioning)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            print(f"T5-XXL processing error: {e}")
            self.set_status(ComponentStatus.ERROR)
            return False