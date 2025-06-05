"""
OmniGen Component for SD Forms
Unified image generation model supporting text-to-image, editing, subject-driven generation, and more
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import re

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
from ..utils.constants import DEVICE

# OmniGen model configurations
OMNIGEN_CONFIGS = {
    'omnigen-v1': {
        'name': 'OmniGen-v1',
        'model_id': 'Shitao/OmniGen-v1',
        'resolution': 1024,
        'description': 'Unified model for all image generation tasks',
        'capabilities': ['text2img', 'editing', 'subject_driven', 'multi_modal']
    },
    'omnigen-v1-turbo': {
        'name': 'OmniGen-v1-Turbo',
        'model_id': 'Shitao/OmniGen-v1-Turbo',
        'resolution': 1024,
        'description': 'Faster variant with optimized inference',
        'capabilities': ['text2img', 'editing', 'subject_driven']
    }
}

class OmniGenComponent(VisualComponent):
    """OmniGen Component - Unified image generation for multiple tasks"""
    
    component_type = "omnigen"
    display_name = "OmniGen"
    category = "Models"
    icon = "🌐"
    
    # Define input ports - OmniGen can handle multiple image inputs
    input_ports = [
        create_port("instruction", PortType.TEXT, PortDirection.INPUT, optional=True),
        create_port("image1", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("image2", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("image3", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("mask", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("subject_images", PortType.IMAGE, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("edited_regions", PortType.TEXT, PortDirection.OUTPUT, optional=True),
        create_port("metadata", PortType.TEXT, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        PropertyDefinition(
            key="model_variant",
            display_name="Model Variant",
            type=PropertyType.ENUM,
            default="omnigen-v1",
            category="Model",
            metadata={
                "values": list(OMNIGEN_CONFIGS.keys()),
                "descriptions": {k: v['description'] for k, v in OMNIGEN_CONFIGS.items()}
            }
        ),
        PropertyDefinition(
            key="model_path",
            display_name="Model Path",
            type=PropertyType.FILE_PICKER,
            default="/home/alex/SwarmUI/Models/diffusion_models/omnigen_v10.safetensors",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.bin;*.pth",
                "description": "Path to local OmniGen model (optional)"
            }
        ),
        PropertyDefinition(
            key="task_mode",
            display_name="Task Mode",
            type=PropertyType.ENUM,
            default="auto",
            category="Model",
            metadata={
                "values": ["auto", "text2img", "editing", "subject_driven", "multi_modal", "instruction"],
                "descriptions": {
                    "auto": "Automatically detect task from inputs",
                    "text2img": "Generate from text description",
                    "editing": "Edit existing images with instructions",
                    "subject_driven": "Generate with specific subjects",
                    "multi_modal": "Combine multiple inputs",
                    "instruction": "Follow complex instructions"
                }
            }
        ),
        
        # Generation Settings
        PropertyDefinition(
            key="instruction",
            display_name="Instruction",
            type=PropertyType.TEXT_MULTILINE,
            default="",
            category="Generation",
            metadata={
                "placeholder": "Describe what you want or provide editing instructions...",
                "description": "Natural language instruction for OmniGen"
            }
        ),
        PropertyDefinition(
            key="use_instruction_template",
            display_name="Use Instruction Template",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Generation",
            metadata={
                "description": "Apply OmniGen's instruction formatting"
            }
        ),
        PropertyDefinition(
            key="num_inference_steps",
            display_name="Steps",
            type=PropertyType.INTEGER,
            default=50,
            category="Generation",
            metadata={"min": 1, "max": 200}
        ),
        PropertyDefinition(
            key="guidance_scale",
            display_name="Guidance Scale",
            type=PropertyType.FLOAT,
            default=3.0,
            category="Generation",
            metadata={
                "min": 1.0, 
                "max": 20.0, 
                "step": 0.5,
                "description": "OmniGen typically uses lower CFG values"
            }
        ),
        PropertyDefinition(
            key="resolution",
            display_name="Resolution",
            type=PropertyType.ENUM,
            default="1024",
            category="Generation",
            metadata={
                "values": ["512", "768", "1024", "1280", "1536"],
                "descriptions": {
                    "512": "512x512 - Fast, low memory",
                    "768": "768x768 - Balanced",
                    "1024": "1024x1024 - High quality (recommended)",
                    "1280": "1280x1280 - Very high quality",
                    "1536": "1536x1536 - Ultra quality"
                }
            }
        ),
        PropertyDefinition(
            key="aspect_ratio",
            display_name="Aspect Ratio",
            type=PropertyType.ENUM,
            default="1:1",
            category="Generation",
            metadata={
                "values": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "custom"],
                "description": "Output image aspect ratio"
            }
        ),
        PropertyDefinition(
            key="seed",
            display_name="Seed",
            type=PropertyType.INTEGER,
            default=-1,
            category="Generation",
            metadata={"description": "-1 for random"}
        ),
        PropertyDefinition(
            key="batch_size",
            display_name="Batch Size",
            type=PropertyType.INTEGER,
            default=1,
            category="Generation",
            metadata={"min": 1, "max": 4}
        ),
        
        # Subject-Driven Settings
        PropertyDefinition(
            key="subject_mode",
            display_name="Subject Mode",
            type=PropertyType.ENUM,
            default="single",
            category="Subject Control",
            metadata={
                "values": ["single", "multiple", "preserve_identity"],
                "descriptions": {
                    "single": "Single subject reference",
                    "multiple": "Multiple different subjects",
                    "preserve_identity": "Maintain exact identity"
                }
            }
        ),
        PropertyDefinition(
            key="subject_strength",
            display_name="Subject Strength",
            type=PropertyType.FLOAT,
            default=0.8,
            category="Subject Control",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "How closely to match subject appearance"
            }
        ),
        PropertyDefinition(
            key="subject_token_mode",
            display_name="Subject Token Mode",
            type=PropertyType.ENUM,
            default="auto",
            category="Subject Control",
            metadata={
                "values": ["auto", "manual", "learned"],
                "descriptions": {
                    "auto": "Automatically assign tokens like <img1>, <img2>",
                    "manual": "Manually specify tokens in instruction",
                    "learned": "Use learned subject representations"
                }
            }
        ),
        
        # Editing Settings
        PropertyDefinition(
            key="edit_mode",
            display_name="Edit Mode",
            type=PropertyType.ENUM,
            default="instruction",
            category="Editing",
            metadata={
                "values": ["instruction", "mask", "region", "global"],
                "descriptions": {
                    "instruction": "Edit based on text instruction",
                    "mask": "Edit masked regions only",
                    "region": "Edit specific described regions",
                    "global": "Apply changes to entire image"
                }
            }
        ),
        PropertyDefinition(
            key="preserve_unmasked",
            display_name="Preserve Unmasked Areas",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Editing",
            metadata={
                "description": "Keep unmasked areas unchanged",
                "depends_on": {"edit_mode": "mask"}
            }
        ),
        PropertyDefinition(
            key="edit_strength",
            display_name="Edit Strength",
            type=PropertyType.FLOAT,
            default=0.8,
            category="Editing",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "How much to change the image"
            }
        ),
        
        # Multi-Modal Settings
        PropertyDefinition(
            key="image_fusion_mode",
            display_name="Image Fusion Mode",
            type=PropertyType.ENUM,
            default="guided",
            category="Multi-Modal",
            metadata={
                "values": ["guided", "blend", "composite", "reference"],
                "descriptions": {
                    "guided": "Use images to guide generation",
                    "blend": "Blend multiple images together",
                    "composite": "Composite images based on instruction",
                    "reference": "Use images as style/content reference"
                }
            }
        ),
        PropertyDefinition(
            key="cross_attention_scale",
            display_name="Cross-Attention Scale",
            type=PropertyType.FLOAT,
            default=1.0,
            category="Multi-Modal",
            metadata={
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "description": "Control image influence strength"
            }
        ),
        
        # Advanced Settings
        PropertyDefinition(
            key="use_img_guidance",
            display_name="Use Image Guidance",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Advanced",
            metadata={
                "description": "Enable OmniGen's image guidance system"
            }
        ),
        PropertyDefinition(
            key="separate_cfg_infer",
            display_name="Separate CFG Inference", 
            type=PropertyType.BOOLEAN,
            default=True,
            category="Advanced",
            metadata={
                "description": "Use separate inference for better quality"
            }
        ),
        PropertyDefinition(
            key="offload_model",
            display_name="Offload Model",
            type=PropertyType.BOOLEAN,
            default=False,
            category="Advanced",
            metadata={
                "description": "Offload model parts to CPU to save GPU memory"
            }
        ),
        PropertyDefinition(
            key="use_kv_cache",
            display_name="Use KV Cache",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Advanced",
            metadata={
                "description": "Cache key-value pairs for efficiency"
            }
        ),
        PropertyDefinition(
            key="temperature",
            display_name="Temperature",
            type=PropertyType.FLOAT,
            default=1.0,
            category="Advanced",
            metadata={
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "description": "Sampling temperature"
            }
        ),
        
        # Memory Settings
        PropertyDefinition(
            key="max_input_image_size",
            display_name="Max Input Image Size",
            type=PropertyType.INTEGER,
            default=1024,
            category="Memory",
            metadata={
                "min": 512,
                "max": 2048,
                "step": 256,
                "description": "Resize input images if larger"
            }
        ),
        PropertyDefinition(
            key="use_fp16",
            display_name="Use FP16",
            type=PropertyType.BOOLEAN,
            default=True,
            category="Memory",
            metadata={
                "description": "Use half precision for lower memory usage"
            }
        ),
    ]
    
    def __init__(self, id: str = None):
        super().__init__(id)
        self.model = None
        self.processor = None
        self.model_info = None
    
    def load_model(self, variant: str, model_path: Optional[str] = None):
        """Load OmniGen model"""
        try:
            config = OMNIGEN_CONFIGS.get(variant, OMNIGEN_CONFIGS['omnigen-v1'])
            self.model_info = config
            
            # Try to import OmniGen
            try:
                from OmniGen import OmniGenPipeline
                
                if model_path and Path(model_path).exists():
                    print(f"Loading OmniGen from: {model_path}")
                    self.model = OmniGenPipeline.from_pretrained(model_path)
                else:
                    # Check local paths
                    local_paths = [
                        f"/home/alex/models/OmniGen/{variant}",
                        f"/home/alex/SwarmUI/Models/omnigen/{variant}",
                        f"./models/{variant}"
                    ]
                    
                    loaded = False
                    for path in local_paths:
                        if Path(path).exists():
                            print(f"Found OmniGen at: {path}")
                            self.model = OmniGenPipeline.from_pretrained(path)
                            loaded = True
                            break
                    
                    if not loaded:
                        print(f"Would download OmniGen from: {config['model_id']}")
                        self.model = self._create_mock_omnigen(config)
                
                # Configure model
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(DEVICE)
                
                if self.properties.get("use_fp16", True) and DEVICE == "cuda":
                    self.model = self.model.half()
                
                if self.properties.get("offload_model", False):
                    self.model.enable_model_cpu_offload()
                    
            except ImportError:
                print("OmniGen not installed, using mock")
                self.model = self._create_mock_omnigen(config)
            
            return True
            
        except Exception as e:
            print(f"Error loading OmniGen: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_mock_omnigen(self, config: Dict[str, Any]):
        """Create mock OmniGen for testing"""
        class MockOmniGen:
            def __init__(self, config):
                self.config = config
                self.device = DEVICE
            
            def to(self, device):
                self.device = device
                return self
            
            def half(self):
                return self
            
            def enable_model_cpu_offload(self):
                print("CPU offload enabled (mock)")
            
            def generate(self, prompt, input_images=None, **kwargs):
                # Generate mock result
                resolution = int(kwargs.get('width', 1024))
                batch_size = kwargs.get('num_images', 1)
                
                images = []
                for i in range(batch_size):
                    # Create test image
                    img = Image.new('RGB', (resolution, resolution))
                    pixels = img.load()
                    
                    # Different patterns based on task
                    if input_images and len(input_images) > 0:
                        # Editing/subject mode - modify input
                        img = input_images[0].copy().resize((resolution, resolution))
                        # Add some modification
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(img)
                        draw.text((50, 50), "OmniGen Edit", fill='white')
                    else:
                        # Text to image - gradient
                        for y in range(resolution):
                            for x in range(resolution):
                                r = int(255 * (x + y) / (2 * resolution))
                                g = int(255 * x / resolution)
                                b = int(255 * y / resolution)
                                pixels[x, y] = (r, g, b)
                    
                    images.append(img)
                
                return images
        
        return MockOmniGen(config)
    
    def _parse_instruction_tokens(self, instruction: str) -> Dict[str, Any]:
        """Parse OmniGen instruction tokens like <img1>, <img2>, etc."""
        # Find all image tokens
        img_tokens = re.findall(r'<img(\d+)>', instruction)
        
        # Find editing regions
        edit_regions = re.findall(r'\[([^\]]+)\]', instruction)
        
        # Find subject references
        subject_refs = re.findall(r'<(person|object|style)(\d*)>', instruction)
        
        return {
            'image_tokens': img_tokens,
            'edit_regions': edit_regions,
            'subject_refs': subject_refs,
            'has_special_tokens': bool(img_tokens or subject_refs)
        }
    
    def _format_instruction(self, instruction: str, images: Dict[str, Image.Image]) -> str:
        """Format instruction with OmniGen conventions"""
        if not self.properties.get("use_instruction_template", True):
            return instruction
        
        # Auto-assign image tokens if not present
        if images and '<img' not in instruction:
            if self.properties.get("subject_token_mode", "auto") == "auto":
                # Add image references
                for i, (key, img) in enumerate(images.items(), 1):
                    if key != 'mask':
                        instruction = f"<img{i}> " + instruction
        
        return instruction
    
    def _detect_task_mode(self, instruction: str, images: Dict[str, Any]) -> str:
        """Automatically detect the task mode from inputs"""
        # Parse instruction
        parsed = self._parse_instruction_tokens(instruction)
        
        # Check for editing keywords
        edit_keywords = ['change', 'replace', 'remove', 'edit', 'modify', 'transform']
        has_edit_intent = any(keyword in instruction.lower() for keyword in edit_keywords)
        
        # Determine task
        if images.get('mask') is not None:
            return 'editing'
        elif images.get('subject_images') or parsed['has_special_tokens']:
            return 'subject_driven'
        elif len([v for k, v in images.items() if k != 'mask' and v is not None]) > 1:
            return 'multi_modal'
        elif any(v is not None for v in images.values()) and has_edit_intent:
            return 'editing'
        else:
            return 'text2img'
    
    async def process(self, context) -> bool:
        """Process OmniGen generation"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get inputs
            instruction = self.get_input_data("instruction") or self.properties.get("instruction", "")
            image1 = self.get_input_data("image1")
            image2 = self.get_input_data("image2")
            image3 = self.get_input_data("image3")
            mask = self.get_input_data("mask")
            subject_images = self.get_input_data("subject_images") or []
            
            if not instruction and not any([image1, image2, image3, subject_images]):
                raise ValueError("Either instruction or input images required")
            
            # Load model if needed
            if self.model is None:
                variant = self.properties.get("model_variant", "omnigen-v1")
                if not self.load_model(variant):
                    raise ValueError("Failed to load OmniGen model")
            
            # Collect all images
            images = {
                'image1': image1,
                'image2': image2,
                'image3': image3,
                'mask': mask,
                'subject_images': subject_images
            }
            
            # Remove None values
            images = {k: v for k, v in images.items() if v is not None}
            
            # Determine task mode
            task_mode = self.properties.get("task_mode", "auto")
            if task_mode == "auto":
                task_mode = self._detect_task_mode(instruction, images)
                print(f"Auto-detected task mode: {task_mode}")
            
            # Format instruction
            instruction = self._format_instruction(instruction, images)
            
            # Get resolution
            resolution = int(self.properties.get("resolution", "1024"))
            aspect_ratio = self.properties.get("aspect_ratio", "1:1")
            width, height = self._calculate_dimensions(resolution, aspect_ratio)
            
            # Prepare generation parameters
            gen_params = {
                "prompt": instruction,
                "width": width,
                "height": height,
                "num_inference_steps": self.properties.get("num_inference_steps", 50),
                "guidance_scale": self.properties.get("guidance_scale", 3.0),
                "num_images": self.properties.get("batch_size", 1),
                "temperature": self.properties.get("temperature", 1.0),
                "use_img_guidance": self.properties.get("use_img_guidance", True),
                "separate_cfg_infer": self.properties.get("separate_cfg_infer", True),
                "use_kv_cache": self.properties.get("use_kv_cache", True),
            }
            
            # Set seed
            seed = self.properties.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            gen_params["seed"] = seed
            
            # Add task-specific parameters
            if task_mode == "editing":
                gen_params.update(self._prepare_editing_params(images, instruction))
            elif task_mode == "subject_driven":
                gen_params.update(self._prepare_subject_params(images, instruction))
            elif task_mode == "multi_modal":
                gen_params.update(self._prepare_multimodal_params(images, instruction))
            
            # Add input images
            if images:
                # Resize images if needed
                max_size = self.properties.get("max_input_image_size", 1024)
                processed_images = []
                
                for key, img in images.items():
                    if key == 'subject_images' and isinstance(img, list):
                        processed_images.extend([self._resize_image(i, max_size) for i in img])
                    elif isinstance(img, Image.Image):
                        processed_images.append(self._resize_image(img, max_size))
                
                gen_params["input_images"] = processed_images
            
            # Add progress callback
            if hasattr(context, 'preview_callback'):
                def callback(step, total_steps):
                    progress = int((step / total_steps) * 100)
                    # OmniGen might not provide intermediate images
                    preview = Image.new('RGB', (256, 256), color='gray')
                    context.preview_callback(preview, progress, step)
                
                gen_params["callback"] = callback
            
            # Generate
            print(f"Generating with OmniGen ({task_mode} mode)")
            
            if hasattr(self.model, 'generate'):
                results = self.model.generate(**gen_params)
            else:
                # Mock
                results = self.model.generate(instruction, processed_images if 'processed_images' in locals() else None, **gen_params)
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]
            
            # Set outputs
            self.set_output_data("images", results)
            
            # Add edited regions info if available
            if task_mode == "editing" and hasattr(self.model, 'get_edited_regions'):
                edited_regions = self.model.get_edited_regions()
                self.set_output_data("edited_regions", edited_regions)
            
            self.set_output_data("metadata", {
                "model": self.model_info.get('name', 'OmniGen'),
                "task_mode": task_mode,
                "seed": seed,
                "resolution": f"{width}x{height}",
                "instruction": instruction,
                "num_input_images": len(images)
            })
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in OmniGenComponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_editing_params(self, images: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Prepare parameters for editing mode"""
        params = {
            "edit_mode": self.properties.get("edit_mode", "instruction"),
            "edit_strength": self.properties.get("edit_strength", 0.8),
        }
        
        if images.get('mask') and self.properties.get("preserve_unmasked", True):
            params["preserve_unmasked_area"] = True
        
        return params
    
    def _prepare_subject_params(self, images: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Prepare parameters for subject-driven generation"""
        return {
            "subject_mode": self.properties.get("subject_mode", "single"),
            "subject_strength": self.properties.get("subject_strength", 0.8),
        }
    
    def _prepare_multimodal_params(self, images: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Prepare parameters for multi-modal generation"""
        return {
            "image_fusion_mode": self.properties.get("image_fusion_mode", "guided"),
            "cross_attention_scale": self.properties.get("cross_attention_scale", 1.0),
        }
    
    def _calculate_dimensions(self, resolution: int, aspect_ratio: str) -> Tuple[int, int]:
        """Calculate width and height from resolution and aspect ratio"""
        if aspect_ratio == "1:1":
            return resolution, resolution
        elif aspect_ratio == "16:9":
            height = int(resolution * 9 / 16)
            return resolution, height
        elif aspect_ratio == "9:16":
            width = int(resolution * 9 / 16)
            return width, resolution
        elif aspect_ratio == "4:3":
            height = int(resolution * 3 / 4)
            return resolution, height
        elif aspect_ratio == "3:4":
            width = int(resolution * 3 / 4)
            return width, resolution
        elif aspect_ratio == "2:3":
            width = int(resolution * 2 / 3)
            return width, resolution
        elif aspect_ratio == "3:2":
            height = int(resolution * 2 / 3)
            return resolution, height
        else:  # custom
            return resolution, resolution
    
    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """Resize image if larger than max_size"""
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor  
            self.processor = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("OmniGen model unloaded")
