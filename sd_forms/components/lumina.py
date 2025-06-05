"""
Lumina Control Component for SD Forms
Supports Lumina-T2X models with various control methods
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)
from ..utils.constants import DEVICE, DIFFUSERS_AVAILABLE

# Lumina model configurations
LUMINA_CONFIGS = {
    'lumina-next-t2i': {
        'name': 'Lumina-Next-T2I',
        'model_id': 'Alpha-VLLM/Lumina-Next-T2I',
        'resolution': 1024,
        'description': 'Latest Lumina text-to-image model',
        'model_type': 'base'
    },
    'lumina-next-t2i-mini': {
        'name': 'Lumina-Next-T2I-Mini',
        'model_id': 'Alpha-VLLM/Lumina-Next-T2I-Mini',
        'resolution': 512,
        'description': 'Lightweight version for faster generation',
        'model_type': 'base'
    }
}

# Lumina Control configurations
LUMINA_CONTROL_CONFIGS = {
    'lumina-canny': {
        'name': 'Lumina Canny Control',
        'model_id': 'Alpha-VLLM/Lumina-Control-Canny',
        'preprocessor': 'canny',
        'description': 'Edge-guided generation with Lumina',
        'control_type': 'canny',
        'resolution': 1024
    },
    'lumina-depth': {
        'name': 'Lumina Depth Control',
        'model_id': 'Alpha-VLLM/Lumina-Control-Depth',
        'preprocessor': 'depth_anything',
        'description': 'Depth-guided generation with Lumina',
        'control_type': 'depth',
        'resolution': 1024
    },
    'lumina-pose': {
        'name': 'Lumina Pose Control',
        'model_id': 'Alpha-VLLM/Lumina-Control-Pose',
        'preprocessor': 'dwpose',
        'description': 'Pose-guided generation with Lumina',
        'control_type': 'pose',
        'resolution': 1024
    },
    'lumina-composition': {
        'name': 'Lumina Composition',
        'model_id': 'Alpha-VLLM/Lumina-Control-Composition',
        'preprocessor': 'composition',
        'description': 'Layout and composition control',
        'control_type': 'composition',
        'resolution': 1024
    },
    'lumina-style': {
        'name': 'Lumina Style Transfer',
        'model_id': 'Alpha-VLLM/Lumina-Control-Style',
        'preprocessor': 'style_extract',
        'description': 'Style-guided generation with reference',
        'control_type': 'style',
        'resolution': 1024
    },
    'lumina-edit': {
        'name': 'Lumina Edit',
        'model_id': 'Alpha-VLLM/Lumina-Control-Edit',
        'preprocessor': 'edit_mask',
        'description': 'Instruction-based image editing',
        'control_type': 'edit',
        'resolution': 1024
    }
}

class LuminaControlComponent(VisualComponent):
    """Lumina Control Component for controlled image generation with Lumina models"""
    
    component_type = "lumina_control"
    display_name = "Lumina Control"
    category = "Control"
    icon = "✨"
    
    # Define input ports
    input_ports = [
        create_port("prompt", PortType.TEXT, PortDirection.INPUT, optional=True),
        create_port("control_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("source_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("mask_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("style_reference", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("edit_instruction", PortType.TEXT, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("control_map", PortType.IMAGE, PortDirection.OUTPUT, optional=True),
        create_port("metadata", PortType.ANY, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        create_property_definition(
            key="model_variant",
            display_name="Model Variant",
            prop_type=PropertyType.CHOICE,
            default="lumina-next-t2i",
            category="Model",
            metadata={
                "choices": list(LUMINA_CONFIGS.keys()) + list(LUMINA_CONTROL_CONFIGS.keys()),
                "descriptions": {**{k: v['description'] for k, v in LUMINA_CONFIGS.items()},
                               **{k: v['description'] for k, v in LUMINA_CONTROL_CONFIGS.items()}}
            }
        ),
        create_property_definition(
            key="control_mode",
            display_name="Control Mode",
            prop_type=PropertyType.CHOICE,
            default="none",
            category="Control",
            metadata={
                "choices": ["none", "canny", "depth", "pose", "composition", "style", "edit"],
                "descriptions": {
                    "none": "No control, pure text-to-image",
                    "canny": "Edge-based control",
                    "depth": "Depth map control",
                    "pose": "Human pose control",
                    "composition": "Layout composition control",
                    "style": "Style transfer from reference",
                    "edit": "Instruction-based editing"
                }
            }
        ),
        create_property_definition(
            key="model_path",
            display_name="Model Path",
            prop_type=PropertyType.FILE_PATH,
            default="",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.bin;*.pth",
                "description": "Path to local Lumina model (optional)"
            }
        ),
        create_property_definition(
            key="use_flash_attention",
            display_name="Use Flash Attention",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="Model",
            metadata={
                "description": "Enable Flash Attention for faster generation"
            }
        ),
        
        # Generation Settings
        create_property_definition(
            key="prompt",
            display_name="Prompt",
            prop_type=PropertyType.TEXT,
            default="",
            category="Generation",
            metadata={
                "placeholder": "Describe what you want to generate..."
            }
        ),
        create_property_definition(
            key="negative_prompt",
            display_name="Negative Prompt",
            prop_type=PropertyType.TEXT,
            default="",
            category="Generation",
            metadata={
                "placeholder": "What to avoid in generation..."
            }
        ),
        create_property_definition(
            key="num_inference_steps",
            display_name="Steps",
            prop_type=PropertyType.INTEGER,
            default=30,
            category="Generation",
            metadata={"min": 1, "max": 200}
        ),
        create_property_definition(
            key="guidance_scale",
            display_name="Guidance Scale",
            prop_type=PropertyType.FLOAT,
            default=7.5,
            category="Generation",
            metadata={"min": 1.0, "max": 30.0, "step": 0.5}
        ),
        create_property_definition(
            key="resolution",
            display_name="Resolution",
            prop_type=PropertyType.CHOICE,
            default="1024",
            category="Generation",
            metadata={
                "choices": ["512", "768", "1024", "1536", "2048"],
                "descriptions": {
                    "512": "512x512 - Fast",
                    "768": "768x768 - Medium",
                    "1024": "1024x1024 - High (Default)",
                    "1536": "1536x1536 - Very High",
                    "2048": "2048x2048 - Ultra"
                }
            }
        ),
        create_property_definition(
            key="aspect_ratio",
            display_name="Aspect Ratio",
            prop_type=PropertyType.CHOICE,
            default="1:1",
            category="Generation",
            metadata={
                "choices": ["1:1", "16:9", "9:16", "4:3", "3:4", "custom"],
                "description": "Image aspect ratio"
            }
        ),
        create_property_definition(
            key="seed",
            display_name="Seed",
            prop_type=PropertyType.INTEGER,
            default=-1,
            category="Generation",
            metadata={"description": "-1 for random"}
        ),
        create_property_definition(
            key="batch_size",
            display_name="Batch Size",
            prop_type=PropertyType.INTEGER,
            default=1,
            category="Generation",
            metadata={"min": 1, "max": 8}
        ),
        
        # Control Settings
        create_property_definition(
            key="control_strength",
            display_name="Control Strength",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Control",
            metadata={
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
                "description": "How strongly to apply control"
            }
        ),
        create_property_definition(
            key="control_start",
            display_name="Control Start",
            prop_type=PropertyType.FLOAT,
            default=0.0,
            category="Control",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "When to start applying control (0=beginning)"
            }
        ),
        create_property_definition(
            key="control_end",
            display_name="Control End",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Control",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "When to stop applying control (1=end)"
            }
        ),
        create_property_definition(
            key="use_preprocessor",
            display_name="Use Preprocessor",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="Control",
            metadata={
                "description": "Apply preprocessing to source image"
            }
        ),
        
        # Advanced Settings
        create_property_definition(
            key="scheduler",
            display_name="Scheduler",
            prop_type=PropertyType.CHOICE,
            default="flow_dpm_solver",
            category="Advanced",
            metadata={
                "choices": ["flow_dpm_solver", "flow_euler", "flow_midpoint", "ddim"],
                "descriptions": {
                    "flow_dpm_solver": "Flow-based DPM Solver (recommended)",
                    "flow_euler": "Flow-based Euler method",
                    "flow_midpoint": "Flow-based Midpoint method",
                    "ddim": "DDIM sampler"
                }
            }
        ),
        create_property_definition(
            key="cfg_type",
            display_name="CFG Type",
            prop_type=PropertyType.CHOICE,
            default="lumina_linear",
            category="Advanced",
            metadata={
                "choices": ["lumina_linear", "lumina_quadratic", "constant"],
                "description": "Classifier-free guidance schedule"
            }
        ),
        create_property_definition(
            key="temperature",
            display_name="Temperature",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Advanced",
            metadata={
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "description": "Sampling temperature"
            }
        ),
        create_property_definition(
            key="top_k",
            display_name="Top K",
            prop_type=PropertyType.INTEGER,
            default=0,
            category="Advanced",
            metadata={
                "min": 0,
                "max": 100,
                "description": "Top-k sampling (0=disabled)"
            }
        ),
        create_property_definition(
            key="top_p",
            display_name="Top P",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Advanced",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Nucleus sampling threshold"
            }
        ),
        
        # Memory Settings
        create_property_definition(
            key="enable_tiling",
            display_name="Enable Tiling",
            prop_type=PropertyType.BOOLEAN,
            default=False,
            category="Memory",
            metadata={
                "description": "Process in tiles to save memory"
            }
        ),
        create_property_definition(
            key="tile_size",
            display_name="Tile Size",
            prop_type=PropertyType.INTEGER,
            default=512,
            category="Memory",
            metadata={
                "min": 256,
                "max": 1024,
                "step": 128,
                "depends_on": {"enable_tiling": True}
            }
        ),
        create_property_definition(
            key="enable_cpu_offload",
            display_name="CPU Offload",
            prop_type=PropertyType.BOOLEAN,
            default=False,
            category="Memory",
            metadata={
                "description": "Offload to CPU to save GPU memory"
            }
        ),
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize visual component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # Lumina-specific attributes
        self.model = None
        self.preprocessor = None
        self.model_info = None
        
        # Visual settings for backward compatibility
        self.colors = {
            "LuminaControl": "#ff6b6b",  # Coral red for Lumina
        }
        self.icons = {
            "LuminaControl": "✨",
        }
        
        if self.scene:
            self.draw()
    
    def load_model(self, variant: str, model_path: Optional[str] = None):
        """Load Lumina model"""
        try:
            # Determine if base model or control model
            if variant in LUMINA_CONFIGS:
                config = LUMINA_CONFIGS[variant]
                self.model = self._load_base_lumina(config, model_path)
            elif variant in LUMINA_CONTROL_CONFIGS:
                config = LUMINA_CONTROL_CONFIGS[variant]
                self.model = self._load_control_lumina(config, model_path)
            else:
                raise ValueError(f"Unknown Lumina variant: {variant}")
            
            self.model_info = config
            return True
            
        except Exception as e:
            print(f"Error loading Lumina model: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to mock
            self.model = self._create_mock_lumina(variant)
            return True
    
    def _load_base_lumina(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load base Lumina T2I model"""
        try:
            # Try to import Lumina
            from lumina_t2x import LuminaT2I
            
            if model_path and Path(model_path).exists():
                model = LuminaT2I.from_pretrained(model_path)
            else:
                # Check common local paths
                local_paths = [
                    f"/home/alex/models/Lumina/{config['model_id'].split('/')[-1]}",
                    f"/home/alex/SwarmUI/Models/lumina/{config['model_id'].split('/')[-1]}",
                ]
                
                loaded = False
                for path in local_paths:
                    if Path(path).exists():
                        model = LuminaT2I.from_pretrained(path)
                        loaded = True
                        break
                
                if not loaded:
                    print(f"Would download from: {config['model_id']}")
                    model = self._create_mock_lumina(config['model_id'])
            
            # Configure model
            if self.properties.get("use_flash_attention", True):
                model.enable_flash_attention()
            
            return model.to(DEVICE)
            
        except ImportError:
            print("Lumina package not found, using mock")
            return self._create_mock_lumina(config['model_id'])
    
    def _load_control_lumina(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load Lumina Control model"""
        try:
            # Try to import Lumina Control
            from lumina_t2x import LuminaControlNet
            
            if model_path and Path(model_path).exists():
                model = LuminaControlNet.from_pretrained(
                    model_path,
                    control_type=config['control_type']
                )
            else:
                print(f"Would load control model: {config['model_id']}")
                model = self._create_mock_lumina_control(config)
            
            return model.to(DEVICE)
            
        except ImportError:
            return self._create_mock_lumina_control(config)
    
    def _create_mock_lumina(self, variant: str):
        """Create mock Lumina model for testing"""
        class MockLumina:
            def __init__(self, variant):
                self.variant = variant
                self.device = DEVICE
            
            def to(self, device):
                self.device = device
                return self
            
            def enable_flash_attention(self):
                print("Flash attention enabled (mock)")
            
            def generate(self, prompt, **kwargs):
                # Generate mock images
                resolution = int(kwargs.get('resolution', 1024))
                batch_size = kwargs.get('batch_size', 1)
                
                images = []
                for i in range(batch_size):
                    # Create gradient test image
                    img = Image.new('RGB', (resolution, resolution))
                    pixels = img.load()
                    
                    for y in range(resolution):
                        for x in range(resolution):
                            r = int(255 * x / resolution)
                            g = int(255 * y / resolution)
                            b = 128
                            pixels[x, y] = (r, g, b)
                    
                    images.append(img)
                
                return images
        
        return MockLumina(variant)
    
    def _create_mock_lumina_control(self, config: Dict[str, Any]):
        """Create mock Lumina Control model"""
        class MockLuminaControl:
            def __init__(self, config):
                self.config = config
                self.control_type = config.get('control_type', 'canny')
            
            def to(self, device):
                return self
            
            def generate(self, prompt, control_image, **kwargs):
                resolution = int(kwargs.get('resolution', 1024))
                
                # Create image based on control
                if isinstance(control_image, Image.Image):
                    img = control_image.resize((resolution, resolution))
                else:
                    img = Image.new('RGB', (resolution, resolution), color='purple')
                
                return [img]
        
        return MockLuminaControl(config)
    
    def load_preprocessor(self, preprocessor_type: str):
        """Load appropriate preprocessor for control type"""
        if preprocessor_type == 'canny':
            self.preprocessor = self._canny_preprocessor
        elif preprocessor_type == 'depth_anything':
            self.preprocessor = self._depth_anything_preprocessor
        elif preprocessor_type == 'dwpose':
            self.preprocessor = self._dwpose_preprocessor
        elif preprocessor_type == 'composition':
            self.preprocessor = self._composition_preprocessor
        elif preprocessor_type == 'style_extract':
            self.preprocessor = self._style_extract_preprocessor
        elif preprocessor_type == 'edit_mask':
            self.preprocessor = self._edit_mask_preprocessor
        else:
            self.preprocessor = None
    
    def _canny_preprocessor(self, image: Image.Image) -> Image.Image:
        """Canny edge detection for Lumina"""
        import cv2
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny with Lumina-optimized thresholds
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def _depth_anything_preprocessor(self, image: Image.Image) -> Image.Image:
        """Depth Anything preprocessor for Lumina"""
        try:
            from transformers import pipeline
            
            # Use Depth Anything model
            depth_estimator = pipeline(
                'depth-estimation',
                model='LiheYoung/depth-anything-base-hf'
            )
            
            depth = depth_estimator(image)['depth']
            
            # Normalize for Lumina
            depth_array = np.array(depth)
            depth_norm = ((depth_array - depth_array.min()) / 
                         (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
            
            # Apply Lumina-specific colormap
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(depth_rgb)
            
        except:
            # Fallback gradient depth
            return self._create_gradient_depth(image)
    
    def _dwpose_preprocessor(self, image: Image.Image) -> Image.Image:
        """DWPose preprocessor for Lumina"""
        try:
            from controlnet_aux import DWposeDetector
            
            dwpose = DWposeDetector.from_pretrained("yzd-v/DWPose")
            result = dwpose(image)
            return result
        except:
            # Fallback: black image with pose hint
            result = Image.new('RGB', image.size, color='black')
            # Add some pose indicators
            from PIL import ImageDraw
            draw = ImageDraw.Draw(result)
            # Simple stick figure
            w, h = image.size
            draw.ellipse([w//2-20, h//4-20, w//2+20, h//4+20], fill='white')  # Head
            draw.line([w//2, h//4+20, w//2, h*3//4], fill='white', width=3)  # Body
            return result
    
    def _composition_preprocessor(self, image: Image.Image) -> Image.Image:
        """Extract composition/layout for Lumina"""
        # Segment image into regions
        from sklearn.cluster import KMeans
        
        # Resize for faster processing
        thumb = image.resize((256, 256))
        img_array = np.array(thumb)
        
        # Flatten pixels
        pixels = img_array.reshape(-1, 3)
        
        # Cluster into regions
        kmeans = KMeans(n_clusters=8, random_state=42)
        kmeans.fit(pixels)
        
        # Create composition map
        labels = kmeans.labels_.reshape(256, 256)
        
        # Color each region
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 255, 255)
        ]
        
        composition = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(8):
            composition[labels == i] = colors[i]
        
        # Resize back
        result = Image.fromarray(composition).resize(image.size)
        return result
    
    def _style_extract_preprocessor(self, image: Image.Image) -> Dict[str, Any]:
        """Extract style features for Lumina style transfer"""
        # Extract various style elements
        style_data = {
            'reference_image': image,
            'color_palette': self._extract_colors(image),
            'texture_features': self._extract_texture(image),
            'composition': self._composition_preprocessor(image)
        }
        
        return style_data
    
    def _edit_mask_preprocessor(self, image: Image.Image, mask: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Prepare edit mask for Lumina instruction-based editing"""
        if mask is None:
            # Create default mask (entire image)
            mask = Image.new('L', image.size, 255)
        
        return {
            'image': image,
            'mask': mask,
            'regions': self._identify_edit_regions(image, mask)
        }
    
    def _extract_colors(self, image: Image.Image, n_colors: int = 8) -> List[Tuple[int, int, int]]:
        """Extract dominant colors"""
        thumb = image.copy()
        thumb.thumbnail((150, 150))
        
        if thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
        
        quantized = thumb.quantize(colors=n_colors)
        palette = quantized.getpalette()
        
        colors = []
        for i in range(n_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            colors.append((r, g, b))
        
        return colors
    
    def _extract_texture(self, image: Image.Image) -> np.ndarray:
        """Extract texture features"""
        import cv2
        
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Compute texture using Gabor filters
        texture_features = []
        
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_features.append(np.mean(filtered))
        
        return np.array(texture_features)
    
    def _identify_edit_regions(self, image: Image.Image, mask: Image.Image) -> List[Dict[str, Any]]:
        """Identify regions for editing"""
        import cv2
        
        mask_array = np.array(mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                'bbox': (x, y, w, h),
                'area': cv2.contourArea(contour),
                'center': (x + w // 2, y + h // 2)
            })
        
        return regions
    
    def _create_gradient_depth(self, image: Image.Image) -> Image.Image:
        """Create simple gradient depth map"""
        width, height = image.size
        depth = Image.new('RGB', (width, height))
        pixels = depth.load()
        
        for y in range(height):
            gray_value = int(255 * (1 - y / height))
            for x in range(width):
                pixels[x, y] = (gray_value, gray_value, gray_value)
        
        return depth
    
    async def process(self, context) -> bool:
        """Process Lumina generation with control"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get inputs
            prompt = self.get_input_data("prompt") or self.properties.get("prompt", "")
            control_image = self.get_input_data("control_image")
            source_image = self.get_input_data("source_image")
            mask_image = self.get_input_data("mask_image")
            style_reference = self.get_input_data("style_reference")
            edit_instruction = self.get_input_data("edit_instruction")
            
            # Load model if needed
            if self.model is None:
                variant = self.properties.get("model_variant", "lumina-next-t2i")
                if not self.load_model(variant):
                    raise ValueError("Failed to load Lumina model")
            
            # Determine control mode
            control_mode = self.properties.get("control_mode", "none")
            
            # Get resolution
            resolution = int(self.properties.get("resolution", "1024"))
            aspect_ratio = self.properties.get("aspect_ratio", "1:1")
            width, height = self._calculate_dimensions(resolution, aspect_ratio)
            
            # Prepare generation parameters
            gen_params = {
                "prompt": prompt,
                "negative_prompt": self.properties.get("negative_prompt", ""),
                "num_inference_steps": self.properties.get("num_inference_steps", 30),
                "guidance_scale": self.properties.get("guidance_scale", 7.5),
                "width": width,
                "height": height,
                "batch_size": self.properties.get("batch_size", 1),
                "temperature": self.properties.get("temperature", 1.0),
                "top_k": self.properties.get("top_k", 0),
                "top_p": self.properties.get("top_p", 1.0),
                "scheduler": self.properties.get("scheduler", "flow_dpm_solver"),
                "cfg_type": self.properties.get("cfg_type", "lumina_linear"),
            }
            
            # Set seed
            seed = self.properties.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            gen_params["seed"] = seed
            
            # Apply control based on mode
            if control_mode != "none":
                control_data = await self._apply_control(
                    control_mode, 
                    control_image, 
                    source_image,
                    mask_image,
                    style_reference,
                    edit_instruction
                )
                gen_params.update(control_data)
            
            # Generate images
            print(f"Generating with Lumina {self.model_info.get('name', 'Unknown')}")
            
            # Add progress callback if available
            if hasattr(context, 'preview_callback'):
                gen_params["callback"] = self._create_progress_callback(context)
            
            # Generate
            if hasattr(self.model, 'generate'):
                images = self.model.generate(**gen_params)
            else:
                # Mock generation
                images = self.model(prompt, **gen_params)
            
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images]
            
            # Set outputs
            self.set_output_data("images", images)
            
            if control_mode != "none" and "control_map" in locals():
                self.set_output_data("control_map", control_data.get("control_map"))
            
            self.set_output_data("metadata", {
                "model": self.model_info.get('name', 'Unknown'),
                "control_mode": control_mode,
                "seed": seed,
                "resolution": f"{width}x{height}",
                "steps": gen_params["num_inference_steps"],
                "cfg": gen_params["guidance_scale"],
            })
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in LuminaControlComponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _apply_control(self, control_mode: str, control_image: Optional[Image.Image],
                           source_image: Optional[Image.Image], mask_image: Optional[Image.Image],
                           style_reference: Optional[Image.Image], edit_instruction: Optional[str]) -> Dict[str, Any]:
        """Apply control based on mode"""
        control_data = {
            "control_strength": self.properties.get("control_strength", 1.0),
            "control_start": self.properties.get("control_start", 0.0),
            "control_end": self.properties.get("control_end", 1.0),
        }
        
        # Load preprocessor if needed
        if self.properties.get("use_preprocessor", True) and control_image is None:
            control_config = next((c for c in LUMINA_CONTROL_CONFIGS.values() 
                                 if c['control_type'] == control_mode), None)
            if control_config:
                self.load_preprocessor(control_config['preprocessor'])
        
        if control_mode == "canny":
            if control_image is None and source_image:
                control_image = self._canny_preprocessor(source_image)
            control_data["control_image"] = control_image
            control_data["control_type"] = "canny"
            
        elif control_mode == "depth":
            if control_image is None and source_image:
                control_image = self._depth_anything_preprocessor(source_image)
            control_data["control_image"] = control_image
            control_data["control_type"] = "depth"
            
        elif control_mode == "pose":
            if control_image is None and source_image:
                control_image = self._dwpose_preprocessor(source_image)
            control_data["control_image"] = control_image
            control_data["control_type"] = "pose"
            
        elif control_mode == "composition":
            if control_image is None and source_image:
                control_image = self._composition_preprocessor(source_image)
            control_data["control_image"] = control_image
            control_data["control_type"] = "composition"
            
        elif control_mode == "style":
            if style_reference:
                style_data = self._style_extract_preprocessor(style_reference)
                control_data.update(style_data)
                control_data["control_type"] = "style"
            
        elif control_mode == "edit":
            if source_image:
                edit_data = self._edit_mask_preprocessor(source_image, mask_image)
                control_data.update(edit_data)
                control_data["edit_instruction"] = edit_instruction or ""
                control_data["control_type"] = "edit"
        
        control_data["control_map"] = control_image
        return control_data
    
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
        else:  # custom
            return resolution, resolution
    
    def _create_progress_callback(self, context):
        """Create progress callback for Lumina"""
        def callback(step: int, total_steps: int, latents: Any):
            progress = int((step / total_steps) * 100)
            
            # Try to decode preview if possible
            if hasattr(self.model, 'decode_latents'):
                try:
                    preview = self.model.decode_latents(latents)
                    if isinstance(preview, list):
                        preview = preview[0]
                    context.preview_callback(preview, progress, step)
                except:
                    pass
        
        return callback
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Lumina model unloaded")
