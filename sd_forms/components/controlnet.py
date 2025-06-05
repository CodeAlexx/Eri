"""
ControlNet Component for SD Forms
Supports multiple ControlNet models and preprocessors for controlled image generation
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)
from ..utils.constants import DEVICE, DIFFUSERS_AVAILABLE

# ControlNet model configurations
CONTROLNET_CONFIGS = {
    'canny': {
        'name': 'Canny Edge',
        'model_id': 'lllyasviel/sd-controlnet-canny',
        'preprocessor': 'canny',
        'description': 'Edge detection control',
        'default_threshold1': 100,
        'default_threshold2': 200
    },
    'depth': {
        'name': 'Depth',
        'model_id': 'lllyasviel/sd-controlnet-depth',
        'preprocessor': 'midas',
        'description': 'Depth map control'
    },
    'openpose': {
        'name': 'OpenPose',
        'model_id': 'lllyasviel/sd-controlnet-openpose',
        'preprocessor': 'openpose',
        'description': 'Human pose control'
    },
    'mlsd': {
        'name': 'M-LSD Lines',
        'model_id': 'lllyasviel/sd-controlnet-mlsd',
        'preprocessor': 'mlsd',
        'description': 'Straight line detection'
    },
    'normal': {
        'name': 'Normal Map',
        'model_id': 'lllyasviel/sd-controlnet-normal',
        'preprocessor': 'normal_bae',
        'description': 'Surface normal control'
    },
    'scribble': {
        'name': 'Scribble',
        'model_id': 'lllyasviel/sd-controlnet-scribble',
        'preprocessor': 'scribble',
        'description': 'Scribble/sketch control'
    },
    'seg': {
        'name': 'Segmentation',
        'model_id': 'lllyasviel/sd-controlnet-seg',
        'preprocessor': 'uniformer',
        'description': 'Semantic segmentation'
    },
    'shuffle': {
        'name': 'Shuffle',
        'model_id': 'lllyasviel/control_v11e_sd15_shuffle',
        'preprocessor': 'shuffle',
        'description': 'Content shuffle control'
    },
    'tile': {
        'name': 'Tile',
        'model_id': 'lllyasviel/control_v11f1e_sd15_tile',
        'preprocessor': 'tile_resample',
        'description': 'Tile/detail control'
    },
    'ip2p': {
        'name': 'InstructPix2Pix',
        'model_id': 'lllyasviel/control_v11e_sd15_ip2p',
        'preprocessor': 'none',
        'description': 'Instruction-based editing'
    }
}

# SDXL ControlNet configs
CONTROLNET_SDXL_CONFIGS = {
    'canny-sdxl': {
        'name': 'Canny SDXL',
        'model_id': 'diffusers/controlnet-canny-sdxl-1.0',
        'preprocessor': 'canny',
        'description': 'Edge detection for SDXL'
    },
    'depth-sdxl': {
        'name': 'Depth SDXL',
        'model_id': 'diffusers/controlnet-depth-sdxl-1.0',
        'preprocessor': 'depth',
        'description': 'Depth map for SDXL'
    }
}

# Flux Control configs
FLUX_CONTROL_CONFIGS = {
    'flux-fill': {
        'name': 'Flux Fill',
        'model_id': 'black-forest-labs/FLUX.1-Fill-dev',
        'preprocessor': 'flux_fill_mask',
        'description': 'Inpainting/outpainting with Flux',
        'model_type': 'flux_fill',
        'requires_mask': True
    },
    'flux-redux': {
        'name': 'Flux Redux',
        'model_id': 'black-forest-labs/FLUX.1-Redux-dev',
        'preprocessor': 'flux_redux',
        'description': 'Image variations and style transfer',
        'model_type': 'flux_redux',
        'requires_reference': True
    },
    'flux-depth': {
        'name': 'Flux Depth',
        'model_id': 'black-forest-labs/FLUX.1-Depth-dev',
        'preprocessor': 'depth',
        'description': 'Depth-controlled generation with Flux',
        'model_type': 'flux_control'
    },
    'flux-canny': {
        'name': 'Flux Canny',
        'model_id': 'black-forest-labs/FLUX.1-Canny-dev',
        'preprocessor': 'canny',
        'description': 'Edge-controlled generation with Flux',
        'model_type': 'flux_control'
    }
}

class ControlNetComponent(VisualComponent):
    """ControlNet Component for controlled image generation"""
    
    component_type = "controlnet"
    display_name = "ControlNet"
    category = "Control"
    icon = "🎮"
    
    # Define input ports
    input_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.INPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.INPUT),
        create_port("control_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("source_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("mask_image", PortType.IMAGE, PortDirection.INPUT, optional=True),  # For Flux Fill
        create_port("reference_image", PortType.IMAGE, PortDirection.INPUT, optional=True),  # For Flux Redux
    ]
    
    # Define output ports  
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT),
        create_port("control_image", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("preprocessed_image", PortType.IMAGE, PortDirection.OUTPUT, optional=True),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        create_property_definition(
            key="control_type",
            display_name="Control Type",
            prop_type=PropertyType.CHOICE,
            default="canny",
            category="Model",
            metadata={
                "choices": list(CONTROLNET_CONFIGS.keys()) + list(CONTROLNET_SDXL_CONFIGS.keys()) + list(FLUX_CONTROL_CONFIGS.keys()),
                "descriptions": {**{k: v['description'] for k, v in CONTROLNET_CONFIGS.items()},
                               **{k: v['description'] for k, v in CONTROLNET_SDXL_CONFIGS.items()},
                               **{k: v['description'] for k, v in FLUX_CONTROL_CONFIGS.items()}}
            }
        ),
        create_property_definition(
            key="model_path",
            display_name="Model Path",
            prop_type=PropertyType.FILE_PATH,
            default="",
            category="Model",
            metadata={
                "filter": "*.safetensors;*.bin",
                "description": "Path to local ControlNet model (optional)"
            }
        ),
        create_property_definition(
            key="use_preprocessor",
            display_name="Use Preprocessor",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="Preprocessing",
            metadata={
                "description": "Apply preprocessing to source image"
            }
        ),
        
        # Control Settings
        create_property_definition(
            key="control_strength",
            display_name="Control Strength",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Control",
            metadata={"min": 0.0, "max": 2.0, "step": 0.05}
        ),
        create_property_definition(
            key="guidance_start",
            display_name="Guidance Start",
            prop_type=PropertyType.FLOAT,
            default=0.0,
            category="Control",
            metadata={
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.05,
                "description": "When control starts (0=beginning)"
            }
        ),
        create_property_definition(
            key="guidance_end",
            display_name="Guidance End",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Control",
            metadata={
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.05,
                "description": "When control ends (1=end)"
            }
        ),
        create_property_definition(
            key="control_mode",
            display_name="Control Mode",
            prop_type=PropertyType.CHOICE,
            default="balanced",
            category="Control",
            metadata={
                "choices": ["balanced", "prompt", "control"],
                "descriptions": {
                    "balanced": "Balance between prompt and control",
                    "prompt": "Prompt is more important",
                    "control": "Control is more important"
                }
            }
        ),
        
        # Canny Settings
        create_property_definition(
            key="canny_low_threshold",
            display_name="Canny Low Threshold",
            prop_type=PropertyType.INTEGER,
            default=100,
            category="Canny Settings",
            metadata={
                "min": 1, 
                "max": 255,
                "depends_on": {"control_type": "canny"}
            }
        ),
        create_property_definition(
            key="canny_high_threshold",
            display_name="Canny High Threshold",
            prop_type=PropertyType.INTEGER,
            default=200,
            category="Canny Settings",
            metadata={
                "min": 1, 
                "max": 255,
                "depends_on": {"control_type": "canny"}
            }
        ),
        
        # Depth Settings
        create_property_definition(
            key="depth_estimator",
            display_name="Depth Estimator",
            prop_type=PropertyType.CHOICE,
            default="DPT",
            category="Depth Settings",
            metadata={
                "choices": ["DPT", "Midas", "Zoe"],
                "depends_on": {"control_type": "depth"}
            }
        ),
        
        # OpenPose Settings
        create_property_definition(
            key="detect_body",
            display_name="Detect Body",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="OpenPose Settings",
            metadata={"depends_on": {"control_type": "openpose"}}
        ),
        create_property_definition(
            key="detect_hand",
            display_name="Detect Hands",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="OpenPose Settings",
            metadata={"depends_on": {"control_type": "openpose"}}
        ),
        create_property_definition(
            key="detect_face",
            display_name="Detect Face",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="OpenPose Settings",
            metadata={"depends_on": {"control_type": "openpose"}}
        ),
        
        # Processing Settings
        create_property_definition(
            key="resize_mode",
            display_name="Resize Mode",
            prop_type=PropertyType.CHOICE,
            default="resize",
            category="Processing",
            metadata={
                "choices": ["resize", "crop", "fill"],
                "descriptions": {
                    "resize": "Resize to fit",
                    "crop": "Crop to fit",
                    "fill": "Fill with padding"
                }
            }
        ),
        create_property_definition(
            key="output_preprocessed",
            display_name="Output Preprocessed",
            prop_type=PropertyType.BOOLEAN,
            default=True,
            category="Processing",
            metadata={
                "description": "Output the preprocessed control image"
            }
        ),
        
        # Multi-ControlNet Settings
        create_property_definition(
            key="enable_multi_controlnet",
            display_name="Enable Multi-ControlNet",
            prop_type=PropertyType.BOOLEAN,
            default=False,
            category="Advanced",
            metadata={
                "description": "Use multiple ControlNets (requires multiple control images)"
            }
        ),
        create_property_definition(
            key="controlnet_conditioning_scale",
            display_name="Conditioning Scale",
            prop_type=PropertyType.FLOAT,
            default=1.0,
            category="Advanced",
            metadata={
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
                "description": "Overall conditioning scale"
            }
        ),
        
        # Flux Fill Settings
        create_property_definition(
            key="fill_mode",
            display_name="Fill Mode",
            prop_type=PropertyType.CHOICE,
            default="inpaint",
            category="Flux Fill",
            metadata={
                "choices": ["inpaint", "outpaint", "extend"],
                "descriptions": {
                    "inpaint": "Fill masked areas",
                    "outpaint": "Extend beyond image bounds",
                    "extend": "Smart extension of image"
                },
                "depends_on": {"control_type": "flux-fill"}
            }
        ),
        create_property_definition(
            key="mask_blur",
            display_name="Mask Blur",
            prop_type=PropertyType.INTEGER,
            default=4,
            category="Flux Fill",
            metadata={
                "min": 0,
                "max": 64,
                "description": "Blur mask edges for smoother transitions",
                "depends_on": {"control_type": "flux-fill"}
            }
        ),
        create_property_definition(
            key="preserve_masked_area",
            display_name="Preserve Masked Area",
            prop_type=PropertyType.BOOLEAN,
            default=False,
            category="Flux Fill",
            metadata={
                "description": "Keep original content in masked areas",
                "depends_on": {"control_type": "flux-fill"}
            }
        ),
        
        # Flux Redux Settings
        create_property_definition(
            key="redux_mode",
            display_name="Redux Mode",
            prop_type=PropertyType.CHOICE,
            default="variation",
            category="Flux Redux",
            metadata={
                "choices": ["variation", "style_transfer", "remix"],
                "descriptions": {
                    "variation": "Create variations of reference image",
                    "style_transfer": "Apply reference style to new content",
                    "remix": "Creative reinterpretation"
                },
                "depends_on": {"control_type": "flux-redux"}
            }
        ),
        create_property_definition(
            key="redux_strength",
            display_name="Redux Strength",
            prop_type=PropertyType.FLOAT,
            default=0.75,
            category="Flux Redux",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "How much to vary from reference",
                "depends_on": {"control_type": "flux-redux"}
            }
        ),
        create_property_definition(
            key="redux_creativity",
            display_name="Creativity Level",
            prop_type=PropertyType.FLOAT,
            default=0.5,
            category="Flux Redux",
            metadata={
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Balance between fidelity and creativity",
                "depends_on": {"control_type": "flux-redux"}
            }
        ),
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize visual component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # ControlNet-specific attributes
        self.controlnet = None
        self.preprocessor = None
        self.control_type_info = None
        
        # Visual settings for backward compatibility
        self.colors = {
            "ControlNet": "#9b59b6",  # Purple for control
        }
        self.icons = {
            "ControlNet": "🎮",
        }
        
        if self.scene:
            self.draw()
    
    def load_controlnet(self, control_type: str, model_path: Optional[str] = None):
        """Load ControlNet model"""
        if not DIFFUSERS_AVAILABLE:
            print("Diffusers not available, using mock ControlNet")
            self.controlnet = self._create_mock_controlnet()
            return True
            
        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
            
            # Get config
            if control_type in CONTROLNET_CONFIGS:
                config = CONTROLNET_CONFIGS[control_type]
            elif control_type in CONTROLNET_SDXL_CONFIGS:
                config = CONTROLNET_SDXL_CONFIGS[control_type]
            elif control_type in FLUX_CONTROL_CONFIGS:
                config = FLUX_CONTROL_CONFIGS[control_type]
                # Handle Flux-specific loading
                return self._load_flux_control(config, model_path)
            else:
                raise ValueError(f"Unknown control type: {control_type}")
            
            self.control_type_info = config
            
            # Load model
            if model_path and Path(model_path).exists():
                print(f"Loading ControlNet from local path: {model_path}")
                self.controlnet = ControlNetModel.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                )
            else:
                # Try local paths first
                local_paths = [
                    f"/home/alex/models/ControlNet/{config['model_id'].split('/')[-1]}",
                    f"/home/alex/SwarmUI/Models/controlnet/{config['model_id'].split('/')[-1]}",
                ]
                
                loaded = False
                for local_path in local_paths:
                    if Path(local_path).exists():
                        print(f"Found local ControlNet at: {local_path}")
                        self.controlnet = ControlNetModel.from_pretrained(
                            local_path,
                            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                        )
                        loaded = True
                        break
                
                if not loaded:
                    print(f"Would download ControlNet from: {config['model_id']}")
                    # In production, would download from HuggingFace
                    self.controlnet = self._create_mock_controlnet()
            
            return True
            
        except Exception as e:
            print(f"Error loading ControlNet: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_flux_control(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load Flux-specific control models (Fill, Redux, etc.)"""
        model_type = config.get('model_type', 'flux_control')
        
        if model_type == 'flux_fill':
            # Flux Fill for inpainting/outpainting
            self.controlnet = self._create_flux_fill_model(config, model_path)
        elif model_type == 'flux_redux':
            # Flux Redux for variations/style transfer
            self.controlnet = self._create_flux_redux_model(config, model_path)
        else:
            # Standard Flux control (depth, canny, etc.)
            self.controlnet = self._create_flux_control_model(config, model_path)
        
        self.control_type_info = config
        return True
    
    def _create_flux_fill_model(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Create Flux Fill model for inpainting"""
        class FluxFillModel:
            def __init__(self, config):
                self.config = config
                self.model_type = 'flux_fill'
            
            def to(self, device):
                return self
            
            def __call__(self, image, mask, prompt, **kwargs):
                # Flux Fill specific processing
                fill_mode = kwargs.get('fill_mode', 'inpaint')
                return {
                    'filled_image': image,
                    'mode': fill_mode,
                    'mask': mask
                }
        
        return FluxFillModel(config)
    
    def _create_flux_redux_model(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Create Flux Redux model for variations"""
        class FluxReduxModel:
            def __init__(self, config):
                self.config = config
                self.model_type = 'flux_redux'
            
            def to(self, device):
                return self
            
            def __call__(self, reference_image, prompt, **kwargs):
                # Flux Redux specific processing
                redux_mode = kwargs.get('redux_mode', 'variation')
                strength = kwargs.get('strength', 0.75)
                
                return {
                    'variation': reference_image,
                    'mode': redux_mode,
                    'strength': strength
                }
        
        return FluxReduxModel(config)
    
    def _create_flux_control_model(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Create standard Flux control model"""
        class FluxControlModel:
            def __init__(self, config):
                self.config = config
                self.model_type = 'flux_control'
            
            def to(self, device):
                return self
            
            def __call__(self, control_image, prompt, **kwargs):
                return {
                    'control': control_image,
                    'type': config.get('preprocessor', 'unknown')
                }
        
        return FluxControlModel(config)
    
    def load_preprocessor(self, preprocessor_type: str):
        """Load the appropriate preprocessor"""
        if preprocessor_type == 'canny':
            self.preprocessor = self._canny_preprocessor
        elif preprocessor_type == 'midas' or preprocessor_type == 'depth':
            self.preprocessor = self._depth_preprocessor
        elif preprocessor_type == 'openpose':
            self.preprocessor = self._openpose_preprocessor
        elif preprocessor_type == 'mlsd':
            self.preprocessor = self._mlsd_preprocessor
        elif preprocessor_type == 'scribble':
            self.preprocessor = self._scribble_preprocessor
        elif preprocessor_type == 'normal_bae':
            self.preprocessor = self._normal_preprocessor
        elif preprocessor_type == 'uniformer' or preprocessor_type == 'seg':
            self.preprocessor = self._segmentation_preprocessor
        elif preprocessor_type == 'shuffle':
            self.preprocessor = self._shuffle_preprocessor
        elif preprocessor_type == 'tile_resample':
            self.preprocessor = self._tile_preprocessor
        elif preprocessor_type == 'flux_fill_mask':
            self.preprocessor = self._flux_fill_preprocessor
        elif preprocessor_type == 'flux_redux':
            self.preprocessor = self._flux_redux_preprocessor
        else:
            self.preprocessor = None
    
    def _canny_preprocessor(self, image: Image.Image) -> Image.Image:
        """Canny edge detection preprocessor"""
        import cv2
        
        low = self.properties.get("canny_low_threshold", 100)
        high = self.properties.get("canny_high_threshold", 200)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low, high)
        
        # Convert edges to 3-channel image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def _depth_preprocessor(self, image: Image.Image) -> Image.Image:
        """Depth estimation preprocessor"""
        try:
            from transformers import pipeline
            
            depth_estimator = pipeline('depth-estimation')
            depth = depth_estimator(image)['depth']
            
            # Normalize depth map
            depth_array = np.array(depth)
            depth_norm = ((depth_array - depth_array.min()) / 
                         (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
            
            # Convert to 3-channel
            depth_rgb = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(depth_rgb)
        except:
            # Fallback: simple gradient depth
            width, height = image.size
            depth = Image.new('RGB', (width, height))
            pixels = depth.load()
            
            for y in range(height):
                gray_value = int(255 * y / height)
                for x in range(width):
                    pixels[x, y] = (gray_value, gray_value, gray_value)
            
            return depth
    
    def _openpose_preprocessor(self, image: Image.Image) -> Image.Image:
        """OpenPose preprocessor"""
        try:
            from controlnet_aux import OpenposeDetector
            
            openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            result = openpose(
                image,
                detect_hand=self.properties.get("detect_hand", True),
                detect_face=self.properties.get("detect_face", True),
                detect_body=self.properties.get("detect_body", True)
            )
            return result
        except:
            # Fallback: return black image
            return Image.new('RGB', image.size, color='black')
    
    def _mlsd_preprocessor(self, image: Image.Image) -> Image.Image:
        """M-LSD line detection preprocessor"""
        try:
            from controlnet_aux import MLSDdetector
            
            mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
            result = mlsd(image)
            return result
        except:
            # Fallback: edge detection
            return self._canny_preprocessor(image)
    
    def _scribble_preprocessor(self, image: Image.Image) -> Image.Image:
        """Scribble preprocessor"""
        try:
            from controlnet_aux import PidiNetDetector
            
            pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            result = pidinet(image, safe=True)
            return result
        except:
            # Fallback: inverted canny
            canny = self._canny_preprocessor(image)
            inverted = Image.eval(canny, lambda x: 255 - x)
            return inverted
    
    def _normal_preprocessor(self, image: Image.Image) -> Image.Image:
        """Normal map preprocessor"""
        try:
            from transformers import pipeline
            
            depth_estimator = pipeline('depth-estimation')
            depth = depth_estimator(image)['depth']
            
            # Convert depth to normal map (simplified)
            depth_array = np.array(depth)
            h, w = depth_array.shape
            
            # Calculate gradients
            dx = cv2.Sobel(depth_array, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(depth_array, cv2.CV_32F, 0, 1, ksize=3)
            
            # Create normal map
            normal = np.zeros((h, w, 3), dtype=np.float32)
            normal[:, :, 0] = -dx
            normal[:, :, 1] = -dy
            normal[:, :, 2] = 1.0
            
            # Normalize
            norm = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
            normal[:, :, 0] /= norm
            normal[:, :, 1] /= norm
            normal[:, :, 2] /= norm
            
            # Convert to RGB
            normal_rgb = ((normal + 1.0) * 127.5).astype(np.uint8)
            
            return Image.fromarray(normal_rgb)
        except:
            # Fallback: blue normal map
            return Image.new('RGB', image.size, color=(128, 128, 255))
    
    def _segmentation_preprocessor(self, image: Image.Image) -> Image.Image:
        """Semantic segmentation preprocessor"""
        try:
            from transformers import pipeline
            
            segmenter = pipeline("image-segmentation")
            segments = segmenter(image)
            
            # Create segmentation map
            seg_map = Image.new('RGB', image.size, color='black')
            
            # Simple color mapping for segments
            for i, segment in enumerate(segments):
                color = (
                    (i * 37) % 255,
                    (i * 67) % 255,
                    (i * 97) % 255
                )
                # Would apply segment mask with color
                
            return seg_map
        except:
            # Fallback: color quantization
            quantized = image.quantize(colors=16)
            return quantized.convert('RGB')
    
    def _shuffle_preprocessor(self, image: Image.Image) -> Image.Image:
        """Shuffle preprocessor"""
        # Shuffle blocks of the image
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        block_size = 64
        
        # Create shuffled version
        shuffled = img_array.copy()
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                # Random swap with another block
                y2 = np.random.randint(0, (h - block_size) // block_size) * block_size
                x2 = np.random.randint(0, (w - block_size) // block_size) * block_size
                
                # Swap blocks
                temp = shuffled[y:y+block_size, x:x+block_size].copy()
                shuffled[y:y+block_size, x:x+block_size] = shuffled[y2:y2+block_size, x2:x2+block_size]
                shuffled[y2:y2+block_size, x2:x2+block_size] = temp
        
        return Image.fromarray(shuffled)
    
    def _tile_preprocessor(self, image: Image.Image) -> Image.Image:
        """Tile preprocessor - just returns the image"""
        return image
    
    def _flux_fill_preprocessor(self, image: Image.Image, mask: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Flux Fill preprocessor - prepares image and mask for inpainting"""
        if mask is None:
            # Create default mask if not provided
            mask = Image.new('L', image.size, 255)  # White = inpaint area
        
        # Apply mask blur if specified
        mask_blur = self.properties.get("mask_blur", 4)
        if mask_blur > 0:
            import cv2
            mask_array = np.array(mask)
            mask_array = cv2.GaussianBlur(mask_array, (mask_blur * 2 + 1, mask_blur * 2 + 1), 0)
            mask = Image.fromarray(mask_array)
        
        # Handle different fill modes
        fill_mode = self.properties.get("fill_mode", "inpaint")
        if fill_mode == "outpaint":
            # Extend canvas for outpainting
            extend_pixels = 256  # Can be made configurable
            new_width = image.width + extend_pixels * 2
            new_height = image.height + extend_pixels * 2
            
            # Create extended image with neutral background
            extended_img = Image.new('RGB', (new_width, new_height), (128, 128, 128))
            extended_img.paste(image, (extend_pixels, extend_pixels))
            
            # Create corresponding mask
            extended_mask = Image.new('L', (new_width, new_height), 255)
            extended_mask.paste(Image.new('L', image.size, 0), (extend_pixels, extend_pixels))
            
            image = extended_img
            mask = extended_mask
        
        return {
            'image': image,
            'mask': mask,
            'mode': fill_mode
        }
    
    def _flux_redux_preprocessor(self, reference_image: Image.Image) -> Dict[str, Any]:
        """Flux Redux preprocessor - prepares reference image for variation/style transfer"""
        redux_mode = self.properties.get("redux_mode", "variation")
        
        # Analyze reference image for key features
        # In a real implementation, this might extract style features, color palette, etc.
        
        # For now, we'll prepare the reference image and metadata
        width, height = reference_image.size
        
        # Extract dominant colors for style transfer
        dominant_colors = self._extract_dominant_colors(reference_image, n_colors=5)
        
        # Prepare Redux configuration
        redux_config = {
            'reference_image': reference_image,
            'mode': redux_mode,
            'strength': self.properties.get("redux_strength", 0.75),
            'creativity': self.properties.get("redux_creativity", 0.5),
            'dominant_colors': dominant_colors,
            'original_size': (width, height)
        }
        
        return redux_config
    
    def _extract_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image for style reference"""
        # Resize for faster processing
        thumb = image.copy()
        thumb.thumbnail((150, 150))
        
        # Convert to RGB if needed
        if thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
        
        # Get colors using quantization
        quantized = thumb.quantize(colors=n_colors)
        palette = quantized.getpalette()
        
        # Extract RGB values
        colors = []
        for i in range(n_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            colors.append((r, g, b))
        
        return colors
    
    def _create_mock_controlnet(self):
        """Create mock ControlNet for testing"""
        class MockControlNet:
            def __init__(self):
                self.config = {"in_channels": 4}
            
            def to(self, device):
                return self
            
            def __call__(self, *args, **kwargs):
                return None
        
        return MockControlNet()
    
    async def process(self, context) -> bool:
        """Process ControlNet conditioning"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get inputs
            pipeline = self.get_input_data("pipeline")
            conditioning = self.get_input_data("conditioning") or {}
            control_image = self.get_input_data("control_image")
            source_image = self.get_input_data("source_image")
            mask_image = self.get_input_data("mask_image")  # For Flux Fill
            reference_image = self.get_input_data("reference_image")  # For Flux Redux
            
            if not pipeline:
                raise ValueError("No pipeline provided")
            
            # Get control type
            control_type = self.properties.get("control_type", "canny")
            
            # Load ControlNet if not loaded
            if self.controlnet is None:
                if not self.load_controlnet(control_type):
                    raise ValueError("Failed to load ControlNet")
            
            # Handle Flux-specific control types
            if control_type == 'flux-fill':
                return await self._process_flux_fill(pipeline, conditioning, source_image, mask_image, context)
            elif control_type == 'flux-redux':
                return await self._process_flux_redux(pipeline, conditioning, reference_image, context)
            
            # Standard ControlNet processing
            # Determine control image source
            if control_image is None and source_image is None:
                raise ValueError("Either control_image or source_image must be provided")
            
            # Process control image
            if control_image is None:
                # Need to preprocess source image
                if self.properties.get("use_preprocessor", True):
                    # Load preprocessor
                    preprocessor_type = self.control_type_info.get('preprocessor', 'canny')
                    self.load_preprocessor(preprocessor_type)
                    
                    if self.preprocessor:
                        print(f"Preprocessing image with {preprocessor_type}")
                        control_image = self.preprocessor(source_image)
                    else:
                        control_image = source_image
                else:
                    control_image = source_image
            
            # Resize control image to match generation size
            if hasattr(pipeline, 'unet'):
                # Get target size from pipeline config or properties
                width = conditioning.get('width', 512)
                height = conditioning.get('height', 512)
                
                if control_image.size != (width, height):
                    resize_mode = self.properties.get("resize_mode", "resize")
                    if resize_mode == "resize":
                        control_image = control_image.resize((width, height), Image.Resampling.LANCZOS)
                    elif resize_mode == "crop":
                        # Center crop
                        control_image = control_image.resize(
                            (max(width, int(control_image.width * height / control_image.height)),
                             max(height, int(control_image.height * width / control_image.width))),
                            Image.Resampling.LANCZOS
                        )
                        left = (control_image.width - width) // 2
                        top = (control_image.height - height) // 2
                        control_image = control_image.crop((left, top, left + width, top + height))
                    elif resize_mode == "fill":
                        # Fill with black padding
                        new_img = Image.new('RGB', (width, height), color='black')
                        paste_x = (width - control_image.width) // 2
                        paste_y = (height - control_image.height) // 2
                        new_img.paste(control_image, (paste_x, paste_y))
                        control_image = new_img
            
            # Apply ControlNet to pipeline
            if DIFFUSERS_AVAILABLE:
                from diffusers import StableDiffusionControlNetPipeline
                
                # Create ControlNet pipeline if needed
                if not isinstance(pipeline, StableDiffusionControlNetPipeline):
                    # Wrap existing pipeline with ControlNet
                    controlnet_pipeline = StableDiffusionControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        tokenizer=pipeline.tokenizer,
                        unet=pipeline.unet,
                        controlnet=self.controlnet,
                        scheduler=pipeline.scheduler,
                        safety_checker=None,
                        requires_safety_checker=False,
                        feature_extractor=None
                    )
                    pipeline = controlnet_pipeline
                else:
                    # Update ControlNet in existing pipeline
                    pipeline.controlnet = self.controlnet
            
            # Update conditioning with ControlNet parameters
            conditioning.update({
                'control_image': control_image,
                'controlnet_conditioning_scale': self.properties.get("control_strength", 1.0),
                'control_guidance_start': self.properties.get("guidance_start", 0.0),
                'control_guidance_end': self.properties.get("guidance_end", 1.0),
                'control_mode': self.properties.get("control_mode", "balanced"),
            })
            
            # Set outputs
            self.set_output_data("pipeline", pipeline)
            self.set_output_data("conditioning", conditioning)
            self.set_output_data("control_image", control_image)
            
            if self.properties.get("output_preprocessed", True) and source_image:
                self.set_output_data("preprocessed_image", control_image)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in ControlNetComponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _process_flux_fill(self, pipeline, conditioning, source_image, mask_image, context):
        """Process Flux Fill for inpainting/outpainting"""
        if not source_image:
            raise ValueError("Source image required for Flux Fill")
        
        # Preprocess with Flux Fill preprocessor
        fill_data = self._flux_fill_preprocessor(source_image, mask_image)
        
        # Update conditioning for Flux Fill
        conditioning.update({
            'image': fill_data['image'],
            'mask': fill_data['mask'],
            'fill_mode': fill_data['mode'],
            'preserve_masked': self.properties.get("preserve_masked_area", False),
            'controlnet_conditioning_scale': self.properties.get("control_strength", 1.0),
        })
        
        # Flux Fill uses a special pipeline configuration
        # In a real implementation, this would set up the Flux Fill model
        
        # Set outputs
        self.set_output_data("pipeline", pipeline)
        self.set_output_data("conditioning", conditioning)
        self.set_output_data("control_image", fill_data['image'])
        self.set_output_data("preprocessed_image", fill_data['mask'])
        
        self.set_status(ComponentStatus.COMPLETE)
        return True
    
    async def _process_flux_redux(self, pipeline, conditioning, reference_image, context):
        """Process Flux Redux for variations/style transfer"""
        if not reference_image:
            raise ValueError("Reference image required for Flux Redux")
        
        # Preprocess with Flux Redux preprocessor
        redux_data = self._flux_redux_preprocessor(reference_image)
        
        # Update conditioning for Flux Redux
        conditioning.update({
            'reference_image': redux_data['reference_image'],
            'redux_mode': redux_data['mode'],
            'redux_strength': redux_data['strength'],
            'redux_creativity': redux_data['creativity'],
            'dominant_colors': redux_data['dominant_colors'],
            'controlnet_conditioning_scale': self.properties.get("control_strength", 1.0),
        })
        
        # Flux Redux uses the reference image to guide generation
        # In a real implementation, this would set up the Flux Redux model
        
        # Set outputs
        self.set_output_data("pipeline", pipeline)
        self.set_output_data("conditioning", conditioning)
        self.set_output_data("control_image", redux_data['reference_image'])
        
        self.set_status(ComponentStatus.COMPLETE)
        return True
    
    def unload_model(self):
        """Unload ControlNet model to free memory"""
        if self.controlnet:
            del self.controlnet
            self.controlnet = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("ControlNet model unloaded")
