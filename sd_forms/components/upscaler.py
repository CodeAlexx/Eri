"""
Upscaler Component for SD Forms
Supports multiple upscaling models including ESRGAN, Real-ESRGAN, SwinIR, and more
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)
from ..utils.constants import DEVICE

# Upscaler model configurations
UPSCALER_CONFIGS = {
    'realesrgan-x4plus': {
        'name': 'Real-ESRGAN x4+',
        'scale': 4,
        'model_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'model_type': 'realesrgan',
        'channels': 3,
        'description': 'Best quality general purpose upscaler'
    },
    'realesrgan-x4plus-anime': {
        'name': 'Real-ESRGAN x4+ Anime',
        'scale': 4,
        'model_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'model_type': 'realesrgan',
        'channels': 3,
        'description': 'Optimized for anime/artwork'
    },
    'esrgan-x4': {
        'name': 'ESRGAN x4',
        'scale': 4,
        'model_type': 'esrgan',
        'channels': 3,
        'description': 'Original ESRGAN model'
    },
    'swinir-x4': {
        'name': 'SwinIR x4',
        'scale': 4,
        'model_type': 'swinir',
        'channels': 3,
        'description': 'Transformer-based upscaler'
    },
    'ldsr-x4': {
        'name': 'LDSR x4',
        'scale': 4,
        'model_type': 'ldsr',
        'channels': 3,
        'description': 'Latent diffusion super resolution'
    }
}

class UpscalerComponent(VisualComponent):
    """Upscaler Component for enhancing image resolution"""
    
    component_type = "upscaler"
    display_name = "Upscaler"
    category = "Enhance"
    icon = "🔍"
    
    # Define input ports
    input_ports = [
        create_port("images", PortType.IMAGE, PortDirection.INPUT),
        create_port("scale_override", PortType.ANY, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("metadata", PortType.ANY, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        create_property_definition("model_type", "Model Type", PropertyType.CHOICE, "preset", "Model",
                                 metadata={"choices": ["preset", "local", "auto"]}),
        create_property_definition("preset_model", "Preset Model", PropertyType.CHOICE, "realesrgan-x4plus", "Model",
                                 metadata={"choices": list(UPSCALER_CONFIGS.keys())}),
        create_property_definition("model_path", "Model Path", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "file_picker", "filter": "*.pth;*.pt;*.safetensors;*.ckpt",
                                          "description": "Path to upscaler model file"}),
        
        # Upscaling Settings
        create_property_definition("scale_factor", "Scale Factor", PropertyType.INTEGER, 4, "Upscaling",
                                 metadata={"min": 2, "max": 8, "step": 1}),
        create_property_definition("tile_size", "Tile Size", PropertyType.INTEGER, 512, "Upscaling",
                                 metadata={"min": 128, "max": 1024, "step": 64, 
                                          "description": "Process in tiles to save memory"}),
        create_property_definition("tile_overlap", "Tile Overlap", PropertyType.INTEGER, 32, "Upscaling",
                                 metadata={"min": 0, "max": 128, "step": 8,
                                          "description": "Overlap between tiles to reduce seams"}),
        create_property_definition("face_enhance", "Face Enhancement", PropertyType.BOOLEAN, False, "Enhancement",
                                 metadata={"description": "Apply additional face enhancement"}),
        
        # Processing Settings
        create_property_definition("batch_size", "Batch Size", PropertyType.INTEGER, 1, "Processing",
                                 metadata={"min": 1, "max": 8, "description": "Images to process simultaneously"}),
        create_property_definition("precision", "Precision", PropertyType.CHOICE, "fp16", "Processing",
                                 metadata={"choices": ["fp32", "fp16", "bf16"], 
                                          "description": "Model precision (fp16 saves memory)"}),
        create_property_definition("denoise_strength", "Denoise Strength", PropertyType.FLOAT, 0.0, "Enhancement",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 1.0, "step": 0.05,
                                          "description": "0 = no denoising, 1 = maximum"}),
        
        # Output Settings
        create_property_definition("output_format", "Output Format", PropertyType.CHOICE, "same", "Output",
                                 metadata={"choices": ["same", "png", "jpeg", "webp"], 
                                          "description": "Output image format"}),
        create_property_definition("preserve_metadata", "Preserve Metadata", PropertyType.BOOLEAN, True, "Output"),
    ]
    
    def __init__(self, id: str = None):
        super().__init__(id, position=(0, 0))
        self.model = None
        self.model_info = None
        self.face_enhancer = None
    
    def load_model(self, model_config: Dict[str, Any], model_path: Optional[str] = None):
        """Load upscaler model"""
        try:
            model_type = model_config.get('model_type', 'realesrgan')
            
            if model_type == 'realesrgan':
                self._load_realesrgan(model_config, model_path)
            elif model_type == 'esrgan':
                self._load_esrgan(model_config, model_path)
            elif model_type == 'swinir':
                self._load_swinir(model_config, model_path)
            elif model_type == 'ldsr':
                self._load_ldsr(model_config, model_path)
            else:
                raise ValueError(f"Unknown upscaler type: {model_type}")
                
            self.model_info = model_config
            print(f"Loaded {model_config['name']} upscaler")
            return True
            
        except Exception as e:
            print(f"Error loading upscaler model: {e}")
            return False
    
    def _load_realesrgan(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load Real-ESRGAN model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            print("Real-ESRGAN not installed, using fallback implementation")
            # Fallback implementation
            self.model = self._create_fallback_upscaler(config)
            return
        
        # Model architecture
        if 'anime' in config.get('name', '').lower():
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
        
        # Load weights
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=DEVICE)
        else:
            # Download from URL if needed
            model_url = config.get('model_url')
            if model_url:
                import requests
                from io import BytesIO
                
                print(f"Downloading model from {model_url}")
                response = requests.get(model_url)
                state_dict = torch.load(BytesIO(response.content), map_location=DEVICE)
            else:
                raise ValueError("No model path or URL provided")
        
        # Initialize upsampler
        self.model = RealESRGANer(
            scale=config.get('scale', 4),
            model_path=model_path,
            model=model,
            tile=self.properties.get('tile_size', 512),
            tile_pad=self.properties.get('tile_overlap', 32),
            pre_pad=0,
            half=self.properties.get('precision', 'fp16') == 'fp16' and DEVICE == 'cuda'
        )
        
        # Load state dict
        self.model.model.load_state_dict(state_dict['params_ema'] if 'params_ema' in state_dict else state_dict)
        self.model.model.eval()
        self.model.model = self.model.model.to(DEVICE)
    
    def _load_esrgan(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load ESRGAN model"""
        # Simplified ESRGAN loading
        import torch.nn as nn
        
        class ESRGAN(nn.Module):
            def __init__(self, scale=4):
                super().__init__()
                self.scale = scale
                # Simplified architecture
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.upscale = nn.PixelShuffle(scale)
                self.final = nn.Conv2d(64 // (scale * scale), 3, 3, padding=1)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.upscale(x)
                x = self.final(x)
                return x
        
        self.model = ESRGAN(config.get('scale', 4))
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
    
    def _load_swinir(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load SwinIR model"""
        try:
            from swinir.models.network_swinir import SwinIR
        except ImportError:
            print("SwinIR not installed, using fallback")
            self.model = self._create_fallback_upscaler(config)
            return
        
        # Create SwinIR model
        self.model = SwinIR(
            upscale=config.get('scale', 4),
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)['params'])
        
        self.model = self.model.to(DEVICE)
        self.model.eval()
    
    def _load_ldsr(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Load LDSR model (Latent Diffusion Super Resolution)"""
        # This would require the LDSR implementation
        print("LDSR support coming soon, using fallback")
        self.model = self._create_fallback_upscaler(config)
    
    def _create_fallback_upscaler(self, config: Dict[str, Any]):
        """Create a simple fallback upscaler using PIL"""
        class FallbackUpscaler:
            def __init__(self, scale):
                self.scale = scale
            
            def enhance(self, img_array, outscale=None):
                # Simple bicubic upscaling as fallback
                img = Image.fromarray(img_array.astype(np.uint8))
                scale = outscale or self.scale
                new_size = (int(img.width * scale), int(img.height * scale))
                upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
                return np.array(upscaled)
        
        return FallbackUpscaler(config.get('scale', 4))
    
    def _load_face_enhancer(self):
        """Load face enhancement model if enabled"""
        if not self.properties.get('face_enhance', False):
            return
            
        try:
            from gfpgan import GFPGANer
            
            self.face_enhancer = GFPGANer(
                model_path='GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("Face enhancer loaded")
        except Exception as e:
            print(f"Could not load face enhancer: {e}")
            self.face_enhancer = None
    
    async def process(self, context) -> bool:
        """Process images through upscaler"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get input images
            images = self.get_input_data("images")
            if not images:
                print("No input images provided")
                self.set_status(ComponentStatus.ERROR)
                return False
            
            # Get scale override if provided
            scale_override = self.get_input_data("scale_override")
            scale = scale_override or self.properties.get("scale_factor", 4)
            
            # Load model if not already loaded
            if self.model is None:
                if self.properties.get("model_type") == "preset":
                    preset_name = self.properties.get("preset_model", "realesrgan-x4plus")
                    model_config = UPSCALER_CONFIGS.get(preset_name)
                    if not self.load_model(model_config):
                        raise ValueError("Failed to load upscaler model")
                elif self.properties.get("model_type") == "local":
                    model_path = self.properties.get("model_path")
                    if not model_path:
                        raise ValueError("Local model path not specified")
                    # Auto-detect model type from path
                    model_config = self._detect_model_type(model_path)
                    if not self.load_model(model_config, model_path):
                        raise ValueError("Failed to load local model")
            
            # Load face enhancer if needed
            if self.properties.get("face_enhance", False) and self.face_enhancer is None:
                self._load_face_enhancer()
            
            # Process images
            upscaled_images = []
            total_images = len(images)
            
            for i, img in enumerate(images):
                print(f"Upscaling image {i+1}/{total_images}")
                
                # Convert PIL to numpy array
                img_array = np.array(img)
                
                # Apply denoising if requested
                if self.properties.get("denoise_strength", 0.0) > 0:
                    img_array = self._apply_denoising(img_array)
                
                # Upscale the image
                if hasattr(self.model, 'enhance'):
                    # Real-ESRGAN style
                    upscaled_array = self.model.enhance(img_array, outscale=scale)[0]
                elif hasattr(self.model, 'forward'):
                    # PyTorch model style
                    with torch.no_grad():
                        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
                        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                        upscaled_tensor = self.model(img_tensor)
                        upscaled_array = (upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    # Fallback
                    upscaled_array = self.model.enhance(img_array, outscale=scale)
                
                # Apply face enhancement if enabled
                if self.face_enhancer:
                    try:
                        _, _, upscaled_array = self.face_enhancer.enhance(
                            upscaled_array, 
                            has_aligned=False, 
                            only_center_face=False, 
                            paste_back=True
                        )
                    except Exception as e:
                        print(f"Face enhancement failed: {e}")
                
                # Convert back to PIL
                upscaled_img = Image.fromarray(upscaled_array.astype(np.uint8))
                
                # Preserve metadata if requested
                if self.properties.get("preserve_metadata", True) and hasattr(img, 'info'):
                    upscaled_img.info = img.info.copy()
                    upscaled_img.info['upscaler'] = self.model_info.get('name', 'unknown')
                    upscaled_img.info['upscale_factor'] = scale
                
                upscaled_images.append(upscaled_img)
                
                # Send progress update
                if hasattr(context, 'progress_callback'):
                    progress = int((i + 1) / total_images * 100)
                    await context.progress_callback(upscaled_img, progress, i + 1)
            
            # Set outputs
            self.set_output_data("images", upscaled_images)
            self.set_output_data("metadata", {
                "upscaler": self.model_info.get('name', 'unknown') if self.model_info else 'unknown',
                "scale_factor": scale,
                "original_sizes": [(img.width, img.height) for img in images],
                "upscaled_sizes": [(img.width, img.height) for img in upscaled_images],
                "face_enhanced": self.properties.get("face_enhance", False),
                "denoise_strength": self.properties.get("denoise_strength", 0.0)
            })
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in UpscalerComponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_model_type(self, model_path: str) -> Dict[str, Any]:
        """Auto-detect upscaler model type from file"""
        path_lower = model_path.lower()
        
        if 'realesrgan' in path_lower or 'real-esrgan' in path_lower:
            if 'anime' in path_lower:
                return UPSCALER_CONFIGS['realesrgan-x4plus-anime'].copy()
            return UPSCALER_CONFIGS['realesrgan-x4plus'].copy()
        elif 'esrgan' in path_lower:
            return UPSCALER_CONFIGS['esrgan-x4'].copy()
        elif 'swinir' in path_lower:
            return UPSCALER_CONFIGS['swinir-x4'].copy()
        elif 'ldsr' in path_lower:
            return UPSCALER_CONFIGS['ldsr-x4'].copy()
        else:
            # Default to ESRGAN
            return {
                'name': Path(model_path).stem,
                'scale': 4,
                'model_type': 'esrgan',
                'channels': 3
            }
    
    def _apply_denoising(self, img_array: np.ndarray) -> np.ndarray:
        """Apply denoising to image"""
        import cv2
        
        strength = self.properties.get("denoise_strength", 0.0)
        if strength <= 0:
            return img_array
            
        # Convert to float
        img_float = img_array.astype(np.float32) / 255.0
        
        # Apply bilateral filter for edge-preserving denoising
        d = int(5 + strength * 10)  # Diameter
        sigma_color = 25 + strength * 50
        sigma_space = 25 + strength * 50
        
        denoised = cv2.bilateralFilter(img_float, d, sigma_color, sigma_space)
        
        # Blend with original based on strength
        result = img_float * (1 - strength) + denoised * strength
        
        return (result * 255).astype(np.uint8)
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.face_enhancer:
            del self.face_enhancer  
            self.face_enhancer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Upscaler model unloaded")
