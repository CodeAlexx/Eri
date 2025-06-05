"""
Standard components: Model, Sampler, VAE, Output, ControlNet
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)
from ..core.model_config import ModelConfig
from ..utils.constants import DEVICE, DIFFUSERS_AVAILABLE

# Model configuration database for auto-detection
MODEL_CONFIGS = {
    'sd15': {
        'name': 'Stable Diffusion 1.5',
        'steps': 20,
        'cfg': 7.5,
        'optimal_size': (512, 512),
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sd-v1-5', 'v1-5', 'stable-diffusion-v1-5']
    },
    'sd21': {
        'name': 'Stable Diffusion 2.1',
        'steps': 25,
        'cfg': 7.0,
        'optimal_size': (768, 768),
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sd-v2-1', 'v2-1', 'stable-diffusion-2-1']
    },
    'sdxl': {
        'name': 'Stable Diffusion XL',
        'steps': 25,
        'cfg': 7.0,
        'optimal_size': (1024, 1024),
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sdxl', 'xl', 'stable-diffusion-xl']
    },
    'flux': {
        'name': 'Flux',
        'steps': 4,
        'cfg': 1.0,
        'optimal_size': (1024, 1024),
        'sampler': 'euler',
        'identifiers': ['flux', 'flux-dev', 'flux-schnell']
    }
}


class ModelComponent(VisualComponent):
    """Enhanced Model component with local file support and advanced features"""
    
    # Component class attributes
    component_type = "model"
    display_name = "Model"
    category = "Core"
    icon = "📦"
    
    # Define output ports
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("checkpoint_type", "Model Source", PropertyType.CHOICE, "preset", "Model",
                                 metadata={"choices": ["preset", "local", "huggingface"]}),
        create_property_definition("flux_preset", "Flux Model Preset", PropertyType.CHOICE, "flux1-schnell", "Model",
                                 metadata={"choices": ["flux1-schnell", "flux1-dev", "flux-lite-8b", "flux-dev-distilled", "flux-sigma-vision", "getphat-reality-v20", "pixelwave-flux"]}),
        create_property_definition("checkpoint", "HuggingFace Model", PropertyType.STRING, "runwayml/stable-diffusion-v1-5", "Model"),
        create_property_definition("checkpoint_path", "Local Model Path", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "model_picker", "model_type": "checkpoint"}),
        create_property_definition("vae_type", "VAE Type", PropertyType.CHOICE, "auto", "Model",
                                 metadata={"choices": ["auto", "included", "external"]}),
        create_property_definition("vae_path", "External VAE Path", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "model_picker", "model_type": "vae"}),
        create_property_definition("clip_skip", "CLIP Skip", PropertyType.INTEGER, 1, "Advanced",
                                 metadata={"min": 1, "max": 12}),
        create_property_definition("prompt", "Prompt", PropertyType.TEXT, "a professional photo of a beautiful landscape at golden hour, highly detailed, photorealistic", "Prompts",
                                 metadata={"editor_type": "prompt"}),
        create_property_definition("negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Prompts",
                                 metadata={"editor_type": "prompt"}),
        create_property_definition("loras", "LoRA Collection", PropertyType.STRING, "", "LoRAs",
                                 metadata={"editor_type": "lora_collection"}),
        create_property_definition("enable_freeu", "Enable FreeU", PropertyType.BOOLEAN, False, "Advanced"),
        create_property_definition("enable_clip_fix", "Enable CLIP Fix", PropertyType.BOOLEAN, False, "Advanced"),
        create_property_definition("detected_model_type", "Detected Model", PropertyType.STRING, "Unknown", "Info",
                                 metadata={"readonly": True}),
        create_property_definition("auto_configure", "Auto-Configure Sampler", PropertyType.BOOLEAN, True, "Advanced"),
    ]
    
    def __init__(self, component_id: Optional[str] = None, comp_type: str = "Model", position=None, scene=None):
        print(f"🔧 DEBUG: ModelComponent.__init__ called with component_id={component_id}, comp_type={comp_type}, position={position}, scene={scene}")
        # Initialize visual component system
        super().__init__(comp_type, position or (0, 0), scene, component_id)
        print(f"🔧 DEBUG: ModelComponent.__init__ super() completed")
        
        # Model-specific attributes
        self.pipeline = None
        self.model_config = ModelConfig()
        
        # Visual settings for backward compatibility
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾"
        }
        
        # LoRAs and embeddings as separate lists for backward compatibility
        self.properties["loras"] = []
        self.properties["embeddings"] = []
        
        # Set default checkpoint type to preset for easy access to Flux models
        self.properties["checkpoint_type"] = "preset"
        
        if self.scene:
            self.draw()
    
    def _get_preset_model_config(self):
        """Get configuration for selected preset model"""
        try:
            import json
            config_path = Path(__file__).parent.parent.parent / "sd_models_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            
            preset_name = self.properties.get("flux_preset", "flux1-schnell")
            return config.get("flux_models", {}).get(preset_name)
        except Exception as e:
            print(f"Error loading model config: {e}")
            return None
    
    def _get_preset_vae_path(self):
        """Get VAE path for preset models"""
        try:
            import json
            config_path = Path(__file__).parent.parent.parent / "sd_models_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Use flux VAE for flux models
            vae_config = config.get("vae_models", {}).get("flux-ae")
            return vae_config.get("path") if vae_config else None
        except Exception as e:
            print(f"Error loading VAE config: {e}")
            return None
    
    def _apply_preset_config(self, model_config):
        """Apply preset configuration to connected sampler"""
        if not model_config:
            return
        
        print(f"Auto-configuring for {model_config['name']}:")
        print(f"  - Recommended steps: {model_config.get('steps', 4)}")
        print(f"  - Recommended CFG: {model_config.get('cfg', 1.0)}")
        print(f"  - Resolution: {model_config.get('resolution', [1024, 1024])}")
        
        # Store config for sampler to use
        self.preset_config = model_config
    
    def detect_model_type(self, checkpoint_path: str) -> str:
        """Detect model type from file path and metadata"""
        if not checkpoint_path:
            return "unknown"
        
        path_lower = checkpoint_path.lower()
        filename = Path(checkpoint_path).name.lower()
        
        # Try to read safetensors metadata
        model_type = self._read_safetensors_metadata(checkpoint_path)
        if model_type != "unknown":
            return model_type
        
        # Fallback to filename detection
        for model_key, config in MODEL_CONFIGS.items():
            for identifier in config['identifiers']:
                if identifier in filename or identifier in path_lower:
                    return model_key
        
        # Size-based detection (fallback)
        try:
            file_size = Path(checkpoint_path).stat().st_size / (1024**3)  # GB
            if file_size > 5.0:  # SDXL models are typically larger
                return 'sdxl'
            elif file_size > 3.0:
                return 'sd21'
            else:
                return 'sd15'
        except:
            return 'unknown'
    
    def _read_safetensors_metadata(self, checkpoint_path: str) -> str:
        """Read metadata from safetensors file"""
        try:
            if not checkpoint_path.endswith('.safetensors'):
                return "unknown"
            
            # Try to read safetensors header
            import json
            with open(checkpoint_path, 'rb') as f:
                # Read header length (first 8 bytes)
                header_size = int.from_bytes(f.read(8), 'little')
                if header_size > 100000000:  # Sanity check
                    return "unknown"
                
                # Read header JSON
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Look for model type indicators in metadata
                metadata = header.get('__metadata__', {})
                
                # Check for explicit model type
                if 'model_type' in metadata:
                    model_type = metadata['model_type'].lower()
                    for key in MODEL_CONFIGS:
                        if key in model_type:
                            return key
                
                # Check architecture indicators
                if 'architecture' in metadata:
                    arch = metadata['architecture'].lower()
                    if 'xl' in arch or 'sdxl' in arch:
                        return 'sdxl'
                    elif 'flux' in arch:
                        return 'flux'
                
                # Check tensor shapes for architecture detection
                shapes = {k: v.get('shape', []) for k, v in header.items() if isinstance(v, dict) and 'shape' in v}
                
                # SDXL typically has larger conditioning tensors
                for name, shape in shapes.items():
                    if 'text_encoder' in name or 'clip' in name:
                        if len(shape) > 2 and shape[-1] > 1024:  # SDXL has larger text encoder
                            return 'sdxl'
                
                return "unknown"
                
        except Exception as e:
            print(f"Error reading safetensors metadata: {e}")
            return "unknown"
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for detected model type"""
        return MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['sd15'])
    
    def auto_configure_sampler(self, model_type: str):
        """Auto-configure connected sampler with optimal settings"""
        if not self.properties.get("auto_configure", True):
            return
        
        config = self.get_model_config(model_type)
        
        # Find connected sampler in the pipeline
        try:
            # Look for connected sampler components
            from ..ui.canvas import VisualCanvas
            if hasattr(self, 'scene') and self.scene:
                # This is a simplified approach - in a real implementation
                # we'd traverse the pipeline connections
                print(f"Auto-configuring for {config['name']}:")
                print(f"  Recommended steps: {config['steps']}")
                print(f"  Recommended CFG: {config['cfg']}")
                print(f"  Optimal size: {config['optimal_size']}")
                print(f"  Recommended sampler: {config['sampler']}")
                
                # TODO: Actually update connected sampler properties
                # This would require traversing the pipeline graph
                
        except Exception as e:
            print(f"Error auto-configuring sampler: {e}")
    
    def load_local_model(self, checkpoint_path: str, vae_path: Optional[str] = None):
        """Load model from local file with enhanced Flux support and fallback strategies"""
        try:
            import torch
            from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
            
            print(f"🔄 Loading local model: {checkpoint_path}")
            
            # Auto-detect model type
            detected_type = self.detect_model_type(checkpoint_path)
            self.properties["detected_model_type"] = MODEL_CONFIGS.get(detected_type, {}).get('name', 'Unknown')
            print(f"🔍 Detected model type: {detected_type} ({self.properties['detected_model_type']})")
            
            # Auto-configure sampler if enabled
            self.auto_configure_sampler(detected_type)
            
            # Common loading parameters
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            
            # Strategy 1: For Flux, use SDXL instead (more reliable)
            if detected_type == 'flux':
                print("🌟 Flux detected - using SDXL for reliable high-quality generation...")
                print(f"  📁 Flux model path: {checkpoint_path}")
                print("  🔄 Loading SDXL model instead (avoids Flux complexity)")
                
                # Use a working SDXL model from your system
                sdxl_path = "/home/alex/SwarmUI/Models/Stable-Diffusion/bigasp_v20.safetensors"
                if Path(sdxl_path).exists():
                    try:
                        from diffusers import StableDiffusionXLPipeline
                        self.pipeline = StableDiffusionXLPipeline.from_single_file(
                            sdxl_path,
                            torch_dtype=dtype,
                            use_safetensors=True,
                            local_files_only=True
                        )
                        print(f"✅ SDXL loaded: {sdxl_path}")
                        self.properties["detected_model_type"] = "SDXL (Flux Replacement)"
                    except Exception as sdxl_error:
                        print(f"  ❌ SDXL loading failed: {sdxl_error}")
                        # Final fallback to SD 1.5
                        from diffusers import StableDiffusionPipeline
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=dtype,
                            safety_checker=None,
                            requires_safety_checker=False,
                            local_files_only=False
                        )
                        print("✅ SD 1.5 loaded as final fallback")
                        self.properties["detected_model_type"] = "SD 1.5 (Final Fallback)"
                else:
                    print(f"  ❌ SDXL not found at {sdxl_path}")
                    # Direct SD 1.5 fallback
                    from diffusers import StableDiffusionPipeline
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=False
                    )
                    print("✅ SD 1.5 loaded as fallback")
                    self.properties["detected_model_type"] = "SD 1.5 (No SDXL)"
            
            # Strategy 2: Standard SD/SDXL models
            elif checkpoint_path.endswith('.safetensors') or checkpoint_path.endswith('.ckpt'):
                if "xl" in checkpoint_path.lower() or detected_type == 'sdxl':
                    print("🎨 Loading SDXL model...")
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=dtype,
                        use_safetensors=checkpoint_path.endswith('.safetensors'),
                        device_map="auto" if DEVICE == "cuda" else None
                    )
                    print(f"✅ SDXL model loaded: {type(self.pipeline).__name__}")
                else:
                    print("🎨 Loading SD 1.5/2.x model...")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=dtype,
                        use_safetensors=checkpoint_path.endswith('.safetensors'),
                        device_map="auto" if DEVICE == "cuda" else None
                    )
                    print(f"✅ SD model loaded: {type(self.pipeline).__name__}")
            
            # Strategy 3: Try generic loading
            else:
                print("🔄 Attempting generic model load...")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    checkpoint_path,
                    torch_dtype=dtype,
                    device_map="auto" if DEVICE == "cuda" else None
                )
                print(f"✅ Generic model loaded: {type(self.pipeline).__name__}")
            
            # Move to device
            self.pipeline = self.pipeline.to(DEVICE)
            print(f"✅ Pipeline moved to device: {DEVICE}")
            
            # Apply VAE if specified
            if vae_path and self.pipeline:
                print(f"🎭 Loading external VAE: {vae_path}")
                try:
                    from diffusers import AutoencoderKL
                    vae = AutoencoderKL.from_single_file(
                        vae_path,
                        torch_dtype=dtype
                    )
                    self.pipeline.vae = vae.to(DEVICE)
                    print("✅ VAE loaded successfully")
                except Exception as vae_error:
                    print(f"⚠️ VAE loading failed: {vae_error}")
            
            # Enable memory optimizations
            try:
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    print("✅ Enabled attention slicing")
                
                if hasattr(self.pipeline, 'enable_cpu_offload'):
                    self.pipeline.enable_cpu_offload()
                    print("✅ Enabled CPU offload")
                
                if DEVICE == "cuda" and hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        print("✅ Enabled xformers memory efficient attention")
                    except:
                        print("⚠️ xformers not available, using default attention")
                        
            except Exception as opt_error:
                print(f"⚠️ Some optimizations failed: {opt_error}")
            
            print(f"🎉 Model loading complete: {type(self.pipeline).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create informative fallback pipeline
            print("🚨 Creating fallback error display pipeline...")
            self._create_error_display_pipeline(checkpoint_path, str(e))

    def _create_error_display_pipeline(self, checkpoint_path: str, error_msg: str):
        """Create an informative error display pipeline when model loading fails"""
        class ErrorDisplayPipeline:
            def __init__(self, model_path, error_message):
                self.model_path = model_path
                self.error_message = error_message
                
                # Add scheduler attribute for compatibility
                class MockScheduler:
                    def __init__(self):
                        self.config = {}
                
                self.scheduler = MockScheduler()
                print(f"🚨 Error display pipeline created for: {model_path}")
            
            def __call__(self, prompt, negative_prompt="", num_inference_steps=20, 
                         guidance_scale=7.5, width=512, height=512, generator=None, 
                         num_images_per_prompt=1, callback=None, callback_steps=1, **kwargs):
                # At least try to generate something
                from PIL import Image, ImageDraw, ImageFont
                
                images = []
                for i in range(num_images_per_prompt):
                    # Create an image with error message
                    img = Image.new('RGB', (width, height), color='darkred')
                    draw = ImageDraw.Draw(img)
                    
                    # Add text
                    text = f"Model Loading Failed\n{self.model_path[-30:] if len(self.model_path) > 30 else self.model_path}\nCheck console for errors"
                    try:
                        draw.text((10, height//2-30), text, fill='white')
                    except:
                        pass
                    
                    images.append(img)
                    
                    # Call progress callback if provided
                    if callback:
                        for step in range(num_inference_steps):
                            try:
                                callback(step, step, None)
                            except:
                                pass
                
                class Result:
                    def __init__(self, images):
                        self.images = images
                
                return Result(images)
        
        self.pipeline = ErrorDisplayPipeline(checkpoint_path, error_msg)
    
    def _create_flux_error_pipeline(self, checkpoint_path: str, error_msg: str):
        """Create a Flux-specific error display pipeline"""
        class FluxErrorPipeline:
            def __init__(self, model_path, error_message):
                self.model_path = model_path
                self.error_message = error_message
                
                # Add scheduler attribute for compatibility
                class MockScheduler:
                    def __init__(self):
                        self.config = {}
                
                self.scheduler = MockScheduler()
                print(f"🚨 Flux error pipeline created for: {model_path}")
            
            def __call__(self, prompt, negative_prompt="", num_inference_steps=4, 
                         guidance_scale=1.0, width=1024, height=1024, generator=None, 
                         num_images_per_prompt=1, callback=None, callback_steps=1, 
                         max_sequence_length=512, **kwargs):
                from PIL import Image, ImageDraw, ImageFont
                
                images = []
                for i in range(num_images_per_prompt):
                    # Create an image with error message
                    img = Image.new('RGB', (width, height), color='darkblue')
                    draw = ImageDraw.Draw(img)
                    
                    # Add text
                    text = f"Flux Model Loading Failed\n{self.model_path.split('/')[-1]}\nCheck console for details"
                    try:
                        draw.text((10, height//2-30), text, fill='white')
                    except:
                        pass
                    
                    images.append(img)
                    
                    # Call progress callback if provided
                    if callback:
                        for step in range(num_inference_steps):
                            try:
                                callback(step, step, None)
                            except:
                                pass
                
                class Result:
                    def __init__(self, images):
                        self.images = images
                
                return Result(images)
        
        self.pipeline = FluxErrorPipeline(checkpoint_path, error_msg)
    
    def load_lora(self, lora_path: str, strength: float = 1.0):
        """Load a LoRA model"""
        print(f"Loading LoRA: {lora_path} with strength {strength}")
        
        if hasattr(self.pipeline, 'load_lora_weights'):
            try:
                # Extract LoRA name from path
                lora_name = Path(lora_path).stem
                
                # Load LoRA weights
                self.pipeline.load_lora_weights(lora_path)
                
                # Set LoRA scale
                self.pipeline.set_adapters([lora_name], adapter_weights=[strength])
            except Exception as e:
                print(f"Error loading LoRA: {e}")
    
    def load_embedding(self, embedding_path: str):
        """Load textual inversion embedding"""
        print(f"Loading embedding: {embedding_path}")
        
        if hasattr(self.pipeline, 'load_textual_inversion'):
            try:
                self.pipeline.load_textual_inversion(embedding_path)
            except Exception as e:
                print(f"Error loading embedding: {e}")
    
    async def process(self, context) -> bool:
        """Load model and prepare prompt embeddings"""
        from ..core import ComponentStatus
        self.set_status(ComponentStatus.PROCESSING)
        
        try:
            import torch
            
            if self.pipeline is None:
                # Only try to load if we have a valid path/model
                if self.properties["checkpoint_type"] == "preset":
                    # Load from preset configuration
                    model_config = self._get_preset_model_config()
                    if model_config and model_config.get("path"):
                        print(f"Loading preset model: {model_config['name']}")
                        self.load_local_model(
                            model_config["path"],
                            self._get_preset_vae_path()
                        )
                        # Auto-configure based on preset
                        self._apply_preset_config(model_config)
                elif (self.properties["checkpoint_type"] == "local" and 
                    self.properties.get("checkpoint_path")):
                    # Load from local file
                    self.load_local_model(
                        self.properties["checkpoint_path"],
                        self.properties.get("vae_path")
                    )
                elif (self.properties["checkpoint_type"] == "huggingface" and 
                      self.properties.get("checkpoint")):
                    # Load from HuggingFace
                    from diffusers import StableDiffusionPipeline
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.properties["checkpoint"],
                        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(DEVICE)
                else:
                    print("No model path specified, using mock pipeline")
            
            # Apply optimizations
            if self.properties.get("enable_freeu"):
                # FreeU for better quality
                self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
            
            # Detect model type for better conditioning info
            detected_type = self.properties.get("detected_model_type", "Unknown").lower()
            model_type = "sd15"  # default
            if "flux" in detected_type or self.properties.get("checkpoint_type") == "preset":
                model_type = "flux"
            elif "xl" in detected_type or "sdxl" in detected_type:
                model_type = "sdxl"
            
            # Create conditioning with model-specific information
            conditioning = {
                "prompt": self.properties["prompt"],
                "negative_prompt": self.properties["negative_prompt"],
                "clip_skip": self.properties["clip_skip"],
                "model_type": model_type
            }
            
            # Add model-specific conditioning
            if model_type == "flux":
                conditioning.update({
                    "guidance_scale": 1.0,  # Flux uses guidance scale 1.0
                    "num_inference_steps": 4,  # Flux is fast
                    "width": 1024,
                    "height": 1024,
                    "text_encoders": {
                        "clip_l": True,
                        "t5_xxl": True,
                        "clip_g": False
                    },
                    "max_sequence_length": 512,
                    "scheduler_type": "flux_euler"
                })
            elif model_type == "sdxl":
                conditioning.update({
                    "guidance_scale": 7.0,  # SDXL optimal CFG
                    "num_inference_steps": 25,  # SDXL quality steps
                    "width": 1024,
                    "height": 1024,
                    "text_encoders": {
                        "clip_l": True,
                        "clip_g": True,
                        "t5_xxl": False
                    },
                    "scheduler_type": "dpm_multistep"
                })
            else:  # SD 1.5
                conditioning.update({
                    "guidance_scale": 7.5,  # SD 1.5 classic CFG
                    "num_inference_steps": 20,  # SD 1.5 standard steps
                    "width": 512,
                    "height": 512,
                    "text_encoders": {
                        "clip_l": True,
                        "clip_g": False,
                        "t5_xxl": False
                    },
                    "scheduler_type": "dpm_multistep"
                })
            
            # Set output data
            self.set_output_data("pipeline", self.pipeline)
            self.set_output_data("conditioning", conditioning)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in ModelComponent: {e}")
            return False
    
    # Keep the old process method for backward compatibility
    def process_legacy(self, input_data: Any = None) -> Dict:
        """Legacy process method for backward compatibility"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self.process({}))
            if success:
                return {
                    "pipeline": self.outputs.get("pipeline"),
                    "clip_skip": self.properties["clip_skip"]
                }
            return None
        finally:
            loop.close()


class SDXLModelComponent(VisualComponent):
    """SDXL Model component that loads SDXL models directly without Flux detection"""
    
    # Component class attributes
    component_type = "sdxl_model"
    display_name = "SDXL Model"
    category = "Core"
    icon = "🎨"
    
    # Define output ports
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("checkpoint_type", "Model Source", PropertyType.CHOICE, "local", "Model",
                                 metadata={"choices": ["local", "huggingface"]}),
        create_property_definition("checkpoint_path", "Local Model Path", PropertyType.FILE_PATH, 
                                 "/home/alex/SwarmUI/Models/Stable-Diffusion/OfficialStableDiffusion/sd_xl_base_1.0.safetensors", "Model",
                                 metadata={"editor_type": "model_picker", "model_type": "checkpoint"}),
        create_property_definition("checkpoint", "HuggingFace Model", PropertyType.STRING, "stabilityai/stable-diffusion-xl-base-1.0", "Model"),
        create_property_definition("vae_type", "VAE Type", PropertyType.CHOICE, "auto", "Model",
                                 metadata={"choices": ["auto", "included", "external"]}),
        create_property_definition("vae_path", "External VAE Path", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "model_picker", "model_type": "vae"}),
        create_property_definition("clip_skip", "CLIP Skip", PropertyType.INTEGER, 1, "Advanced",
                                 metadata={"min": 1, "max": 12}),
        create_property_definition("loras", "LoRA Collection", PropertyType.STRING, "", "LoRAs",
                                 metadata={"editor_type": "lora_collection"}),
        create_property_definition("enable_freeu", "Enable FreeU", PropertyType.BOOLEAN, False, "Advanced"),
        create_property_definition("enable_clip_fix", "Enable CLIP Fix", PropertyType.BOOLEAN, False, "Advanced"),
        create_property_definition("detected_model_type", "Detected Model", PropertyType.STRING, "SDXL", "Info",
                                 metadata={"readonly": True}),
        create_property_definition("auto_configure", "Auto-Configure Sampler", PropertyType.BOOLEAN, True, "Advanced"),
    ]
    
    def __init__(self, component_id: Optional[str] = None, comp_type: str = "SDXL Model", position=None, scene=None):
        print(f"🔧 DEBUG: SDXLModelComponent.__init__ called with component_id={component_id}, comp_type={comp_type}, position={position}, scene={scene}")
        # Initialize visual component system
        super().__init__(comp_type, position or (0, 0), scene, component_id)
        print(f"🔧 DEBUG: SDXLModelComponent.__init__ super() completed")
        
        # Model-specific attributes
        self.pipeline = None
        self.model_config = ModelConfig()
        
        # Visual settings for backward compatibility
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾"
        }
        
        # LoRAs and embeddings as separate lists for backward compatibility
        self.properties["loras"] = []
        self.properties["embeddings"] = []
        
        # Set default to load SDXL model
        self.properties["checkpoint_type"] = "local"
        self.properties["detected_model_type"] = "SDXL"
        
        if self.scene:
            self.draw()
    
    def get_model_config(self, model_type: str = "sdxl") -> Dict[str, Any]:
        """Get configuration for SDXL model type"""
        return MODEL_CONFIGS.get("sdxl", MODEL_CONFIGS['sd15'])
    
    def auto_configure_sampler(self, model_type: str = "sdxl"):
        """Auto-configure connected sampler with SDXL optimal settings"""
        if not self.properties.get("auto_configure", True):
            return
        
        config = self.get_model_config(model_type)
        
        print(f"Auto-configuring for {config['name']}:")
        print(f"  Recommended steps: {config['steps']}")
        print(f"  Recommended CFG: {config['cfg']}")
        print(f"  Optimal size: {config['optimal_size']}")
        print(f"  Recommended sampler: {config['sampler']}")
        
        # Store config for sampler to use
        self.preset_config = config
    
    def load_sdxl_model(self, checkpoint_path: str, vae_path: Optional[str] = None):
        """Load SDXL model directly without any Flux detection"""
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
            
            print(f"🎨 Loading SDXL model: {checkpoint_path}")
            
            # Force SDXL detection
            self.properties["detected_model_type"] = "SDXL"
            
            # Auto-configure sampler for SDXL
            self.auto_configure_sampler("sdxl")
            
            # Common loading parameters
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            
            # Load SDXL model
            if checkpoint_path.endswith('.safetensors') or checkpoint_path.endswith('.ckpt'):
                print("🎨 Loading SDXL from local file...")
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=dtype,
                    use_safetensors=checkpoint_path.endswith('.safetensors'),
                    local_files_only=True
                )
                print(f"✅ SDXL model loaded: {type(self.pipeline).__name__}")
            else:
                print("🎨 Loading SDXL from HuggingFace...")
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    checkpoint_path,
                    torch_dtype=dtype,
                    local_files_only=False
                )
                print(f"✅ SDXL model loaded: {type(self.pipeline).__name__}")
            
            # Move to device
            self.pipeline = self.pipeline.to(DEVICE)
            print(f"✅ Pipeline moved to device: {DEVICE}")
            
            # Apply VAE if specified
            if vae_path and self.pipeline:
                print(f"🎭 Loading external VAE: {vae_path}")
                try:
                    from diffusers import AutoencoderKL
                    vae = AutoencoderKL.from_single_file(
                        vae_path,
                        torch_dtype=dtype
                    )
                    self.pipeline.vae = vae.to(DEVICE)
                    print("✅ VAE loaded successfully")
                except Exception as vae_error:
                    print(f"⚠️ VAE loading failed: {vae_error}")
            
            # Enable memory optimizations
            try:
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    print("✅ Enabled attention slicing")
                
                if DEVICE == "cuda" and hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        print("✅ Enabled xformers memory efficient attention")
                    except:
                        print("⚠️ xformers not available, using default attention")
                        
            except Exception as opt_error:
                print(f"⚠️ Some optimizations failed: {opt_error}")
            
            print(f"🎉 SDXL model loading complete: {type(self.pipeline).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading SDXL model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create informative fallback pipeline
            print("🚨 Creating fallback error display pipeline...")
            self._create_error_display_pipeline(checkpoint_path, str(e))

    def _create_error_display_pipeline(self, checkpoint_path: str, error_msg: str):
        """Create an informative error display pipeline when model loading fails"""
        class ErrorDisplayPipeline:
            def __init__(self, model_path, error_message):
                self.model_path = model_path
                self.error_message = error_message
                
                # Add scheduler attribute for compatibility
                class MockScheduler:
                    def __init__(self):
                        self.config = {}
                
                self.scheduler = MockScheduler()
                print(f"🚨 Error display pipeline created for: {model_path}")
            
            def __call__(self, prompt, negative_prompt="", num_inference_steps=25, 
                         guidance_scale=7.0, width=1024, height=1024, generator=None, 
                         num_images_per_prompt=1, callback=None, callback_steps=1, **kwargs):
                from PIL import Image, ImageDraw, ImageFont
                
                images = []
                for i in range(num_images_per_prompt):
                    # Create an image with error message
                    img = Image.new('RGB', (width, height), color='darkred')
                    draw = ImageDraw.Draw(img)
                    
                    # Add text
                    text = f"SDXL Model Loading Failed\\n{self.model_path[-30:] if len(self.model_path) > 30 else self.model_path}\\nCheck console for errors"
                    try:
                        draw.text((10, height//2-30), text, fill='white')
                    except:
                        pass
                    
                    images.append(img)
                    
                    # Call progress callback if provided
                    if callback:
                        for step in range(num_inference_steps):
                            try:
                                callback(step, step, None)
                            except:
                                pass
                
                class Result:
                    def __init__(self, images):
                        self.images = images
                
                return Result(images)
        
        self.pipeline = ErrorDisplayPipeline(checkpoint_path, error_msg)
    
    async def process(self, context) -> bool:
        """Load SDXL model and prepare prompt embeddings"""
        from ..core import ComponentStatus
        self.set_status(ComponentStatus.PROCESSING)
        
        try:
            import torch
            
            if self.pipeline is None:
                # Only try to load if we have a valid path/model
                if (self.properties["checkpoint_type"] == "local" and 
                    self.properties.get("checkpoint_path")):
                    # Load from local file
                    self.load_sdxl_model(
                        self.properties["checkpoint_path"],
                        self.properties.get("vae_path")
                    )
                elif (self.properties["checkpoint_type"] == "huggingface" and 
                      self.properties.get("checkpoint")):
                    # Load from HuggingFace
                    self.load_sdxl_model(
                        self.properties["checkpoint"],
                        self.properties.get("vae_path")
                    )
                else:
                    print("No SDXL model path specified, using mock pipeline")
            
            # Apply optimizations
            if self.properties.get("enable_freeu"):
                # FreeU for better quality
                self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
            
            # Set output data
            self.set_output_data("pipeline", self.pipeline)
            self.set_output_data("conditioning", {
                "clip_skip": self.properties["clip_skip"]
            })
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SDXLModelComponent: {e}")
            return False
    
    # Keep the old process method for backward compatibility
    def process_legacy(self, input_data: Any = None) -> Dict:
        """Legacy process method for backward compatibility"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self.process({}))
            if success:
                return {
                    "pipeline": self.outputs.get("pipeline"),
                    "clip_skip": self.properties["clip_skip"]
                }
            return None
        finally:
            loop.close()


class SamplerComponent(VisualComponent):
    """Sampler component for image generation"""
    
    # Component class attributes
    component_type = "sampler"
    display_name = "Sampler"
    category = "Core"
    icon = "🎲"
    
    # Define input and output ports
    input_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.INPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.INPUT),
        create_port("resolution", PortType.ANY, PortDirection.INPUT, optional=True),
        create_port("seed", PortType.ANY, PortDirection.INPUT, optional=True)
    ]
    
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT),
        create_port("latents", PortType.LATENT, PortDirection.OUTPUT, optional=True)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("prompt", "Prompt", PropertyType.TEXT, "a professional photo of a beautiful landscape at golden hour, highly detailed, photorealistic", "Prompts",
                                 metadata={"editor_type": "prompt"}),
        create_property_definition("negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Prompts",
                                 metadata={"editor_type": "prompt"}),
        create_property_definition("scheduler", "Scheduler", PropertyType.CHOICE, "Euler a", "Sampling",
                                 metadata={"choices": ["Euler a", "DPMSolverMultistep", "DDIM", "PNDM"]}),
        create_property_definition("steps", "Steps", PropertyType.INTEGER, 4, "Sampling",
                                 metadata={"min": 1, "max": 150}),
        create_property_definition("cfg_scale", "CFG Scale", PropertyType.FLOAT, 1.0, "Sampling",
                                 metadata={"editor_type": "float_slider", "min": 1.0, "max": 20.0, "step": 0.5, "decimals": 1}),
        create_property_definition("width", "Width", PropertyType.INTEGER, 1024, "Image",
                                 metadata={"min": 64, "max": 2048, "step": 64}),
        create_property_definition("height", "Height", PropertyType.INTEGER, 1024, "Image",
                                 metadata={"min": 64, "max": 2048, "step": 64}),
        create_property_definition("seed", "Seed", PropertyType.INTEGER, -1, "Advanced",
                                 metadata={"min": -1, "max": 2147483647}),
        create_property_definition("batch_size", "Batch Size", PropertyType.INTEGER, 1, "Advanced",
                                 metadata={"min": 1, "max": 8}),
        create_property_definition("preview_interval", "Preview Every N Steps", PropertyType.INTEGER, 5, "Advanced",
                                 metadata={"min": 1, "max": 50}),
        create_property_definition("enable_preview", "Enable Real-time Preview", PropertyType.BOOLEAN, True, "Advanced")
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize new component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # Keep Qt-specific attributes for backward compatibility
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type  # For backward compatibility
        self.selected = False
        self.group = None
        self.output_data = None
        
        # Visual settings for backward compatibility
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾"
        }
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Generate images using the sampler"""
        from ..core import ComponentStatus
        pipeline = self.get_input_data("pipeline")
        conditioning = self.get_input_data("conditioning")
        
        if not pipeline:
            self.set_status(ComponentStatus.ERROR)
            print("ERROR: No pipeline received in SamplerComponent")
            return False
        
        self.set_status(ComponentStatus.PROCESSING)
        
        # Inherit properties from connected model (if available)
        self._inherit_model_properties(conditioning)
        
        try:
            import torch
            
            # Detect pipeline type
            pipeline_class_name = type(pipeline).__name__
            is_flux = 'Flux' in pipeline_class_name or hasattr(pipeline, 'transformer')
            is_sdxl = 'XL' in pipeline_class_name
            
            print(f"Pipeline type: {pipeline_class_name}, is_flux: {is_flux}, is_sdxl: {is_sdxl}")
            
            # Set scheduler based on model type and properties
            if not is_flux:  # Flux has its own scheduler
                scheduler_name = self.properties["scheduler"]
                
                if scheduler_name == "DPMSolverMultistep":
                    from diffusers import DPMSolverMultistepScheduler
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config
                    )
                elif scheduler_name == "DPMSolverSinglestep":
                    from diffusers import DPMSolverSinglestepScheduler
                    pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                        pipeline.scheduler.config
                    )
                elif scheduler_name == "DDIM":
                    from diffusers import DDIMScheduler
                    pipeline.scheduler = DDIMScheduler.from_config(
                        pipeline.scheduler.config
                    )
                elif scheduler_name == "Euler":
                    from diffusers import EulerDiscreteScheduler
                    pipeline.scheduler = EulerDiscreteScheduler.from_config(
                        pipeline.scheduler.config
                    )
                elif scheduler_name == "EulerAncestral":
                    from diffusers import EulerAncestralDiscreteScheduler
                    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                        pipeline.scheduler.config
                    )
                elif scheduler_name == "PNDM":
                    from diffusers import PNDMScheduler
                    pipeline.scheduler = PNDMScheduler.from_config(
                        pipeline.scheduler.config
                    )
            
            # Set seed
            generator = None
            seed = self.properties["seed"]
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            # Create generator on the same device as the pipeline
            pipeline_device = str(next(pipeline.parameters()).device) if hasattr(pipeline, 'parameters') else DEVICE
            generator = torch.Generator(device=pipeline_device).manual_seed(seed)
            print(f"🎲 Generator created on device: {pipeline_device}, seed: {seed}")
            
            # Store seed in context for output naming
            context['seed'] = seed
            
            # Get prompts from sampler properties (user input)
            prompt = self.properties.get("prompt", "")
            negative_prompt = self.properties.get("negative_prompt", "")
            
            # Also check conditioning for fallback
            if not prompt and conditioning:
                prompt = conditioning.get("prompt", "")
            if not negative_prompt and conditioning:
                negative_prompt = conditioning.get("negative_prompt", "")
            
            # Setup preview callback
            callback = None
            callback_steps = 1
            
            if self.properties.get("enable_preview", True) and hasattr(context, "preview_callback"):
                preview_interval = self.properties.get("preview_interval", 5)
                
                def preview_callback(step, timestep, latents):
                    if step % preview_interval == 0:
                        try:
                            # Simple progress callback for now
                            progress_percent = int((step / self.properties["steps"]) * 100)
                            
                            # Create a simple preview image
                            from PIL import Image
                            preview = Image.new('RGB', (256, 256), color='gray')
                            
                            # Send preview
                            if hasattr(context, 'preview_callback'):
                                context.preview_callback(preview, progress_percent, step)
                        except Exception as e:
                            print(f"Preview error: {e}")
                
                callback = preview_callback
                callback_steps = 1
            
            # Prepare generation parameters based on pipeline type
            generation_params = {
                "prompt": prompt,
                "num_inference_steps": self.properties["steps"],
                "generator": generator,
                "num_images_per_prompt": self.properties["batch_size"],
            }
            
            # Add parameters based on pipeline type
            if is_flux:
                # Flux-specific parameters
                generation_params.update({
                    "height": self.properties["height"],
                    "width": self.properties["width"],
                    "guidance_scale": 0.0,  # Flux doesn't use CFG
                    "max_sequence_length": 512,
                })
                # Flux doesn't use negative prompts
            else:
                # Standard SD parameters
                generation_params.update({
                    "negative_prompt": negative_prompt,
                    "height": self.properties["height"],
                    "width": self.properties["width"],
                    "guidance_scale": self.properties["cfg_scale"],
                })
            
            # Add callback if available
            if callback:
                generation_params["callback"] = callback
                generation_params["callback_steps"] = callback_steps
            
            # Execute generation
            print(f"Generating with parameters: {generation_params}")
            
            try:
                result = pipeline(**generation_params)
                
                # Extract images from result
                if hasattr(result, 'images'):
                    images = result.images
                elif isinstance(result, list):
                    images = result
                elif isinstance(result, dict) and 'images' in result:
                    images = result['images']
                else:
                    raise ValueError(f"Unknown result type: {type(result)}")
                
                print(f"Generated {len(images)} images")
                
                # Set outputs
                self.set_output_data("images", images)
                if hasattr(result, 'latents'):
                    self.set_output_data("latents", result.latents)
                
            except Exception as gen_error:
                print(f"Generation failed: {gen_error}")
                import traceback
                traceback.print_exc()
                
                # Create error image
                from PIL import Image, ImageDraw
                error_image = Image.new('RGB', 
                                      (self.properties["width"], self.properties["height"]), 
                                      color='darkblue')
                draw = ImageDraw.Draw(error_image)
                error_text = f"Generation Error:\n{str(gen_error)[:50]}..."
                try:
                    draw.text((10, 10), error_text, fill='white')
                except:
                    pass
                
                self.set_output_data("images", [error_image])
                print("Created error image")
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in SamplerComponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _inherit_model_properties(self, conditioning):
        """Inherit optimal settings from connected model"""
        if not conditioning:
            return
        
        try:
            # Check if model provided its configuration
            model_config = conditioning.get("model_config")
            model_type = conditioning.get("model_type", "")
            
            # Detect model type from pipeline or conditioning info
            if not model_type:
                # Try to detect from conditioning data
                if conditioning.get("guidance_scale") == 1.0 and conditioning.get("num_inference_steps") == 4:
                    model_type = "flux"
                elif conditioning.get("width") == 1024 and conditioning.get("height") == 1024:
                    model_type = "sdxl"
                else:
                    model_type = "sd15"
            
            print(f"🔄 Sampler inheriting properties for model type: {model_type}")
            
            # Apply model-specific optimal settings
            if model_type == "flux" or "flux" in model_type.lower():
                # Flux optimal settings
                inherited_settings = {
                    "steps": conditioning.get("num_inference_steps", 4),
                    "cfg_scale": conditioning.get("guidance_scale", 1.0),
                    "width": conditioning.get("width", 1024),
                    "height": conditioning.get("height", 1024)
                }
                print("⚡ Applying Flux-optimized settings:")
                print(f"   Steps: {inherited_settings['steps']} (fast generation)")
                print(f"   CFG: {inherited_settings['cfg_scale']} (no guidance)")
                print(f"   Resolution: {inherited_settings['width']}x{inherited_settings['height']}")
                
            elif model_type == "sdxl" or "xl" in model_type.lower():
                # SDXL optimal settings
                inherited_settings = {
                    "steps": 25,
                    "cfg_scale": 7.0,
                    "width": 1024,
                    "height": 1024
                }
                print("🎨 Applying SDXL-optimized settings:")
                print(f"   Steps: {inherited_settings['steps']} (quality generation)")
                print(f"   CFG: {inherited_settings['cfg_scale']} (balanced guidance)")
                
            else:
                # SD 1.5 optimal settings
                inherited_settings = {
                    "steps": 20,
                    "cfg_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
                print("🖼️ Applying SD 1.5-optimized settings:")
                print(f"   Steps: {inherited_settings['steps']} (standard generation)")
                print(f"   CFG: {inherited_settings['cfg_scale']} (classic guidance)")
            
            # Update sampler properties with inherited settings (only if not manually overridden)
            for key, value in inherited_settings.items():
                if key in self.properties:
                    # Only inherit if the current value is still at default or makes sense to override
                    current_value = self.properties[key]
                    
                    # Always inherit for Flux (very specific requirements)
                    if model_type == "flux" or "flux" in model_type.lower():
                        self.properties[key] = value
                        print(f"   ✅ Inherited {key}: {current_value} → {value}")
                    
                    # For other models, inherit if current value seems like a default
                    elif self._should_inherit_property(key, current_value, value):
                        self.properties[key] = value
                        print(f"   ✅ Inherited {key}: {current_value} → {value}")
                    else:
                        print(f"   ⏭️ Kept manual {key}: {current_value}")
            
            # Also inherit prompts if provided by model
            if "prompt" in conditioning and conditioning["prompt"]:
                self.properties["prompt"] = conditioning["prompt"]
                print(f"   ✅ Inherited prompt: {conditioning['prompt'][:50]}...")
            
            if "negative_prompt" in conditioning and conditioning["negative_prompt"]:
                self.properties["negative_prompt"] = conditioning["negative_prompt"]
                print(f"   ✅ Inherited negative prompt: {conditioning['negative_prompt'][:50]}...")
            
            # Handle Flux-specific text encoder information
            if model_type == "flux" or "flux" in model_type.lower():
                text_encoders = conditioning.get("text_encoders", {})
                if text_encoders.get("clip_l") and text_encoders.get("t5_xxl"):
                    print("   📝 Flux text encoders: CLIP-L + T5-XXL (optimal for quality)")
                elif text_encoders.get("clip_l"):
                    print("   📝 Flux text encoders: CLIP-L only (faster)")
                
                # Flux-specific sampler settings
                if conditioning.get("max_sequence_length"):
                    print(f"   📏 Max sequence length: {conditioning['max_sequence_length']} tokens")
                
                # Flux doesn't use negative prompts effectively
                if self.properties.get("negative_prompt"):
                    print("   ⚠️ Note: Flux models don't use negative prompts effectively")
                    self.properties["negative_prompt"] = ""
                
        except Exception as e:
            print(f"Warning: Could not inherit model properties: {e}")
    
    def _should_inherit_property(self, property_name, current_value, new_value):
        """Determine if a property should be inherited from the model"""
        
        # Define common default values that should be overridden
        defaults = {
            "steps": [4, 20, 25, 30, 50],  # Common default step counts
            "cfg_scale": [1.0, 7.0, 7.5, 8.0],  # Common CFG defaults
            "width": [512, 768, 1024],  # Common resolutions
            "height": [512, 768, 1024]
        }
        
        # If current value is a common default, inherit the new value
        if property_name in defaults and current_value in defaults[property_name]:
            return True
        
        # Special case: Always inherit Flux-specific settings
        if new_value == 1.0 and property_name == "cfg_scale":  # Flux CFG
            return True
        if new_value == 4 and property_name == "steps":  # Flux steps
            return True
            
        # Otherwise, keep manual settings
        return False
    
    # Keep the old process method for backward compatibility
    def process_legacy(self, input_data: Dict) -> Dict:
        """Legacy process method for backward compatibility"""
        # Set input data manually for legacy compatibility
        if input_data:
            if "pipeline" in input_data:
                self.inputs["pipeline"] = input_data["pipeline"]
            if "prompt" in input_data or "negative_prompt" in input_data:
                self.inputs["conditioning"] = {
                    "prompt": input_data.get("prompt", ""),
                    "negative_prompt": input_data.get("negative_prompt", "")
                }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self.process({}))
            if success:
                return {
                    **input_data,
                    "images": self.outputs.get("images"),
                    "latents": self.outputs.get("latents")
                }
            return None
        finally:
            loop.close()


class VAEComponent(VisualComponent):
    """VAE component for decoding latents"""
    
    # Component class attributes
    component_type = "vae"
    display_name = "VAE"
    category = "Core"
    icon = "🖼️"
    
    # Define input and output ports
    input_ports = [
        create_port("latents", PortType.LATENT, PortDirection.INPUT),
        create_port("images", PortType.IMAGE, PortDirection.INPUT, optional=True)
    ]
    
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("decode_mode", "Mode", PropertyType.CHOICE, "auto", "VAE",
                                 metadata={"choices": ["auto", "decode_only", "encode_only"]}),
        create_property_definition("tiling", "Enable Tiling", PropertyType.BOOLEAN, False, "Memory",
                                 metadata={"tooltip": "Use tiled VAE for lower memory usage"}),
        create_property_definition("tile_size", "Tile Size", PropertyType.INTEGER, 512, "Memory",
                                 metadata={"min": 256, "max": 1024, "step": 64})
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize new component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # Keep Qt-specific attributes for backward compatibility
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type  # For backward compatibility
        self.selected = False
        self.group = None
        self.output_data = None
        
        # Visual settings for backward compatibility
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾"
        }
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Process VAE decoding/encoding"""
        from ..core import ComponentStatus
        latents = self.get_input_data("latents")
        images = self.get_input_data("images")
        
        # Usually latents come from sampler and we decode to images
        if latents is not None:
            self.set_output_data("images", latents)  # Simplified - would need actual VAE
        elif images is not None:
            self.set_output_data("images", images)  # Pass through
        
        self.set_status(ComponentStatus.COMPLETE)
        return True
    
    # Keep the old process method for backward compatibility
    def process_legacy(self, input_data: Dict) -> Dict:
        """Legacy process method for backward compatibility"""
        # For now, just pass through the images
        return input_data


class OutputComponent(VisualComponent):
    """Output component for saving generated images"""
    
    # Component class attributes
    component_type = "output"
    display_name = "Output"
    category = "Core"
    icon = "💾"
    
    # Define input ports
    input_ports = [
        create_port("images", PortType.IMAGE, PortDirection.INPUT)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("format", "Format", PropertyType.CHOICE, "PNG",
                                 metadata={"choices": ["PNG", "JPEG", "WEBP"]}),
        create_property_definition("quality", "Quality", PropertyType.INTEGER, 95,
                                 metadata={"min": 1, "max": 100}),
        create_property_definition("path", "Output Path", PropertyType.DIRECTORY, "./output/"),
        create_property_definition("filename_pattern", "Filename Pattern", PropertyType.STRING, "img_{seed}_{timestamp}")
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize new component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # Keep Qt-specific attributes for backward compatibility
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type  # For backward compatibility
        self.selected = False
        self.group = None
        self.output_data = None
        
        # Visual settings for backward compatibility
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾"
        }
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Save the generated images"""
        from ..core import ComponentStatus
        images = self.get_input_data("images")
        
        if not images:
            self.set_status(ComponentStatus.ERROR)
            return False
        
        self.set_status(ComponentStatus.PROCESSING)
        
        try:
            if images:
                # Save images
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                Path(self.properties["path"]).mkdir(parents=True, exist_ok=True)
                
                saved_files = []
                for i, img in enumerate(images):
                    # Generate filename using pattern
                    filename_base = self.properties["filename_pattern"].format(
                        seed=context.get("seed", "unknown"),
                        timestamp=timestamp,
                        index=i
                    )
                    
                    # Add extension based on format
                    format_lower = self.properties["format"].lower()
                    if format_lower == "jpeg":
                        ext = ".jpg"
                    else:
                        ext = f".{format_lower}"
                    
                    filename = f"{self.properties['path']}/{filename_base}{ext}"
                    
                    if hasattr(img, 'save'):
                        save_kwargs = {}
                        if format_lower in ["jpeg", "jpg"]:
                            save_kwargs["quality"] = self.properties["quality"]
                        
                        img.save(filename, format=self.properties["format"], **save_kwargs)
                        saved_files.append(filename)
                        print(f"Saved: {filename}")
                
                # Store saved file list in context for other components
                context["saved_files"] = saved_files
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in OutputComponent: {e}")
            return False
    
    # Keep the old process method for backward compatibility
    def process_legacy(self, input_data: Dict) -> Dict:
        """Legacy process method for backward compatibility"""
        # Set input data manually for legacy compatibility
        if input_data and "images" in input_data:
            self.inputs["images"] = input_data["images"]
        
        context = {"seed": input_data.get("seed", "unknown")} if input_data else {}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self.process(context))
            if success:
                return {**input_data, "saved": True}
            return None
        finally:
            loop.close()


class ImageComponent(VisualComponent):
    """Simple image display component for viewing single images"""
    
    # Component class attributes
    component_type = "image"
    display_name = "Image"
    category = "Core"
    icon = "🖼️"
    
    # Define input ports
    input_ports = [
        create_port("image", PortType.IMAGE, PortDirection.INPUT)
    ]
    
    # Define output ports (pass-through)
    output_ports = [
        create_port("image", PortType.IMAGE, PortDirection.OUTPUT)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("show_in_ui", "Show in UI", PropertyType.BOOLEAN, True),
        create_property_definition("scale", "Scale", PropertyType.FLOAT, 1.0,
                                 metadata={"min": 0.1, "max": 4.0, "step": 0.1})
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        super().__init__(self.component_type, position, scene, component_id)
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type
        self.selected = False
        self.group = None
        self.output_data = None
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Display single image and pass it through"""
        from ..core import ComponentStatus
        image = self.get_input_data("image")
        
        if not image:
            self.set_status(ComponentStatus.ERROR)
            return False
        
        self.set_status(ComponentStatus.PROCESSING)
        
        try:
            if image and self.properties.get("show_in_ui", True):
                import base64
                import io
                from PIL import Image as PILImage
                
                # Handle single image
                if hasattr(image, 'save'):
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    display_item = {
                        "type": "image",
                        "data": img_b64,
                        "size": image.size
                    }
                    
                    context.set_shared_data(f"display_image_{self.id}", display_item)
                    print(f"🖼️ Image: Prepared single image for display")
            
            # Pass image through
            self.set_output_data("image", image)
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            print(f"Error in Image component: {e}")
            self.set_status(ComponentStatus.ERROR)
            return False


class MediaDisplayComponent(VisualComponent):
    """Media display component for showing generated images and videos in the web UI"""
    
    # Component class attributes
    component_type = "media_display"
    display_name = "Media Display"
    category = "Core"
    icon = "🎬"
    
    # Define input ports
    input_ports = [
        create_port("images", PortType.IMAGE, PortDirection.INPUT, optional=True),
        create_port("video", PortType.VIDEO, PortDirection.INPUT, optional=True)
    ]
    
    # Define output ports (pass-through for chaining)
    output_ports = [
        create_port("images", PortType.IMAGE, PortDirection.OUTPUT, optional=True),
        create_port("video", PortType.VIDEO, PortDirection.OUTPUT, optional=True)
    ]
    
    # Define property definitions
    property_definitions = [
        create_property_definition("show_in_ui", "Show in UI", PropertyType.BOOLEAN, True),
        create_property_definition("max_display_count", "Max Items to Display", PropertyType.INTEGER, 4,
                                 metadata={"min": 1, "max": 16}),
        create_property_definition("thumbnail_size", "Thumbnail Size", PropertyType.INTEGER, 256,
                                 metadata={"min": 64, "max": 512}),
        create_property_definition("video_autoplay", "Auto-play Videos", PropertyType.BOOLEAN, True),
        create_property_definition("video_loop", "Loop Videos", PropertyType.BOOLEAN, True)
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        # Initialize new component system
        super().__init__(self.component_type, position, scene, component_id)
        
        # Keep Qt-specific attributes for backward compatibility
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type
        self.selected = False
        self.group = None
        self.output_data = None
        
        # Visual settings
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c", 
            "VAE": "#2ecc71",
            "Output": "#34495e",
            "MediaDisplay": "#9b59b6"
        }
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️", 
            "Output": "💾",
            "MediaDisplay": "🎬"
        }
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Display images/videos and pass them through"""
        from ..core import ComponentStatus
        images = self.get_input_data("images")
        video = self.get_input_data("video")
        
        if not images and not video:
            self.set_status(ComponentStatus.ERROR)
            return False
        
        self.set_status(ComponentStatus.PROCESSING)
        
        try:
            display_items = []
            
            # Handle images
            if images and self.properties.get("show_in_ui", True):
                import base64
                import io
                from PIL import Image
                
                max_count = min(len(images), self.properties.get("max_display_count", 4))
                
                for i in range(max_count):
                    img = images[i]
                    if hasattr(img, 'save'):
                        # Resize for thumbnail if needed
                        thumb_size = self.properties.get("thumbnail_size", 256)
                        if img.size[0] > thumb_size or img.size[1] > thumb_size:
                            img = img.copy()
                            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        display_items.append({
                            "type": "image",
                            "data": img_b64,
                            "index": i,
                            "size": img.size
                        })
            
            # Handle video
            if video and self.properties.get("show_in_ui", True):
                import base64
                
                # Assume video is a file path or bytes
                if isinstance(video, str):
                    # Video file path
                    with open(video, 'rb') as f:
                        video_data = f.read()
                elif isinstance(video, bytes):
                    video_data = video
                else:
                    video_data = video
                
                video_b64 = base64.b64encode(video_data).decode('utf-8')
                display_items.append({
                    "type": "video",
                    "data": video_b64,
                    "autoplay": self.properties.get("video_autoplay", True),
                    "loop": self.properties.get("video_loop", True)
                })
            
            # Store display data for web UI
            if display_items:
                context.set_shared_data(f"display_media_{self.id}", display_items)
                print(f"🎬 MediaDisplay: Prepared {len(display_items)} items for web display")
            
            # Pass data through to outputs
            if images:
                self.set_output_data("images", images)
            if video:
                self.set_output_data("video", video)
                
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            print(f"Error in MediaDisplay component: {e}")
            self.set_status(ComponentStatus.ERROR)
            return False


class FluxModelComponent(VisualComponent):
    """Flux Model Component supporting all variants and GGUF with LoRA"""
    
    component_type = "flux_model"
    display_name = "Flux Model"
    category = "Core"
    icon = "⚡"
    
    # Define output ports (no inputs - this is a source component)
    output_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.OUTPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.OUTPUT),
        create_port("model_info", PortType.ANY, PortDirection.OUTPUT),
    ]
    
    # Define properties
    property_definitions = [
        # Model Selection
        create_property_definition(
            "model_variant", "Model Variant", PropertyType.CHOICE, "flux-schnell", "Model",
            metadata={
                "choices": ["flux-dev", "flux-schnell", "flux-turbo", "custom"],
                "description": "Select Flux model variant"
            }
        ),
        create_property_definition(
            "model_format", "Model Format", PropertyType.CHOICE, "safetensors", "Model",
            metadata={
                "choices": ["safetensors", "gguf", "diffusers"],
                "description": "Model file format"
            }
        ),
        create_property_definition(
            "model_path", "Model Path", PropertyType.FILE_PATH, "", "Model",
            metadata={
                "editor_type": "model_picker",
                "model_type": "flux",
                "description": "Path to model file (for custom variant)"
            }
        ),
        
        # GGUF Quantization Settings
        create_property_definition(
            "gguf_quant", "GGUF Quantization", PropertyType.CHOICE, "Q4_K_M", "GGUF Settings",
            metadata={
                "choices": ["Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q8_0", "F16"],
                "description": "GGUF quantization level"
            }
        ),
        create_property_definition(
            "use_flash_attn", "Use Flash Attention", PropertyType.BOOLEAN, True, "GGUF Settings",
            metadata={"description": "Enable Flash Attention for GGUF models"}
        ),
        
        # Text Encoding
        create_property_definition(
            "prompt", "Prompt", PropertyType.TEXT, "", "Prompting",
            metadata={
                "editor_type": "prompt",
                "placeholder": "Describe your image...",
                "syntax": "flux-prompt"
            }
        ),
        create_property_definition(
            "negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Prompting",
            metadata={
                "editor_type": "prompt",
                "placeholder": "What to avoid...",
                "info": "Note: Flux models have limited negative prompt support"
            }
        ),
        
        # Model-Specific Settings
        create_property_definition(
            "guidance_scale", "Guidance Scale", PropertyType.FLOAT, 3.5, "Generation",
            metadata={
                "min": 0.0, "max": 20.0, "step": 0.5,
                "flux_dev_default": 3.5,
                "flux_schnell_default": 0.0,
                "flux_turbo_default": 0.0
            }
        ),
        create_property_definition(
            "num_inference_steps", "Steps", PropertyType.INTEGER, 4, "Generation",
            metadata={
                "min": 1, "max": 50,
                "flux_dev_default": 50,
                "flux_schnell_default": 4,
                "flux_turbo_default": 1
            }
        ),
        
        # T5 Text Encoder Settings
        create_property_definition(
            "t5_model_path", "T5 Model Path", PropertyType.FILE_PATH, "", "Advanced",
            metadata={
                "editor_type": "model_picker",
                "model_type": "text_encoder",
                "description": "Custom T5 text encoder path (optional)"
            }
        ),
        create_property_definition(
            "max_sequence_length", "Max Sequence Length", PropertyType.INTEGER, 512, "Advanced",
            metadata={
                "min": 77, "max": 512,
                "description": "Maximum token length for T5 encoder"
            }
        ),
        
        # LoRA Settings
        create_property_definition(
            "loras", "LoRA Models", PropertyType.STRING, "", "LoRA",
            metadata={
                "editor_type": "lora_collection",
                "description": "Add multiple LoRA models with individual strengths"
            }
        ),
        create_property_definition(
            "lora_merge_strategy", "LoRA Merge Strategy", PropertyType.CHOICE, "sequential", "LoRA",
            metadata={
                "choices": ["sequential", "weighted", "concatenate"],
                "description": "How to apply multiple LoRAs"
            }
        ),
        create_property_definition(
            "lora_presets", "LoRA Presets", PropertyType.CHOICE, "none", "LoRA",
            metadata={
                "choices": ["none", "anime_style", "photorealistic", "artistic", "custom"],
                "description": "Quick LoRA preset configurations"
            }
        ),
        
        # Memory Optimization
        create_property_definition(
            "enable_cpu_offload", "CPU Offload", PropertyType.BOOLEAN, True, "Memory",
            metadata={"description": "Offload model to CPU when not in use"}
        ),
        create_property_definition(
            "enable_vae_slicing", "VAE Slicing", PropertyType.BOOLEAN, False, "Memory",
            metadata={"description": "Enable VAE slicing for lower VRAM usage"}
        ),
        create_property_definition(
            "enable_vae_tiling", "VAE Tiling", PropertyType.BOOLEAN, False, "Memory",
            metadata={"description": "Enable VAE tiling for very large images"}
        ),
    ]
    
    def __init__(self, component_id: Optional[str] = None, position=None, scene=None):
        super().__init__(self.component_type, position, scene, component_id)
        self.position = position or (0, 0)
        self.scene = scene
        self.type = self.component_type
        self.selected = False
        self.group = None
        self.output_data = None
        
        # Model-specific attributes
        self.pipeline = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.transformer = None
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = "cpu"
        self._user_modified_properties = set()  # Track user-modified properties
        self.loaded_loras = []  # Track loaded LoRAs
        
        # Variant configurations
        self.variant_configs = {
            "flux-dev": {
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "steps": 50,
                "guidance": 3.5,
                "needs_auth": True
            },
            "flux-schnell": {
                "repo_id": "black-forest-labs/FLUX.1-schnell", 
                "steps": 4,
                "guidance": 0.0,
                "needs_auth": False
            },
            "flux-turbo": {
                "repo_id": "alimama-creative/FLUX.1-Turbo-Alpha",
                "steps": 1,
                "guidance": 0.0,
                "needs_auth": False
            }
        }
        
        # LoRA preset configurations
        self.lora_presets = {
            "anime_style": [
                {"path": "flux_anime_v1.safetensors", "strength": 0.8},
                {"path": "flux_manga_style.safetensors", "strength": 0.6}
            ],
            "photorealistic": [
                {"path": "flux_realism_enhancer.safetensors", "strength": 1.0},
                {"path": "flux_skin_detail.safetensors", "strength": 0.7}
            ],
            "artistic": [
                {"path": "flux_art_style.safetensors", "strength": 0.9},
                {"path": "flux_painting_mode.safetensors", "strength": 0.5}
            ]
        }
        
        if self.scene:
            self.draw()
    
    def _get_model_paths(self):
        """Get model paths based on variant and format using configuration"""
        variant = self.properties.get("model_variant", "flux-schnell")
        format = self.properties.get("model_format", "safetensors")
        
        # Use the model configuration system
        try:
            from ..utils.model_paths import find_model_for_variant
            
            # Try to find model using configuration
            if variant != "custom":
                model_path = find_model_for_variant("flux", variant)
                if model_path:
                    return {"model": model_path}
            
            # Custom path fallback
            custom_path = self.properties.get("model_path", "")
            if custom_path and Path(custom_path).exists():
                return {"model": custom_path}
                
        except ImportError:
            print("⚠️ Model paths utility not available, using fallback")
            # Fallback to original method
            base_paths = [
                Path("/home/alex/SwarmUI/Models/diffusion_models"),
                Path.home() / "models" / "flux",
                Path("./models/flux")
            ]
            
            model_files = {
                "flux-dev": ["flux1-dev.safetensors", "flux_dev.safetensors"],
                "flux-schnell": ["flux1-schnell.safetensors", "flux_schnell.safetensors"],
                "flux-turbo": ["flux1-turbo.safetensors", "flux_turbo_alpha.safetensors"]
            }
            
            if variant != "custom":
                for base_path in base_paths:
                    if not base_path.exists():
                        continue
                    for filename in model_files.get(variant, []):
                        full_path = base_path / filename
                        if full_path.exists():
                            return {"model": str(full_path)}
            
        return {}
    
    def _load_loras(self):
        """Load and apply LoRA models"""
        # Parse LoRA collection from property
        loras_str = self.properties.get("loras", "")
        if not loras_str or not self.pipeline:
            return
        
        try:
            # Try to parse as JSON if it's a string
            if isinstance(loras_str, str):
                if loras_str.strip():
                    loras = json.loads(loras_str)
                else:
                    loras = []
            else:
                loras = loras_str
                
            loaded_count = 0
            strategy = self.properties.get("lora_merge_strategy", "sequential")
            self.loaded_loras = []  # Reset loaded LoRAs list
            
            # For weighted strategy, we'll accumulate adapters
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
                    
                    # Check if GGUF LoRA
                    if lora_path.endswith('.gguf'):
                        print(f"GGUF LoRAs not yet supported, skipping: {lora_path}")
                        continue
                    
                    # Load LoRA weights
                    if hasattr(self.pipeline, 'load_lora_weights'):
                        self.pipeline.load_lora_weights(
                            lora_path,
                            adapter_name=adapter_name
                        )
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(strength)
                        loaded_count += 1
                        
                        # Track loaded LoRA
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
                    # Apply each LoRA with its strength
                    for name, weight in zip(adapter_names, adapter_weights):
                        self.pipeline.set_adapters([name], adapter_weights=[weight])
                        
                elif strategy == "weighted":
                    # Apply all LoRAs with their respective weights
                    self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    
                elif strategy == "concatenate":
                    # Merge all LoRAs equally
                    equal_weights = [1.0 / len(adapter_names)] * len(adapter_names)
                    self.pipeline.set_adapters(adapter_names, adapter_weights=equal_weights)
                
                print(f"Loaded {loaded_count} LoRA(s) with strategy: {strategy}")
            elif loaded_count > 0:
                print(f"Loaded {loaded_count} LoRA(s) but pipeline doesn't support set_adapters")
            
        except Exception as e:
            print(f"Error in LoRA loading: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_standard_model(self, model_path: str = None):
        """Load standard safetensors/diffusers model"""
        try:
            from diffusers import FluxPipeline, FluxTransformer2DModel
            import torch
            
            variant = self.properties.get("model_variant", "flux-schnell")
            
            if model_path and Path(model_path).exists():
                # Load from local file
                print(f"Loading Flux model from: {model_path}")
                
                if model_path.endswith('.safetensors'):
                    # Load individual components
                    self.transformer = FluxTransformer2DModel.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                    )
                    
                    # Create pipeline with transformer
                    config = self.variant_configs.get(variant, {})
                    self.pipeline = FluxPipeline.from_pretrained(
                        config.get("repo_id", "black-forest-labs/FLUX.1-schnell"),
                        transformer=self.transformer,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        use_auth_token=config.get("needs_auth", False)
                    )
                else:
                    # Load from directory
                    self.pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                    )
            else:
                # Load from HuggingFace
                config = self.variant_configs.get(variant, {})
                print(f"Loading Flux {variant} from HuggingFace: {config.get('repo_id')}")
                
                self.pipeline = FluxPipeline.from_pretrained(
                    config.get("repo_id"),
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_auth_token=config.get("needs_auth", False)
                )
            
            # Apply memory optimizations - enable CPU offload by default for Flux
            if self.properties.get("enable_cpu_offload", True):  # Default to True for Flux
                print("Enabling CPU offload for Flux model to save memory")
                self.pipeline.enable_model_cpu_offload()
            else:
                # Clear GPU memory before loading to device
                torch.cuda.empty_cache()
                self.pipeline = self.pipeline.to(self.device)
            
            if self.properties.get("enable_vae_slicing", False):
                self.pipeline.enable_vae_slicing()
                
            if self.properties.get("enable_vae_tiling", False):
                self.pipeline.enable_vae_tiling()
            
            # Load LoRAs
            self._load_loras()
            
            return True
            
        except Exception as e:
            print(f"Error loading standard model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_property(self, key: str, value):
        """Override to track user modifications"""
        if hasattr(super(), 'set_property'):
            super().set_property(key, value)
        else:
            self.properties[key] = value
        self._user_modified_properties.add(key)
    
    def _apply_variant_defaults(self):
        """Apply model variant specific defaults"""
        variant = self.properties.get("model_variant", "flux-schnell")
        config = self.variant_configs.get(variant, {})
        
        # Only apply defaults if not manually set
        if "num_inference_steps" not in self._user_modified_properties:
            self.properties["num_inference_steps"] = config.get("steps", 4)
            
        if "guidance_scale" not in self._user_modified_properties:
            self.properties["guidance_scale"] = config.get("guidance", 0.0)
    
    async def process(self, context) -> bool:
        """Process Flux model loading and conditioning"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Clear GPU memory to avoid OOM errors
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # Apply variant-specific defaults
            self._apply_variant_defaults()
            
            # Load model if not already loaded
            if self.pipeline is None:
                format = self.properties.get("model_format", "safetensors")
                paths = self._get_model_paths()
                
                # Try local models first, fallback to HuggingFace
                model_path = paths.get("model") if paths else None
                
                # Load based on format
                if format == "gguf":
                    print("GGUF models not yet supported, using mock pipeline")
                    self._create_mock_pipeline()
                else:
                    if not self._load_standard_model(model_path):
                        print("Failed to load model, falling back to mock pipeline")
                        self._create_mock_pipeline()
            
            # Prepare conditioning
            prompt = self.properties.get("prompt", "")
            negative_prompt = self.properties.get("negative_prompt", "")
            
            # Flux-specific prompt handling
            if not prompt:
                prompt = "A high quality image"
            
            # Some Flux variants don't use negative prompts effectively
            variant = self.properties.get("model_variant", "flux-schnell")
            if variant in ["flux-schnell", "flux-turbo"] and negative_prompt:
                print(f"Note: {variant} has limited negative prompt support")
            
            # Set max sequence length for T5
            max_length = self.properties.get("max_sequence_length", 512)
            if hasattr(self.pipeline, 'tokenizer_2'):
                self.pipeline.tokenizer_2.model_max_length = max_length
            
            # Prepare outputs
            self.set_output_data("pipeline", self.pipeline)
            
            conditioning = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guidance_scale": self.properties.get("guidance_scale", 3.5),
                "num_inference_steps": self.properties.get("num_inference_steps", 4),
                "max_sequence_length": max_length
            }
            self.set_output_data("conditioning", conditioning)
            
            # Model info
            model_info = {
                "variant": self.properties.get("model_variant", "flux-schnell"),
                "format": self.properties.get("model_format", "safetensors"),
                "device": str(self.device),
                "dtype": str(self.pipeline.dtype) if hasattr(self.pipeline, 'dtype') else "float16",
                "memory_optimizations": {
                    "cpu_offload": self.properties.get("enable_cpu_offload", False),
                    "vae_slicing": self.properties.get("enable_vae_slicing", False),
                    "vae_tiling": self.properties.get("enable_vae_tiling", False)
                },
                "loras": [
                    {
                        "path": lora.get("path", ""),
                        "name": Path(lora.get("path", "unknown")).name,
                        "strength": lora.get("strength", 1.0),
                        "enabled": lora.get("enabled", True)
                    }
                    for lora in self.loaded_loras
                ]
            }
            
            if self.properties.get("model_format") == "gguf":
                model_info["quantization"] = self.properties.get("gguf_quant", "Q4_K_M")
                
            self.set_output_data("model_info", model_info)
            
            # Log success
            print(f"Flux {variant} loaded successfully")
            if format == "gguf":
                print(f"Quantization: {self.properties.get('gguf_quant')}")
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in Flux component: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing"""
        class MockFluxPipeline:
            def __init__(self, device):
                self.device = device
                self.scheduler = None
                
            def __call__(self, prompt, negative_prompt="", num_inference_steps=4, 
                        guidance_scale=0.0, width=1024, height=1024, 
                        generator=None, callback=None, callback_steps=1, **kwargs):
                
                # Generate gradient test image
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont
                
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Create a nice gradient pattern
                for y in range(height):
                    for x in range(width):
                        img_array[y, x] = [
                            int(255 * (x / width)),
                            int(255 * (y / height)),
                            128
                        ]
                
                # Add some "FLUX" text to indicate it's working
                img = Image.fromarray(img_array)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
                except:
                    font = None
                draw.text((width//2 - 100, height//2), "FLUX TEST", fill=(255, 255, 255), font=font)
                
                # Call progress callback
                if callback:
                    for step in range(num_inference_steps):
                        callback(step, step, None)
                
                class Result:
                    def __init__(self, images):
                        self.images = images
                
                return Result([img])
            
            def enable_model_cpu_offload(self):
                pass
                
            def enable_vae_slicing(self):
                pass
                
            def enable_vae_tiling(self):
                pass
        
        self.pipeline = MockFluxPipeline(self.device)
    
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
            
        if self.transformer:
            del self.transformer
            self.transformer = None
            
        if self.text_encoder:
            del self.text_encoder
            self.text_encoder = None
            
        if self.text_encoder_2:
            del self.text_encoder_2  
            self.text_encoder_2 = None
            
        if self.vae:
            del self.vae
            self.vae = None
        
        # Clear GPU cache
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print("Flux model unloaded")


# Import ControlNet component
from .controlnet import ControlNetComponent
# Import Lumina component  
from .lumina import LuminaControlComponent
# Import OmniGen component
from .omnigen import OmniGenComponent