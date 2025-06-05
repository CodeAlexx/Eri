"""
LTX Video Model Component for SD Forms
Supports LTX-Video for high-quality video generation
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from PIL import Image

from .base import VisualComponent
from ..core import (
    create_port,
    create_property_definition,
    PropertyType,
    ComponentStatus,
    PortType,
    PortDirection
)


class LTXVideoComponent(VisualComponent):
    """LTX Video Generation Model Component"""
    
    component_type = "ltx_video"
    display_name = "LTX Video"
    category = "Video Models"
    icon = "🎥"
    
    # Define input ports
    input_ports = [
        create_port("prompt", PortType.TEXT, PortDirection.INPUT, optional=True),
        create_port("negative_prompt", PortType.TEXT, PortDirection.INPUT, optional=True),
        create_port("reference_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("video", PortType.VIDEO, PortDirection.OUTPUT),
        create_port("metadata", PortType.ANY, PortDirection.OUTPUT, optional=True),
    ]
    
    # Define properties
    property_definitions = [
        # Model Settings
        create_property_definition(
            "model_type", "Model Type", PropertyType.CHOICE, "huggingface", "Model",
            metadata={"choices": ["huggingface", "local", "preset"]}
        ),
        create_property_definition(
            "model_id", "Model ID/Path", PropertyType.STRING, "Lightricks/LTX-Video", "Model"
        ),
        
        # Generation Settings
        create_property_definition(
            "prompt", "Video Prompt", PropertyType.TEXT, "a beautiful nature scene", "Generation",
            metadata={"editor_type": "prompt"}
        ),
        create_property_definition(
            "negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Generation",
            metadata={"editor_type": "prompt"}
        ),
        create_property_definition(
            "num_frames", "Number of Frames", PropertyType.INTEGER, 121, "Generation",
            metadata={"min": 1, "max": 257, "step": 8}
        ),
        create_property_definition(
            "fps", "Frames Per Second", PropertyType.INTEGER, 24, "Generation",
            metadata={"min": 8, "max": 60}
        ),
        create_property_definition(
            "width", "Width", PropertyType.INTEGER, 768, "Generation",
            metadata={"min": 256, "max": 1920, "step": 64}
        ),
        create_property_definition(
            "height", "Height", PropertyType.INTEGER, 512, "Generation",
            metadata={"min": 256, "max": 1080, "step": 64}
        ),
        
        # Advanced Settings
        create_property_definition(
            "guidance_scale", "Guidance Scale", PropertyType.FLOAT, 7.5, "Advanced",
            metadata={"min": 1.0, "max": 20.0, "step": 0.5}
        ),
        create_property_definition(
            "num_inference_steps", "Inference Steps", PropertyType.INTEGER, 50, "Advanced",
            metadata={"min": 1, "max": 100}
        ),
        create_property_definition(
            "motion_bucket_id", "Motion Bucket", PropertyType.INTEGER, 127, "Advanced",
            metadata={"min": 1, "max": 255}
        ),
        create_property_definition(
            "seed", "Seed", PropertyType.INTEGER, -1, "Advanced"
        ),
        
        # Memory Settings
        create_property_definition(
            "enable_model_cpu_offload", "CPU Offload", PropertyType.BOOLEAN, True, "Memory"
        ),
        create_property_definition(
            "enable_vae_tiling", "VAE Tiling", PropertyType.BOOLEAN, True, "Memory"
        ),
        
        # Output Settings
        create_property_definition(
            "output_format", "Output Format", PropertyType.CHOICE, "frames", "Output",
            metadata={"choices": ["frames", "mp4", "gif", "webm"]}
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
        self.pipe = None
        
        if self.scene:
            self.draw()
    
    def _create_mock_ltx_pipeline(self):
        """Create mock pipeline for testing"""
        class MockLTXPipeline:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, prompt, num_frames=121, height=512, width=768,
                        num_inference_steps=50, guidance_scale=7.5,
                        negative_prompt="", image=None, generator=None, 
                        callback=None, callback_steps=1, **kwargs):
                # Generate mock frames with moving pattern
                import numpy as np
                frames = []
                
                for i in range(num_frames):
                    # Create test pattern with motion
                    frame_array = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Moving gradient effect
                    t = i / num_frames
                    x_offset = int(width * 0.3 * np.sin(t * np.pi * 2))
                    y_offset = int(height * 0.3 * np.cos(t * np.pi * 2))
                    
                    # Create animated pattern
                    for y in range(height):
                        for x in range(width):
                            r = int(128 + 127 * np.sin((x + x_offset) * 0.01 + t * np.pi * 4))
                            g = int(128 + 127 * np.cos((y + y_offset) * 0.01 + t * np.pi * 3))
                            b = int(128 + 127 * np.sin((x + y + t * 100) * 0.005))
                            
                            frame_array[y, x] = [r, g, b]
                    
                    frame = Image.fromarray(frame_array)
                    frames.append(frame)
                    
                    # Progress callback
                    if callback and i % callback_steps == 0:
                        callback(i, None, None)
                
                class Result:
                    def __init__(self, frames):
                        self.frames = frames
                
                return Result(frames)
        
        return MockLTXPipeline("cpu")
    
    async def process(self, context) -> bool:
        """Generate video using LTX model"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Load model if needed (mock for now)
            if self.pipe is None:
                print("🎥 Loading LTX Video model...")
                self.pipe = self._create_mock_ltx_pipeline()
            
            # Get inputs
            prompt = self.get_input_data("prompt") or self.properties.get("prompt", "")
            negative_prompt = self.get_input_data("negative_prompt") or self.properties.get("negative_prompt", "")
            reference_image = self.get_input_data("reference_image")
            
            if not prompt and not reference_image:
                raise ValueError("Either prompt or reference image required")
            
            # Setup generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": self.properties.get("num_frames", 121),
                "height": self.properties.get("height", 512),
                "width": self.properties.get("width", 768),
                "num_inference_steps": self.properties.get("num_inference_steps", 50),
                "guidance_scale": self.properties.get("guidance_scale", 7.5),
            }
            
            # Add reference image for img2vid
            if reference_image:
                gen_kwargs["image"] = reference_image
            
            # Setup seed
            seed = self.properties.get("seed", -1)
            if seed != -1:
                import torch
                gen_kwargs["generator"] = torch.Generator().manual_seed(seed)
            
            # Progress callback
            if hasattr(context, 'preview_callback'):
                def callback(step, timestep, latents):
                    progress = int((step / gen_kwargs["num_inference_steps"]) * 100)
                    if context.preview_callback:
                        # Create a preview frame
                        preview_frame = Image.new('RGB', (256, 256), 
                                                color=(100 + step * 3, 150, 200 - step))
                        context.preview_callback(preview_frame, progress, step)
                
                gen_kwargs["callback"] = callback
                gen_kwargs["callback_steps"] = 5
            
            print(f"🎬 Generating LTX video: {gen_kwargs['num_frames']} frames at {gen_kwargs['width']}x{gen_kwargs['height']}")
            
            # Generate video
            output = self.pipe(**gen_kwargs)
            
            # Extract frames
            frames = output.frames if hasattr(output, 'frames') else output
            
            print(f"✅ Generated {len(frames)} frames")
            
            # Convert to video data
            video_data = self._frames_to_video(frames)
            
            # Set outputs
            self.set_output_data("video", video_data)
            
            # Metadata
            metadata = {
                "model": "LTX-Video",
                "num_frames": len(frames),
                "fps": self.properties.get("fps", 24),
                "resolution": f"{frames[0].width}x{frames[0].height}",
                "seed": seed,
                "guidance_scale": gen_kwargs["guidance_scale"],
                "steps": gen_kwargs["num_inference_steps"]
            }
            self.set_output_data("metadata", metadata)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"❌ Error in LTX Video generation: {e}")
            return False
    
    def _frames_to_video(self, frames: List[Image.Image]) -> bytes:
        """Convert frames to video bytes"""
        import io
        
        # For now, return the first frame as video data
        # In real implementation, this would create an MP4/GIF
        buffer = io.BytesIO()
        if frames:
            # Save as GIF for demo
            frames[0].save(
                buffer,
                format='GIF',
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=1000 // self.properties.get("fps", 24),
                loop=0
            )
        
        return buffer.getvalue()