"""
AnimateDiff Component for SD Forms
Animate still images using AnimateDiff motion modules
"""

from typing import List, Optional, Dict, Any, Union
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


class AnimateDiffComponent(VisualComponent):
    """AnimateDiff Motion Module Component for creating animated videos"""
    
    component_type = "animatediff"
    display_name = "AnimateDiff"
    category = "Video Models"
    icon = "🎞️"
    
    # Define input ports
    input_ports = [
        create_port("pipeline", PortType.PIPELINE, PortDirection.INPUT),
        create_port("conditioning", PortType.CONDITIONING, PortDirection.INPUT),
        create_port("init_image", PortType.IMAGE, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("video_frames", PortType.VIDEO, PortDirection.OUTPUT),
        create_port("latents", PortType.LATENT, PortDirection.OUTPUT, optional=True),
    ]
    
    # Define properties
    property_definitions = [
        # Motion Module Settings
        create_property_definition(
            "motion_module", "Motion Module", PropertyType.CHOICE, "mm_sd_v15_v2", "Motion",
            metadata={
                "choices": [
                    "mm_sd_v15_v2",
                    "mm_sd_v15_v3", 
                    "mm_sdxl_v10",
                    "temporaldiff-v1",
                    "custom"
                ]
            }
        ),
        create_property_definition(
            "motion_module_path", "Custom Motion Module Path", PropertyType.FILE_PATH, "", "Motion",
            metadata={"editor_type": "model_picker", "model_type": "animatediff"}
        ),
        
        # Animation Settings
        create_property_definition(
            "num_frames", "Number of Frames", PropertyType.INTEGER, 16, "Animation",
            metadata={"min": 8, "max": 32, "step": 8}
        ),
        create_property_definition(
            "fps", "FPS", PropertyType.INTEGER, 8, "Animation",
            metadata={"min": 1, "max": 30}
        ),
        create_property_definition(
            "motion_scale", "Motion Scale", PropertyType.FLOAT, 1.0, "Animation",
            metadata={"min": 0.0, "max": 3.0, "step": 0.1}
        ),
        
        # Generation Settings
        create_property_definition(
            "context_overlap", "Context Overlap", PropertyType.INTEGER, 4, "Generation",
            metadata={"min": 0, "max": 8}
        ),
        create_property_definition(
            "closed_loop", "Closed Loop", PropertyType.BOOLEAN, True, "Generation"
        ),
        create_property_definition(
            "temporal_consistency", "Temporal Consistency", PropertyType.FLOAT, 0.85, "Generation",
            metadata={"min": 0.0, "max": 1.0, "step": 0.05}
        ),
        
        # Prompt Settings
        create_property_definition(
            "prompt", "Prompt", PropertyType.TEXT, "a beautiful animated scene", "Prompts",
            metadata={"editor_type": "prompt"}
        ),
        create_property_definition(
            "negative_prompt", "Negative Prompt", PropertyType.TEXT, "", "Prompts",
            metadata={"editor_type": "prompt"}
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
        self.motion_module = None
        
        if self.scene:
            self.draw()
    
    async def process(self, context) -> bool:
        """Generate animation using AnimateDiff"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get inputs
            pipeline = self.get_input_data("pipeline")
            conditioning = self.get_input_data("conditioning")
            
            if not pipeline:
                raise ValueError("No pipeline provided")
            
            print(f"🎞️ AnimateDiff: Starting animation generation...")
            
            # Get properties
            num_frames = self.properties.get("num_frames", 16)
            fps = self.properties.get("fps", 8)
            motion_scale = self.properties.get("motion_scale", 1.0)
            motion_module = self.properties.get("motion_module", "mm_sd_v15_v2")
            
            # Mock implementation for now - in real implementation this would:
            # 1. Load the motion module
            # 2. Apply it to the pipeline
            # 3. Generate animated frames
            # 4. Export as video file
            
            print(f"🎬 Generating {num_frames} frames at {fps} FPS with motion scale {motion_scale}")
            
            # Generate test frames with motion effect
            frames = []
            for i in range(num_frames):
                # Create frames that simulate motion
                hue_shift = int((i / num_frames) * 360 * motion_scale) % 360
                brightness = 100 + int(50 * motion_scale * abs(0.5 - (i / num_frames)))
                
                # Create test frame with animated colors
                frame = Image.new('RGB', (512, 512), 
                                color=(
                                    (brightness + hue_shift) % 255,
                                    (150 + i * 5) % 255, 
                                    (200 - i * 3) % 255
                                ))
                frames.append(frame)
                
                # Progress callback
                if context.preview_callback and i % 4 == 0:
                    progress = int((i / num_frames) * 100)
                    context.preview_callback(frame, progress, i)
            
            # Convert frames to video data (mock)
            video_data = self._frames_to_video(frames, fps)
            
            # Set outputs
            self.set_output_data("video_frames", video_data)
            
            print(f"✅ AnimateDiff: Generated {len(frames)} frames successfully")
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"❌ Error in AnimateDiff: {e}")
            return False
    
    def _frames_to_video(self, frames: List[Image.Image], fps: int) -> bytes:
        """Convert frames to video bytes (mock implementation)"""
        import io
        
        # In real implementation, this would use ffmpeg or cv2 to create MP4
        # For now, return the first frame as PNG data
        buffer = io.BytesIO()
        if frames:
            frames[0].save(buffer, format='PNG')
        return buffer.getvalue()