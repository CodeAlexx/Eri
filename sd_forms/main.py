"""
SD Forms - Component registration and utilities
Backend component system without PyQt5 dependencies
"""

import sys
import json

# Core system imports (no PyQt5 dependencies)
from .core import component_registry, Connection
from .components import ModelComponent, SamplerComponent, VAEComponent, OutputComponent, ImageComponent, MediaDisplayComponent, SDXLModelComponent, FluxModelComponent, WanVACEComponent, UpscalerComponent, SimpleTunerTrainingComponent, ControlNetComponent, LuminaControlComponent, OmniGenComponent
from .components.animatediff import AnimateDiffComponent
from .components.ltx_video import LTXVideoComponent
from .components.hidream import HiDreamI1Component
from .components.sd15_simple import SD15Component
from .components.sd35_simple import SD35Component
from .components.chroma_simple import ChromaComponent
from .utils import DIFFUSERS_AVAILABLE


def register_property_editors():
    """Register custom property editors - stub for backend compatibility"""
    # Property editors are now handled by the Flutter frontend
    # This function exists for backward compatibility with the FastAPI backend
    print("✅ Property editors registration (web frontend handles UI)")
    pass


def register_components():
    """Register components with the component registry"""
    print("🔧 Registering core components...")
    
    # component_registry.register(ModelComponent)  # Generic model removed - use specific model types
    component_registry.register(SDXLModelComponent)
    component_registry.register(FluxModelComponent)
    component_registry.register(HiDreamI1Component)
    component_registry.register(SamplerComponent) 
    component_registry.register(VAEComponent)
    component_registry.register(OutputComponent)
    component_registry.register(ImageComponent)
    component_registry.register(MediaDisplayComponent)
    component_registry.register(WanVACEComponent)
    component_registry.register(UpscalerComponent)
    component_registry.register(SimpleTunerTrainingComponent)
    component_registry.register(ControlNetComponent)
    component_registry.register(LuminaControlComponent)
    component_registry.register(OmniGenComponent)
    component_registry.register(AnimateDiffComponent)
    component_registry.register(LTXVideoComponent)
    # Register the REAL working model components  
    component_registry.register(SD15Component)  # Fixed and re-enabled
    component_registry.register(SD35Component)  # Re-enabled for component palette
    component_registry.register(ChromaComponent)  # Re-enabled for component palette
    
    print("✅ Real model components registered")
    
    # Try to import and register text encoders (optional)
    try:
        from .components.text_encoders import CLIPComponent, T5XXLComponent
        component_registry.register(CLIPComponent)
        component_registry.register(T5XXLComponent)
        print("✅ Text encoder components registered")
    except ImportError:
        print("ℹ️ Text encoder components not available (optional)")
    
    print("✅ All components registered successfully")


def check_dependencies():
    """Check if required dependencies are available"""
    if not DIFFUSERS_AVAILABLE:
        print("⚠️ WARNING: Diffusers library not found!")
        print("Please install: pip install diffusers>=0.30.0 transformers>=4.38.0")
        return False
    
    print("✅ Dependencies check passed")
    return True


def initialize_backend():
    """Initialize the backend component system"""
    print("🚀 Initializing SD Forms backend...")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Register property editors (web frontend compatibility)
    register_property_editors()
    
    # Register components
    register_components()
    
    print("✅ Backend initialization complete")
    return True


# Keep the main function for backward compatibility, but make it backend-only
def main():
    """Backend initialization entry point (no GUI)"""
    print("🔧 SD Forms - Backend Mode (No GUI)")
    print("For web interface, use: cd backend && python server.py")
    
    success = initialize_backend()
    
    if success:
        print("✅ Backend ready - use FastAPI server for web interface")
        return 0
    else:
        print("❌ Backend initialization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())