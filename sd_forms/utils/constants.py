"""
Constants and utility functions for SD Forms UI
"""

import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Try to import diffusers - show instructions if not installed or incompatible
try:
    # Test basic diffusers import first
    import diffusers
    from diffusers import StableDiffusionPipeline
    
    # Try additional schedulers (optional)
    try:
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            DDIMScheduler,
            PNDMScheduler
        )
    except ImportError as scheduler_err:
        print(f"⚠️ Some schedulers not available: {scheduler_err}")
    
    # Try transformers (optional for some use cases)
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as transformers_err:
        print(f"⚠️ Transformers components not available: {transformers_err}")
    
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers successfully imported")
    
except (ImportError, RuntimeError) as e:
    DIFFUSERS_AVAILABLE = False
    print(f"❌ Diffusers not available: {e}")
    print("Note: Will try direct imports in components")
    print("For full functionality, ensure compatible versions:")

# Configuration file for model paths
CONFIG_FILE = "sd_models_config.json"