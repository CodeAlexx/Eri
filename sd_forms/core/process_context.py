"""
Process context for component execution with model caching and shared resources
"""

from typing import Any, Dict, Optional
from ..utils.constants import DIFFUSERS_AVAILABLE


class ProcessContext:
    """Context manager for component execution with model caching and shared resources"""
    
    def __init__(self, pipeline: 'Pipeline', cache: Optional[Dict[str, Any]] = None):
        self.pipeline = pipeline
        self.cache = cache or {}
        self.model_cache = {}  # Cache for loaded models to avoid reloading
        self.shared_data = {}  # Shared data between components
        self.current_step = 0
        self.total_steps = 0
        self.preview_callback = None  # Callback for real-time previews
        
    async def load_model(self, path: str, model_type: str = "checkpoint") -> Any:
        """Load model with caching logic to avoid reloading"""
        cache_key = f"{model_type}:{path}"
        
        if cache_key in self.model_cache:
            print(f"Using cached model: {path}")
            return self.model_cache[cache_key]
        
        print(f"Loading model: {path}")
        
        try:
            if model_type == "checkpoint":
                # Load main checkpoint
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
                import torch
                
                # Determine device
                from ..utils.constants import DEVICE
                device = DEVICE
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                # Auto-detect model type
                if "xl" in path.lower():
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        path,
                        torch_dtype=dtype,
                        use_safetensors=path.endswith('.safetensors')
                    )
                else:
                    pipeline = StableDiffusionPipeline.from_single_file(
                        path,
                        torch_dtype=dtype,
                        use_safetensors=path.endswith('.safetensors')
                    )
                
                pipeline = pipeline.to(device)
                self.model_cache[cache_key] = pipeline
                return pipeline
                
            elif model_type == "vae":
                # Load VAE
                from diffusers import AutoencoderKL
                import torch
                
                from ..utils.constants import DEVICE
                device = DEVICE
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                vae = AutoencoderKL.from_single_file(
                    path,
                    torch_dtype=dtype
                )
                vae = vae.to(device)
                self.model_cache[cache_key] = vae
                return vae
                
            elif model_type == "lora":
                # LoRA loading will be handled by apply_lora
                return path
                
            elif model_type == "embedding":
                # Embedding loading will be handled separately
                return path
                
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            return None
    
    async def apply_lora(self, pipeline: Any, path: str, strength: float = 1.0) -> bool:
        """Apply LoRA to pipeline with specified strength"""
        try:
            if hasattr(pipeline, 'load_lora_weights'):
                from pathlib import Path
                
                # Load LoRA weights
                pipeline.load_lora_weights(path)
                
                # Set LoRA scale
                lora_name = Path(path).stem
                pipeline.set_adapters([lora_name], adapter_weights=[strength])
                
                print(f"Applied LoRA: {lora_name} (strength: {strength})")
                return True
            else:
                print(f"Pipeline does not support LoRA: {type(pipeline)}")
                return False
                
        except Exception as e:
            print(f"Error applying LoRA {path}: {e}")
            return False
    
    async def encode_prompt(self, pipeline: Any, prompt: str, negative_prompt: str = "") -> Dict[str, Any]:
        """Encode text prompts into embeddings"""
        try:
            if hasattr(pipeline, 'encode_prompt'):
                # Use pipeline's built-in prompt encoding
                result = pipeline.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    device=pipeline.device if hasattr(pipeline, 'device') else 'cpu'
                )
                return {
                    "prompt_embeds": result[0] if isinstance(result, tuple) else result,
                    "negative_prompt_embeds": result[1] if isinstance(result, tuple) and len(result) > 1 else None
                }
            else:
                # Fallback: use tokenizer and text encoder directly
                if hasattr(pipeline, 'tokenizer') and hasattr(pipeline, 'text_encoder'):
                    import torch
                    
                    # Tokenize prompts
                    text_inputs = pipeline.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    
                    negative_inputs = pipeline.tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    
                    # Encode to embeddings
                    with torch.no_grad():
                        prompt_embeds = pipeline.text_encoder(text_inputs.input_ids.to(pipeline.device))[0]
                        negative_embeds = pipeline.text_encoder(negative_inputs.input_ids.to(pipeline.device))[0]
                    
                    return {
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds": negative_embeds
                    }
                else:
                    print("Pipeline does not have prompt encoding capabilities")
                    return {"error": "No prompt encoding available"}
                    
        except Exception as e:
            print(f"Error encoding prompts: {e}")
            return {"error": str(e)}
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data by key"""
        return self.shared_data.get(key, default)
    
    def set_shared_data(self, key: str, value: Any):
        """Set shared data by key"""
        self.shared_data[key] = value
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        self.model_cache.clear()
        self.shared_data.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            "cached_models": list(self.model_cache.keys()),
            "shared_data_keys": list(self.shared_data.keys()),
            "cache_size": len(self.model_cache)
        }
    
    # Dictionary-like interface for shared_data
    def __getitem__(self, key: str) -> Any:
        """Get shared data like a dictionary"""
        return self.shared_data[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set shared data like a dictionary"""
        self.shared_data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in shared data"""
        return key in self.shared_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get shared data with default value"""
        return self.shared_data.get(key, default)