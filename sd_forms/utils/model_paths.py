"""
Model Path Configuration Loader
Loads model directories and paths from JSON configuration
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import glob

class ModelPathConfig:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize model path configuration"""
        if config_path is None:
            # Look for config in the project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "model_paths_config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    print(f"✅ Loaded model paths config from: {self.config_path}")
                    return config
            else:
                print(f"⚠️ Model paths config not found at: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading model paths config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration as fallback"""
        return {
            "model_directories": {
                "flux": ["/home/alex/SwarmUI/Models/diffusion_models"],
                "sd15": ["/home/alex/SwarmUI/Models/diffusion_models"],
                "sdxl": ["/home/alex/SwarmUI/Models/diffusion_models"],
                "sd35": ["/home/alex/SwarmUI/Models/diffusion_models"],
                "chroma": ["/home/alex/SwarmUI/Models/diffusion_models"]
            },
            "fallback_directories": ["/home/alex/SwarmUI/Models"],
            "file_extensions": {
                "models": [".safetensors", ".ckpt", ".pt", ".bin"],
                "gguf": [".gguf"]
            }
        }
    
    def get_directories_for_model_type(self, model_type: str) -> List[Path]:
        """Get directories for a specific model type"""
        directories = self.config.get("model_directories", {}).get(model_type, [])
        
        # Add fallback directories
        fallback_dirs = self.config.get("fallback_directories", [])
        directories.extend(fallback_dirs)
        
        # Convert to Path objects and expand home directory
        path_objects = []
        for dir_str in directories:
            path = Path(dir_str).expanduser()
            if path.exists():
                path_objects.append(path)
            else:
                print(f"⚠️ Directory not found: {path}")
        
        return path_objects
    
    def find_model_file(self, model_type: str, filename_patterns: List[str]) -> Optional[str]:
        """Find a model file matching the given patterns"""
        directories = self.get_directories_for_model_type(model_type)
        extensions = self.config.get("file_extensions", {}).get("models", [".safetensors"])
        
        for directory in directories:
            for pattern in filename_patterns:
                # Try exact filename match first
                full_path = directory / pattern
                if full_path.exists():
                    print(f"✅ Found model: {full_path}")
                    return str(full_path)
                
                # Try glob pattern matching
                for ext in extensions:
                    if not pattern.endswith(ext):
                        glob_pattern = str(directory / f"{pattern}*{ext}")
                        matches = glob.glob(glob_pattern)
                        if matches:
                            print(f"✅ Found model via pattern: {matches[0]}")
                            return matches[0]
        
        print(f"❌ No model found for patterns: {filename_patterns}")
        return None
    
    def get_model_patterns(self, model_type: str, variant: str) -> List[str]:
        """Get filename patterns for a specific model variant"""
        patterns = self.config.get("model_filename_patterns", {})
        model_patterns = patterns.get(model_type, {})
        variant_patterns = model_patterns.get(variant, [])
        
        if not variant_patterns:
            print(f"⚠️ No patterns found for {model_type}:{variant}")
            # Fallback to generic patterns
            return [f"{variant}.safetensors", f"{variant}_*.safetensors"]
        
        return variant_patterns
    
    def scan_available_models(self, model_type: str) -> Dict[str, List[str]]:
        """Scan directories and return available models by variant"""
        directories = self.get_directories_for_model_type(model_type)
        extensions = self.config.get("file_extensions", {}).get("models", [".safetensors"])
        found_models = {}
        
        for directory in directories:
            if not directory.exists():
                continue
                
            for ext in extensions:
                pattern = f"*{ext}"
                model_files = list(directory.glob(pattern))
                
                for model_file in model_files:
                    filename = model_file.name
                    
                    # Try to categorize by variant
                    variant = self._categorize_model_file(model_type, filename)
                    if variant not in found_models:
                        found_models[variant] = []
                    found_models[variant].append(str(model_file))
        
        return found_models
    
    def _categorize_model_file(self, model_type: str, filename: str) -> str:
        """Categorize a model file into a variant based on filename"""
        filename_lower = filename.lower()
        
        # Check configured patterns first
        patterns = self.config.get("model_filename_patterns", {}).get(model_type, {})
        for variant, variant_patterns in patterns.items():
            for pattern in variant_patterns:
                pattern_lower = pattern.lower().replace("*", "")
                if pattern_lower in filename_lower:
                    return variant
        
        # Fallback categorization
        if model_type == "flux":
            if "schnell" in filename_lower:
                return "flux-schnell"
            elif "dev" in filename_lower:
                return "flux-dev"
            elif "turbo" in filename_lower:
                return "flux-turbo"
            elif "lite" in filename_lower:
                return "flux-lite"
            elif "sigma" in filename_lower:
                return "flux-sigma"
        elif model_type == "sd15":
            if "dreamshaper" in filename_lower:
                return "dreamshaper"
            elif "realistic" in filename_lower:
                return "realistic-vision"
        
        return "custom"

# Global instance
_model_config = None

def get_model_config() -> ModelPathConfig:
    """Get the global model configuration instance"""
    global _model_config
    if _model_config is None:
        _model_config = ModelPathConfig()
    return _model_config

def find_model_for_variant(model_type: str, variant: str) -> Optional[str]:
    """Convenience function to find a model file for a specific variant"""
    config = get_model_config()
    patterns = config.get_model_patterns(model_type, variant)
    return config.find_model_file(model_type, patterns)

def get_available_models(model_type: str) -> Dict[str, List[str]]:
    """Convenience function to get available models for a type"""
    config = get_model_config()
    return config.scan_available_models(model_type)