"""
Model browser dialog - Stub for backend compatibility
Note: Model browser moved to Flutter web frontend
"""

from typing import Dict, List
from pathlib import Path
# PyQt5 functionality removed - using web frontend now

from ...core.model_config import ModelConfig


class ModelBrowser:
    """Model browser stub for backend compatibility"""
    
    def __init__(self, model_type="checkpoint", parent=None):
        print("ℹ️ ModelBrowser: UI functionality moved to Flutter web frontend")
        self.model_type = model_type
        self.parent = parent
        
    def show(self):
        """Show dialog - stub for compatibility"""
        print(f"ℹ️ ModelBrowser: Would show {self.model_type} browser")
        
    def exec_(self):
        """Execute dialog - stub for compatibility"""
        print(f"ℹ️ ModelBrowser: Would execute {self.model_type} browser")
        return False
        
    def get_selected_model(self):
        """Get selected model - stub for compatibility"""
        print("ℹ️ ModelBrowser: Would return selected model")
        return None


# Keep import compatibility for backend
__all__ = ['ModelBrowser']