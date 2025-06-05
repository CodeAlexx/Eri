"""
Visual canvas - Stub for backend compatibility
Note: Canvas functionality moved to Flutter web frontend
"""

from typing import List, Optional
# PyQt5 functionality removed - using web frontend now

from ..core import Component, Pipeline, Connection
from ..components import (
    BaseComponent, ConnectionInfo, 
    ModelComponent, SamplerComponent, VAEComponent, OutputComponent
)


class VisualCanvas:
    """Canvas stub for backend compatibility - UI moved to Flutter frontend"""
    
    def __init__(self):
        print("ℹ️ VisualCanvas: UI functionality moved to Flutter web frontend")
        
        # Maintain compatibility for backend components
        self.components = {}
        self.connections = []
        self.main_window = None
        
    def add_component(self, comp_type: str, position=None):
        """Add component - stub for compatibility"""
        print(f"ℹ️ Canvas: Component {comp_type} would be added at {position}")
        return None
        
    def create_connection(self, from_comp, to_comp):
        """Create connection - stub for compatibility"""
        print(f"ℹ️ Canvas: Connection would be created from {from_comp} to {to_comp}")
        return None
        
    def clear_canvas(self):
        """Clear canvas - stub for compatibility"""
        print("ℹ️ Canvas: Canvas would be cleared")
        
    def get_pipeline(self):
        """Get pipeline from canvas - stub for compatibility"""
        print("ℹ️ Canvas: Pipeline construction handled by FastAPI backend")
        return Pipeline()


# Keep import compatibility for backend
__all__ = ['VisualCanvas']