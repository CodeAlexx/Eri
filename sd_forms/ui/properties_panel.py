"""
Properties panel - Stub for backend compatibility
Note: Properties panel moved to Flutter web frontend
"""

from typing import Any
from pathlib import Path
# PyQt5 functionality removed - using web frontend now

from ..core import PropertyDefinition, PropertyType


class PropertiesPanel:
    """Properties panel stub for backend compatibility"""
    
    def __init__(self):
        print("ℹ️ PropertiesPanel: UI functionality moved to Flutter web frontend")
        self.current_component = None
        
    def set_component(self, component):
        """Set current component - stub for compatibility"""
        print(f"ℹ️ Properties: Would show properties for {component}")
        self.current_component = component
        
    def update_properties(self, properties):
        """Update properties display - stub for compatibility"""
        print(f"ℹ️ Properties: Would update {len(properties)} properties")
        
    def clear_properties(self):
        """Clear properties display - stub for compatibility"""
        print("ℹ️ Properties: Would clear property display")


# Keep import compatibility for backend
__all__ = ['PropertiesPanel']