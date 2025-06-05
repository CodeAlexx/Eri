"""
Property Editors - Stub for backend compatibility
Note: Property editors moved to Flutter web frontend
"""

import re
from typing import Any, Dict, List, Callable, Optional
# PyQt5 functionality removed - using web frontend now

from ..core.component_system import PropertyEditor, PropertyRegistry


class ModelPickerEditor(PropertyEditor):
    """Model picker editor stub for backend compatibility"""
    
    def create_widget(self, parent, value: Any, metadata: Dict[str, Any], on_change: Callable[[Any], None]):
        print("ℹ️ ModelPickerEditor: UI functionality moved to Flutter web frontend")
        return None
        
    def get_value(self):
        return None
        
    def set_value(self, value):
        pass


class LoRACollectionEditor(PropertyEditor):
    """LoRA collection editor stub for backend compatibility"""
    
    def create_widget(self, parent, value: Any, metadata: Dict[str, Any], on_change: Callable[[Any], None]):
        print("ℹ️ LoRACollectionEditor: UI functionality moved to Flutter web frontend")
        return None
        
    def get_value(self):
        return []
        
    def set_value(self, value):
        pass


class PromptEditor(PropertyEditor):
    """Prompt editor stub for backend compatibility"""
    
    def create_widget(self, parent, value: Any, metadata: Dict[str, Any], on_change: Callable[[Any], None]):
        print("ℹ️ PromptEditor: UI functionality moved to Flutter web frontend")
        return None
        
    def get_value(self):
        return ""
        
    def set_value(self, value):
        pass


class FloatSliderEditor(PropertyEditor):
    """Float slider editor stub for backend compatibility"""
    
    def create_widget(self, parent, value: Any, metadata: Dict[str, Any], on_change: Callable[[Any], None]):
        print("ℹ️ FloatSliderEditor: UI functionality moved to Flutter web frontend")
        return None
        
    def get_value(self):
        return 0.0
        
    def set_value(self, value):
        pass


def register_property_editors():
    """Register property editors - stub for backend compatibility"""
    print("ℹ️ Property editors: Registration handled by Flutter web frontend")
    
    # Register stubs for backend compatibility
    try:
        PropertyRegistry.register_editor("model_picker", ModelPickerEditor)
        PropertyRegistry.register_editor("lora_collection", LoRACollectionEditor)
        PropertyRegistry.register_editor("prompt", PromptEditor)
        PropertyRegistry.register_editor("float_slider", FloatSliderEditor)
        print("✅ Property editor stubs registered for backend compatibility")
    except Exception as e:
        print(f"ℹ️ Property editor registration: {e}")


# Keep import compatibility for backend
__all__ = [
    'ModelPickerEditor', 'LoRACollectionEditor', 'PromptEditor', 'FloatSliderEditor',
    'register_property_editors'
]