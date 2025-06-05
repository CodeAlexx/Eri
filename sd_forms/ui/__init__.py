"""
UI classes - Stubs for backend compatibility
Note: UI functionality moved to Flutter web frontend
"""

from .canvas import VisualCanvas
from .component_toolbar import ComponentToolbar
from .properties_panel import PropertiesPanel
from .property_editors import (
    ModelPickerEditor, LoRACollectionEditor, 
    PromptEditor, FloatSliderEditor, register_property_editors
)

__all__ = [
    'VisualCanvas', 'ComponentToolbar', 'PropertiesPanel',
    'ModelPickerEditor', 'LoRACollectionEditor',
    'PromptEditor', 'FloatSliderEditor', 'register_property_editors'
]