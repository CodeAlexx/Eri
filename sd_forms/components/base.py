"""
Base component classes for visual components
"""

from typing import Any, Optional
# PyQt5 imports removed - using web-based frontend now
from dataclasses import dataclass

from ..core import Component


@dataclass
class ConnectionInfo:
    """Info about connections between components"""
    from_component: 'BaseComponent'
    to_component: 'BaseComponent'
    connection_id: str  # Unique identifier for web frontend


class BaseComponent:
    """Base class for all pipeline components"""
    def __init__(self, comp_type: str, position: tuple, scene=None):
        self.type = comp_type
        self.position = position
        # Scene parameter kept for compatibility but not used
        self.scene = scene  # Legacy parameter - web frontend handles rendering
        self.id = f"{comp_type}_{id(self)}"
        self.selected = False
        self.properties = {}
        self.group_id = None  # Web frontend group identifier
        self.output_data = None
        
        # Visual settings
        self.colors = {
            "Model": "#3498db",
            "Sampler": "#e74c3c",
            "VAE": "#2ecc71",
            "Output": "#34495e"
        }
        
        self.icons = {
            "Model": "📦",
            "Sampler": "🎲",
            "VAE": "🖼️",
            "Output": "💾"
        }
    
    def draw(self):
        """Update component visual state - actual drawing handled by web frontend"""
        print(f"🔧 DEBUG: BaseComponent.draw() called for {self.type}")
        
        # No Qt scene operations - web frontend handles all visual rendering
        # Just update internal state for serialization to frontend
        
        x, y = self.position[0], self.position[1]
        color = self.colors.get(self.type, "#95a5a6")
        
        # Update visual state for web frontend serialization
        self.visual_bounds = {
            'x': x, 'y': y, 'width': 150, 'height': 70,
            'color': color, 'selected': self.selected
        }
        
        # Status indicator state (no Qt graphics)
        self.status_color = "#95a5a6"  # Gray = idle
        
        # Connection points for serialization
        self.connection_points = {
            'input': {'x': x, 'y': y + 35},
            'output': {'x': x + 150, 'y': y + 35}
        }
        
        print(f"🔧 DEBUG: BaseComponent.draw() completed for {self.type} - state updated for web frontend")
    
    def set_status(self, status: str):
        """Update status indicator color"""
        colors = {
            "idle": "#95a5a6",      # Gray
            "processing": "#f39c12", # Orange
            "complete": "#27ae60",   # Green
            "error": "#e74c3c"       # Red
        }
        self.status_color = colors.get(status, "#95a5a6")
        # Status change will be sent to web frontend via API
    
    def get_output_point(self) -> tuple:
        return (self.position[0] + 150, self.position[1] + 35)
    
    def get_input_point(self) -> tuple:
        return (self.position[0], self.position[1] + 35)
    
    def process(self, input_data: Any) -> Any:
        """Override in subclasses to implement processing"""
        raise NotImplementedError
    
    def set_position(self, x: float, y: float):
        """Update component position and recalculate connection points"""
        self.position = (x, y)
        # Update connection points
        self.connection_points = {
            'input': {'x': x, 'y': y + 35},
            'output': {'x': x + 150, 'y': y + 35}
        }
        # Update visual bounds
        if hasattr(self, 'visual_bounds'):
            self.visual_bounds['x'] = x
            self.visual_bounds['y'] = y
    
    def set_selection(self, selected: bool):
        """Update selection state"""
        self.selected = selected
        if hasattr(self, 'visual_bounds'):
            self.visual_bounds['selected'] = selected
    
    def to_dict(self) -> dict:
        """Serialize component state for web frontend"""
        return {
            'id': self.id,
            'type': self.type,
            'position': {'x': self.position[0], 'y': self.position[1]},
            'selected': self.selected,
            'properties': self.properties,
            'visual_bounds': getattr(self, 'visual_bounds', {}),
            'status_color': getattr(self, 'status_color', '#95a5a6'),
            'connection_points': getattr(self, 'connection_points', {}),
            'icon': self.icons.get(self.type, "📄"),
            'color': self.colors.get(self.type, "#95a5a6")
        }


class VisualComponent(Component, BaseComponent):
    """Hybrid component that supports both new system and visual display"""
    
    def __init__(self, comp_type: str, position, scene=None, component_id: Optional[str] = None):
        print(f"🔧 DEBUG: VisualComponent.__init__ called with comp_type={comp_type}, position={position}")
        # Initialize Component system
        Component.__init__(self, component_id)
        print(f"🔧 DEBUG: VisualComponent Component.__init__ completed")
        
        # Save the correct ID before BaseComponent overwrites it
        correct_id = self.id
        
        # Initialize BaseComponent visuals  
        BaseComponent.__init__(self, comp_type, position, scene)
        print(f"🔧 DEBUG: VisualComponent BaseComponent.__init__ completed")
        
        # Restore the correct ID
        self.id = correct_id
        
        # Sync the type attribute
        self.type = self.component_type
        
        # Ensure properties are initialized from the subclass property definitions
        self._init_properties()
        
        print(f"🔧 DEBUG: VisualComponent.__init__ completed, type={self.type}, id={self.id}")
    
    def to_dict(self) -> dict:
        """Serialize component state for web frontend with core component data"""
        base_dict = BaseComponent.to_dict(self)
        # Add core component system data
        base_dict.update({
            'component_id': self.component_id,
            'input_ports': [{
                'name': port.name,
                'data_type': port.data_type.value if hasattr(port.data_type, 'value') else str(port.data_type),
                'required': port.required
            } for port in self.input_ports],
            'output_ports': [{
                'name': port.name, 
                'data_type': port.data_type.value if hasattr(port.data_type, 'value') else str(port.data_type)
            } for port in self.output_ports],
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status)
        })
        return base_dict