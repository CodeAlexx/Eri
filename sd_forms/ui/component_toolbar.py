"""
Component toolbar - Stub for backend compatibility
Note: Component toolbar moved to Flutter web frontend
"""

# PyQt5 functionality removed - using web frontend now


class ComponentToolbar:
    """Component toolbar stub for backend compatibility"""
    
    def __init__(self, on_component_select=None):
        print("ℹ️ ComponentToolbar: UI functionality moved to Flutter web frontend")
        self.on_component_select = on_component_select
        
    def add_component_button(self, name, callback):
        """Add component button - stub for compatibility"""
        print(f"ℹ️ Toolbar: Component button {name} would be added")
        
    def update_components(self, components):
        """Update available components - stub for compatibility"""
        print(f"ℹ️ Toolbar: Would update with {len(components)} components")


# Keep import compatibility for backend
__all__ = ['ComponentToolbar']