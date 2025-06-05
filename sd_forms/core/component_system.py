"""
Component System for Forms-Based SD Generator
Base classes for a flexible, extensible component architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque


class PropertyType(Enum):
    """Property types for component properties"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    FILE_PATH = "file_path"
    DIRECTORY = "directory"
    CHOICE = "choice"
    SLIDER = "slider"
    COLOR = "color"
    TEXT = "text"


class PortDirection(Enum):
    """Port direction types"""
    INPUT = "input"
    OUTPUT = "output"


class PortType(Enum):
    """Port data types"""
    ANY = "any"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    TENSOR = "tensor"
    MODEL = "model"
    PIPELINE = "pipeline"
    CONDITIONING = "conditioning"
    LATENT = "latent"
    VAE = "vae"
    CLIP = "clip"


class ComponentStatus(Enum):
    """Component execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PropertyDefinition:
    """Definition of a component property"""
    key: str
    display_name: str
    type: PropertyType
    category: str = "General"
    default: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validator: Optional[Callable[[Any], bool]] = None
    depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate property definition"""
        if not self.key:
            raise ValueError("Property key cannot be empty")
        if not self.display_name:
            raise ValueError("Property display_name cannot be empty")


@dataclass
class Port:
    """Component port definition"""
    name: str
    type: PortType
    direction: PortDirection
    multiple: bool = False
    optional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate port definition"""
        if not self.name:
            raise ValueError("Port name cannot be empty")


@dataclass
class Connection:
    """Connection between two component ports"""
    id: str
    from_component: str  # Component ID
    from_port: str
    to_component: str    # Component ID
    to_port: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def validate(self, pipeline: 'Pipeline') -> bool:
        """Validate connection port type compatibility"""
        try:
            from_comp = pipeline.components[self.from_component]
            to_comp = pipeline.components[self.to_component]
            
            # Find ports
            from_port = None
            to_port = None
            
            for port in from_comp.output_ports:
                if port.name == self.from_port:
                    from_port = port
                    break
            
            for port in to_comp.input_ports:
                if port.name == self.to_port:
                    to_port = port
                    break
            
            if not from_port or not to_port:
                return False
            
            # Check type compatibility
            if from_port.type == PortType.ANY or to_port.type == PortType.ANY:
                return True
            
            return from_port.type == to_port.type
            
        except (KeyError, AttributeError):
            return False


class Component(ABC):
    """Abstract base class for all components"""
    
    # Class attributes - to be overridden by subclasses
    component_type: str = "base"
    display_name: str = "Base Component"
    category: str = "Base"
    icon: str = "📦"
    input_ports: List[Port] = []
    output_ports: List[Port] = []
    property_definitions: List[PropertyDefinition] = []
    
    def __init__(self, component_id: Optional[str] = None):
        """Initialize component instance"""
        self.id = component_id or str(uuid.uuid4())
        self.properties: Dict[str, Any] = {}
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.status = ComponentStatus.IDLE
        self.metadata: Dict[str, Any] = {}
        self.input_connections: Dict[str, List[Connection]] = defaultdict(list)
        self.output_connections: Dict[str, List[Connection]] = defaultdict(list)
        
        # Initialize properties with defaults
        self._init_properties()
    
    def _init_properties(self):
        """Initialize properties from property definitions with defaults"""
        # Use class attribute for property definitions
        prop_defs = getattr(self.__class__, 'property_definitions', [])
        for prop_def in prop_defs:
            self.properties[prop_def.key] = prop_def.default
    
    def validate_inputs(self) -> bool:
        """Check if all required input ports are connected"""
        for port in self.input_ports:
            if not port.optional and port.name not in self.input_connections:
                return False
        return True
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get property value with optional default"""
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> bool:
        """Set property value with validation"""
        # Find property definition
        prop_def = None
        for pd in self.property_definitions:
            if pd.key == key:
                prop_def = pd
                break
        
        if not prop_def:
            return False
        
        # Validate if validator exists
        if prop_def.validator and not prop_def.validator(value):
            return False
        
        self.properties[key] = value
        return True
    
    def get_input_data(self, port_name: str) -> Any:
        """Get input data for a specific port"""
        return self.inputs.get(port_name)
    
    def set_output_data(self, port_name: str, data: Any):
        """Set output data for a specific port"""
        self.outputs[port_name] = data
    
    def get_output_data(self, port_name: str) -> Any:
        """Get output data for a specific port"""
        return self.outputs.get(port_name)
    
    def set_input_data(self, port_name: str, data: Any):
        """Set input data for a specific port"""
        self.inputs[port_name] = data
    
    def set_status(self, status: ComponentStatus):
        """Update component status"""
        self.status = status
    
    @abstractmethod
    async def process(self, context: Any) -> bool:
        """
        Process the component with given context
        Returns True if processing was successful
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, status={self.status.value})"


class Pipeline:
    """Component pipeline manager"""
    
    def __init__(self, name: str = "Untitled Pipeline"):
        self.name = name
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []
        self.execution_order: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_component(self, component: Component) -> bool:
        """Add a component to the pipeline"""
        if component.id in self.components:
            return False
        
        self.components[component.id] = component
        self._update_execution_order()
        return True
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component and its connections"""
        if component_id not in self.components:
            return False
        
        # Remove all connections involving this component
        self.connections = [
            conn for conn in self.connections
            if conn.from_component != component_id and conn.to_component != component_id
        ]
        
        # Remove from components
        del self.components[component_id]
        self._update_execution_order()
        return True
    
    def add_connection(self, connection: Connection) -> bool:
        """Add a connection between components"""
        # Validate connection
        if not connection.validate(self):
            return False
        
        # Check if components exist
        if (connection.from_component not in self.components or
            connection.to_component not in self.components):
            return False
        
        # Add to connections list
        self.connections.append(connection)
        
        # Update component connection tracking
        from_comp = self.components[connection.from_component]
        to_comp = self.components[connection.to_component]
        
        from_comp.output_connections[connection.from_port].append(connection)
        to_comp.input_connections[connection.to_port].append(connection)
        
        self._update_execution_order()
        return True
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove a connection"""
        connection = None
        for conn in self.connections:
            if conn.id == connection_id:
                connection = conn
                break
        
        if not connection:
            return False
        
        # Remove from components' connection tracking
        from_comp = self.components[connection.from_component]
        to_comp = self.components[connection.to_component]
        
        from_comp.output_connections[connection.from_port].remove(connection)
        to_comp.input_connections[connection.to_port].remove(connection)
        
        # Remove from connections list
        self.connections.remove(connection)
        self._update_execution_order()
        return True
    
    def _update_execution_order(self):
        """Update execution order using topological sort"""
        # Build adjacency list
        in_degree = {comp_id: 0 for comp_id in self.components}
        adj_list = defaultdict(list)
        
        for connection in self.connections:
            from_id = connection.from_component
            to_id = connection.to_component
            adj_list[from_id].append(to_id)
            in_degree[to_id] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([comp_id for comp_id, degree in in_degree.items() if degree == 0])
        execution_order = []
        
        while queue:
            current = queue.popleft()
            execution_order.append(current)
            
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(execution_order) != len(self.components):
            raise ValueError("Pipeline contains cycles - cannot determine execution order")
        
        self.execution_order = execution_order
    
    def validate(self) -> List[str]:
        """Validate the entire pipeline and return list of errors"""
        errors = []
        
        # Check each component's inputs
        for component in self.components.values():
            if not component.validate_inputs():
                errors.append(f"Component {component.id} has unconnected required inputs")
        
        # Validate all connections
        for connection in self.connections:
            if not connection.validate(self):
                errors.append(f"Invalid connection {connection.id}")
        
        return errors
    
    async def execute(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Execute the pipeline in topological order"""
        if not context:
            context = {}
        
        errors = self.validate()
        if errors:
            raise ValueError(f"Pipeline validation failed: {errors}")
        
        # Execute components in order
        for component_id in self.execution_order:
            component = self.components[component_id]
            
            try:
                # Prepare input data
                for port in component.input_ports:
                    if port.name in component.input_connections:
                        # Get data from connected output
                        connections = component.input_connections[port.name]
                        if connections:
                            conn = connections[0]  # Take first connection for single inputs
                            from_comp = self.components[conn.from_component]
                            data = from_comp.outputs.get(conn.from_port)
                            component.inputs[port.name] = data
                
                # Process component
                component.set_status(ComponentStatus.PROCESSING)
                success = await component.process(context)
                
                if success:
                    component.set_status(ComponentStatus.COMPLETE)
                else:
                    component.set_status(ComponentStatus.ERROR)
                    return False
                    
            except Exception as e:
                component.set_status(ComponentStatus.ERROR)
                context['last_error'] = str(e)
                return False
        
        return True
    
    def get_component_by_type(self, component_type: str) -> List[Component]:
        """Get all components of a specific type"""
        return [comp for comp in self.components.values() 
                if comp.component_type == component_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary"""
        return {
            'name': self.name,
            'components': {
                comp_id: {
                    'type': comp.component_type,
                    'properties': comp.properties,
                    'metadata': comp.metadata
                }
                for comp_id, comp in self.components.items()
            },
            'connections': [
                {
                    'id': conn.id,
                    'from_component': conn.from_component,
                    'from_port': conn.from_port,
                    'to_component': conn.to_component,
                    'to_port': conn.to_port,
                    'metadata': conn.metadata
                }
                for conn in self.connections
            ],
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return f"Pipeline(name={self.name}, components={len(self.components)}, connections={len(self.connections)})"


# Utility functions for component system

def create_property_definition(
    key: str,
    display_name: str,
    prop_type: PropertyType,
    default: Any = None,
    category: str = "General",
    **kwargs
) -> PropertyDefinition:
    """Helper function to create property definitions"""
    return PropertyDefinition(
        key=key,
        display_name=display_name,
        type=prop_type,
        default=default,
        category=category,
        **kwargs
    )


def create_port(
    name: str,
    port_type: PortType,
    direction: PortDirection,
    optional: bool = False,
    multiple: bool = False,
    **kwargs
) -> Port:
    """Helper function to create ports"""
    return Port(
        name=name,
        type=port_type,
        direction=direction,
        optional=optional,
        multiple=multiple,
        **kwargs
    )


# Property Editor System
class PropertyEditor(ABC):
    """Abstract base class for property editors"""
    
    @abstractmethod
    def create_widget(self, parent: Any, value: Any, metadata: Dict[str, Any], on_change: Callable[[Any], None]) -> Any:
        """Create Qt widget for this property type"""
        pass
    
    def validate_value(self, value: Any) -> bool:
        """Validate property value (optional override)"""
        return True
    
    def format_display_value(self, value: Any) -> str:
        """Format value for display (optional override)"""
        return str(value)


class PropertyRegistry:
    """Registry for property editors"""
    
    _editors: Dict[str, Type[PropertyEditor]] = {}
    
    @classmethod
    def register(cls, type_name: str, editor_class: Type[PropertyEditor]):
        """Register a property editor for a type"""
        cls._editors[type_name] = editor_class
    
    @classmethod
    def get_editor(cls, type_name: str) -> Optional[Type[PropertyEditor]]:
        """Get property editor for a type"""
        return cls._editors.get(type_name)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available property editor types"""
        return list(cls._editors.keys())


# Component registry for dynamic component creation
class ComponentRegistry:
    """Registry for component types"""
    
    def __init__(self):
        self._components: Dict[str, Type[Component]] = {}
    
    def register(self, component_class: Type[Component]):
        """Register a component class"""
        print(f"🔧 DEBUG: Registering component: {component_class.component_type} -> {component_class.__name__}")
        self._components[component_class.component_type] = component_class
    
    def create_component(self, component_type: str, component_id: Optional[str] = None) -> Optional[Component]:
        """Create a component instance by type"""
        if component_type not in self._components:
            return None
        
        component_class = self._components[component_type]
        return component_class(component_id)
    
    def get_available_types(self) -> List[str]:
        """Get list of available component types"""
        return list(self._components.keys())
    
    def get_component_info(self, component_type: str) -> Optional[Dict[str, Any]]:
        """Get component class information"""
        if component_type not in self._components:
            return None
        
        comp_class = self._components[component_type]
        return {
            'type': comp_class.component_type,
            'display_name': comp_class.display_name,
            'category': comp_class.category,
            'icon': comp_class.icon,
            'input_ports': [
                {'name': p.name, 'type': p.type.value, 'optional': p.optional}
                for p in comp_class.input_ports
            ],
            'output_ports': [
                {'name': p.name, 'type': p.type.value}
                for p in comp_class.output_ports
            ],
            'properties': [
                {
                    'key': p.key,
                    'display_name': p.display_name,
                    'type': p.type.value,
                    'category': p.category,
                    'default': p.default
                }
                for p in comp_class.property_definitions
            ]
        }


# Global component registry instance
component_registry = ComponentRegistry()


# ProcessContext moved to process_context.py to avoid circular imports