# CLAUDE.md - SD Forms Project Reference

## Project Overview

**SD Forms** is a sophisticated forms-based Stable Diffusion generator with a FastAPI backend and Flutter web frontend, featuring a visual node-based interface for creating AI image generation workflows. The system emphasizes modularity, extensibility, and ease of use through a component-driven architecture.

**Key Characteristics:**
- **Visual Programming Interface**: Drag-and-drop component placement with automatic connections
- **Component-Based Architecture**: Modular, reusable processing units following Unix philosophy
- **Async Processing**: Non-blocking UI with real-time preview capabilities
- **Extensible Design**: Plugin-style component registration and property editor system
- **Professional AI Workflow Tool**: Designed for both beginners and advanced users

## Architecture Summary

### Core System Foundation
The system is built on several key architectural layers:

1. **Foundation Layer**: Property system, port system, connection management
2. **Core Layer**: Pipeline system, process context, generation threading  
3. **Component Layer**: Base component system, visual components, standard components
4. **UI Layer**: Visual canvas, component toolbar, properties panel
5. **Application Layer**: Main window, toolbars, menus

### Key Design Patterns
- **Observer Pattern**: Component selection, property changes, status updates
- **Factory Pattern**: Component registry, property editor factory, connection factory
- **Command Pattern**: Undo/redo system, macro recording, batch operations
- **Strategy Pattern**: Sampling strategies, model loading, export strategies

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary language  
- **FastAPI**: Backend API framework with WebSocket support
- **Flutter Web**: Modern cross-platform frontend
- **Diffusers**: HuggingFace library for Stable Diffusion models
- **PyTorch**: Deep learning framework
- **PIL/Pillow**: Image processing
- **Asyncio**: Asynchronous programming

### Key Dependencies
```python
# Essential imports found throughout codebase
import asyncio
import torch
from diffusers import StableDiffusionPipeline, FluxPipeline
from PIL import Image
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
```

## Project Structure Deep Dive

### Critical Files by Importance

#### 1. Core Architecture (`sd_forms/core/`)
- **`component_system.py`** (571 lines) - Heart of the system
  - `Component` abstract base class
  - `Pipeline` execution orchestrator  
  - Property and port type definitions
  - Component and property registries
  - Topological sorting for execution order

- **`process_context.py`** (184 lines) - Resource management
  - Model caching to prevent redundant loads
  - Shared data between components
  - Device management (CPU/GPU)
  - Preview callback system

#### 2. Standard Components (`sd_forms/components/`)
- **`standard.py`** (845 lines) - Core AI workflow components
  - `ModelComponent`: Loads SD models, LoRAs, VAE, handles prompts
  - `SamplerComponent`: Image generation with scheduler control
  - `VAEComponent`: Latent encoding/decoding
  - `OutputComponent`: Image saving with format options

- **`base.py`** (156 lines) - Visual component foundation
  - `VisualComponent`: Hybrid class combining logic + Qt rendering
  - Connection point management and drawing

#### 3. User Interface (`sd_forms/ui/`)
- **`canvas.py`** (252 lines) - Visual workflow canvas
  - Component placement and selection
  - Auto-connection between components
  - Grid-based positioning
- **`property_editors.py`** (500 lines) - Custom property editors
  - Model picker with file browser
  - LoRA collection manager
  - Prompt editor with syntax highlighting

#### 4. Application Entry (`sd_forms/main.py`)
- **`main.py`** (583 lines) - Main application window
  - Component registration system
  - Generation workflow orchestration
  - UI layout and event handling

### Data Flow Architecture

```
User Interaction → Canvas → Component Creation → Pipeline Assembly
                                                      ↓
Generation Request → Validation → Execution Order → Async Processing
                                                      ↓
Component Processing → Data Flow → Preview Updates → Final Output
```

## Component System Details

### Component Lifecycle
1. **Registration**: `component_registry.register(ComponentClass)`
2. **Creation**: User places component on canvas
3. **Configuration**: Properties panel shows component settings
4. **Connection**: Visual connections establish data flow
5. **Validation**: Pipeline validates connections and requirements
6. **Execution**: Async processing in topological order
7. **Cleanup**: Resource management and memory cleanup

### Key Component Interface
```python
class Component(ABC):
    # Class attributes - define in subclass
    component_type: str = "my_component"
    display_name: str = "My Component"
    input_ports: List[Port] = []
    output_ports: List[Port] = []
    property_definitions: List[PropertyDefinition] = []
    
    @abstractmethod
    async def process(self, context: ProcessContext) -> bool:
        """Main processing method"""
        pass
```

### Property System
```python
# Example property definition
create_property_definition(
    "intensity", "Processing Intensity", PropertyType.FLOAT,
    default=1.0, category="Processing",
    metadata={"editor_type": "float_slider", "min": 0.0, "max": 2.0}
)
```

## Development Patterns

### Adding New Components
1. Create class inheriting from `VisualComponent`
2. Define `component_type`, `display_name`, ports, properties
3. Implement `async def process(self, context) -> bool`
4. Register in `sd_forms/main.py:register_components()`
5. Add to toolbar in `component_toolbar.py`

### Common Code Patterns

#### Async Processing Pattern
```python
async def process(self, context) -> bool:
    try:
        self.set_status(ComponentStatus.PROCESSING)
        
        # Get inputs
        input_data = self.get_input_data("input_port")
        
        # Process with context resources
        result = await self.process_data(input_data, context)
        
        # Set outputs
        self.set_output_data("output_port", result)
        
        self.set_status(ComponentStatus.COMPLETE)
        return True
    except Exception as e:
        self.set_status(ComponentStatus.ERROR)
        return False
```

#### Model Loading Pattern
```python
# In component process method
model = await context.load_model(model_path, "checkpoint")
if not model:
    return False
```

#### Progress Reporting Pattern
```python
# For real-time previews
if context.preview_callback:
    context.preview_callback(preview_image, progress_percent, step)
```

## Current System Capabilities

### Supported AI Models
- **Stable Diffusion 1.5**: Standard 512x512 generation
- **Stable Diffusion 2.1**: Enhanced 768x768 generation  
- **Stable Diffusion XL**: High-resolution 1024x1024 generation
- **Flux**: Fast 4-step generation models
- **Custom Models**: Local .safetensors and .ckpt files

### Advanced Features
- **LoRA Support**: Multiple LoRA loading with adjustable strengths
- **External VAE**: Custom VAE model integration
- **Real-time Previews**: Live generation progress with intermediate images
- **Batch Processing**: Multiple image generation
- **Model Caching**: Intelligent caching prevents redundant model loads
- **Auto-Configuration**: Automatic optimal settings based on model type

### UI Features
- **Drag-and-Drop**: Visual component placement
- **Auto-Connection**: Smart port matching and connection
- **Properties Panel**: Dynamic property editing with custom editors
- **Workflow Saving**: JSON-based workflow persistence
- **Dark Theme**: Professional dark UI theme

## Known Issues and Limitations

### Current Limitations
1. **Single Pipeline**: No support for branching workflows yet
2. **Memory Management**: Large models can consume significant GPU memory
3. **Model Detection**: Some custom models may not auto-detect correctly
4. **Connection Validation**: Limited port type compatibility checking

### Technical Debt
1. **Legacy Files**: Root directory contains older implementations
2. **Qt Threading**: Some threading patterns could be improved
3. **Error Handling**: Could benefit from more granular error reporting
4. **Testing Coverage**: Component tests could be more comprehensive

## Development Guidelines

### Code Style
- **Async First**: All component processing should be async
- **Type Hints**: Comprehensive type annotations throughout
- **Error Handling**: Graceful degradation with meaningful error messages
- **Documentation**: Docstrings for all public classes and methods

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline execution
- **UI Tests**: pytest-qt for interface testing
- **Performance Tests**: Memory and speed optimization

### Common Development Tasks

#### Debugging Component Issues
1. Check component registration in `main.py`
2. Verify port definitions and connections
3. Test async processing in isolation
4. Validate property definitions and defaults

#### Adding Model Support
1. Update model detection in `ModelComponent.detect_model_type()`
2. Add configuration to `MODEL_CONFIGS` dictionary
3. Test auto-configuration logic
4. Update documentation

#### UI Modifications
1. Qt layouts use proper spacing and margins
2. Signal/slot connections for reactive updates
3. Dark theme color consistency
4. Responsive design for different screen sizes

## Extension Points

### Plugin Architecture
The system supports plugins through:
- **Component Registry**: Register new component types
- **Property Registry**: Register custom property editors
- **Model Loading**: Extend `ProcessContext` for new model types
- **UI Extensions**: Add new panels and dialogs

### Future Enhancement Areas
1. **Distributed Processing**: Cloud/remote execution support
2. **Advanced Workflows**: Branching and conditional execution
3. **Model Management**: Built-in model downloading and organization
4. **Collaboration**: Multi-user workflow sharing
5. **Performance**: GPU optimization and memory pooling

## Troubleshooting Guide

### Common Issues

#### Component Not Appearing
- Check registration in `register_components()`
- Verify component_type string matches toolbar
- Ensure proper inheritance from `VisualComponent`

#### Generation Failures
- Verify required ports are connected
- Check model file paths and accessibility
- Validate property values and ranges
- Review error messages in console output

#### Performance Issues
- Monitor GPU memory usage
- Check model caching effectiveness
- Profile async execution timing
- Optimize property validation frequency

#### UI Problems
- Verify Qt signal/slot connections
- Check parent/child widget relationships
- Validate layout management
- Test across different screen resolutions

## Testing Approaches

### Component Testing
```python
@pytest.mark.asyncio
async def test_component_process():
    component = MyComponent()
    component.inputs["test_input"] = test_data
    
    context = ProcessContext(Mock())
    result = await component.process(context)
    
    assert result is True
    assert component.outputs["test_output"] is not None
```

### Pipeline Testing
```python
def test_pipeline_execution():
    pipeline = Pipeline()
    # Add components and connections
    # Execute and verify results
```

## Performance Characteristics

### Memory Usage
- **Model Loading**: 2-8GB GPU memory per model
- **Image Processing**: ~50MB per 1024x1024 image
- **Component Overhead**: Minimal (<1MB per component)

### Processing Speed
- **SD 1.5**: ~2-5 seconds per image (GPU)
- **SDXL**: ~5-15 seconds per image (GPU)
- **Flux**: ~1-3 seconds per image (GPU)
- **CPU Fallback**: 10-50x slower than GPU

### Optimization Strategies
- Model caching reduces subsequent load times by 90%
- Async processing keeps UI responsive
- Preview generation uses low-resolution for speed
- Batch processing amortizes setup costs

## Security Considerations

### Model File Safety
- Validates file extensions and headers
- Sandboxed model loading
- Path sanitization for file operations

### Memory Safety
- Automatic cleanup of large tensors
- GPU memory monitoring and management
- Prevention of memory leaks in long sessions

## Future Roadmap Insights

### Short-term (3-6 months)
- Improved error reporting and debugging
- Additional sampling methods and schedulers
- Enhanced model management features
- Performance optimizations

### Medium-term (6-12 months)
- Branching workflow support
- Cloud/remote processing integration
- Advanced model fine-tuning tools
- Collaborative workflow sharing

### Long-term (1+ years)
- AI-assisted workflow creation
- Advanced preprocessing pipelines
- Video generation support
- Distributed processing architecture

---

*This document serves as a comprehensive reference for understanding, developing, and maintaining the SD Forms project. It should be updated as the project evolves to maintain accuracy and usefulness for future development efforts.*