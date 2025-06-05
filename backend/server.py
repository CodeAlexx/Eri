#!/usr/bin/env python3
"""
FastAPI backend server for SD Forms
Wraps the existing component system with web API
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import our existing SD Forms components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sd_forms.core import component_registry, Pipeline, Connection
from sd_forms.components import ModelComponent, SamplerComponent, VAEComponent, OutputComponent
from sd_forms.generation import GenerationThread
from sd_forms.ui import register_property_editors
from sd_forms.main import register_components


# Pydantic models for API
class ComponentInfo(BaseModel):
    id: str
    type: str
    display_name: str
    category: str
    icon: str
    input_ports: List[Dict[str, Any]]
    output_ports: List[Dict[str, Any]]
    property_definitions: List[Dict[str, Any]]

class WorkflowComponent(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any]
    position: Dict[str, float]  # {x: float, y: float}

class WorkflowConnection(BaseModel):
    id: str
    from_component: str
    from_port: str
    to_component: str
    to_port: str

class WorkflowData(BaseModel):
    components: List[WorkflowComponent]
    connections: List[WorkflowConnection]

class ExecuteRequest(BaseModel):
    workflow: WorkflowData
    settings: Optional[Dict[str, Any]] = {}


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast_progress(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json({"type": "progress", "message": message})
            except:
                self.active_connections.remove(connection)
    
    async def broadcast_preview(self, image_data: bytes, progress: int, step: int):
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        for connection in self.active_connections.copy():
            try:
                await connection.send_json({
                    "type": "preview",
                    "image": image_b64,
                    "progress": progress,
                    "step": step
                })
            except:
                self.active_connections.remove(connection)
    
    async def broadcast_result(self, images: List[bytes]):
        import base64
        images_b64 = [base64.b64encode(img).decode('utf-8') for img in images]
        for connection in self.active_connections.copy():
            try:
                await connection.send_json({
                    "type": "result",
                    "images": images_b64
                })
            except:
                self.active_connections.remove(connection)


# Global connection manager
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting SD Forms FastAPI backend...")
    
    # Register property editors and components
    register_property_editors()
    register_components()
    
    print("✅ Components registered successfully")
    yield
    # Shutdown
    print("🛑 Shutting down SD Forms backend...")


# Create FastAPI app
app = FastAPI(
    title="SD Forms API",
    description="FastAPI backend for SD Forms visual workflow system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for LiteGraph web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Static files will be mounted after all API routes are defined


def serialize_port(port) -> Dict[str, Any]:
    """Convert port object to dictionary"""
    return {
        "name": port.name,
        "type": port.type.value if hasattr(port.type, 'value') else str(port.type),
        "direction": port.direction.value if hasattr(port.direction, 'value') else str(port.direction),
        "optional": getattr(port, 'optional', False)
    }


def serialize_property_definition(prop_def) -> Dict[str, Any]:
    """Convert property definition to dictionary"""
    return {
        "name": prop_def.key,
        "display_name": prop_def.display_name,
        "type": prop_def.type.value if hasattr(prop_def.type, 'value') else str(prop_def.type),
        "default_value": getattr(prop_def, 'default', None),
        "category": prop_def.category,
        "metadata": prop_def.metadata or {}
    }


@app.get("/api")
async def root():
    return {"message": "SD Forms API is running", "version": "1.0.0"}


@app.get("/api/components", response_model=List[ComponentInfo])
async def get_components():
    """Get list of available components with their ports and properties"""
    components = []
    
    # Get registered component classes
    for comp_type, comp_class in component_registry._components.items():
        try:
            # Create a temporary instance to get metadata
            temp_instance = comp_class()
            
            component_info = ComponentInfo(
                id=comp_type,
                type=comp_type,
                display_name=getattr(comp_class, 'display_name', comp_type.title()),
                category=getattr(comp_class, 'category', 'Core'),
                icon=getattr(comp_class, 'icon', '🔧'),
                input_ports=[serialize_port(port) for port in getattr(comp_class, 'input_ports', [])],
                output_ports=[serialize_port(port) for port in getattr(comp_class, 'output_ports', [])],
                property_definitions=[serialize_property_definition(prop) for prop in getattr(comp_class, 'property_definitions', [])]
            )
            components.append(component_info)
            
        except Exception as e:
            print(f"Error getting info for component {comp_type}: {e}")
            continue
    
    return components


@app.post("/api/execute")
async def execute_workflow(request: ExecuteRequest):
    """Execute a workflow and return results"""
    try:
        # Create pipeline from workflow data
        pipeline = Pipeline()
        
        # Create components
        component_instances = {}
        for workflow_comp in request.workflow.components:
            print(f"Creating component: {workflow_comp.type} with id {workflow_comp.id}")
            component = component_registry.create_component(
                workflow_comp.type, 
                workflow_comp.id
            )
            if component:
                print(f"✅ Component created: {component}")
                # Set properties
                component.properties.update(workflow_comp.properties)
                component_instances[workflow_comp.id] = component
                pipeline.add_component(component)
            else:
                print(f"❌ Failed to create component: {workflow_comp.type}")
                raise HTTPException(status_code=400, detail=f"Failed to create component: {workflow_comp.type}")
        
        # Create connections
        for workflow_conn in request.workflow.connections:
            connection = Connection(
                id=workflow_conn.id,
                from_component=workflow_conn.from_component,
                from_port=workflow_conn.from_port,
                to_component=workflow_conn.to_component,
                to_port=workflow_conn.to_port
            )
            pipeline.add_connection(connection)
        
        # Validate pipeline
        errors = pipeline.validate()
        print(f"🔍 Validation errors: {errors}")
        if errors:
            raise HTTPException(status_code=400, detail=f"Pipeline validation failed: {'; '.join(errors)}")
        
        # Execute pipeline
        print("🚀 Executing pipeline...")
        await manager.broadcast_progress("Starting execution...")
        
        # Create context with preview callback
        from sd_forms.core.process_context import ProcessContext
        
        class ExecutionContextWrapper(ProcessContext):
            def __init__(self, pipeline):
                super().__init__(pipeline)
                # Don't set async callbacks for now, they cause issues
                self.preview_callback = None
                self.progress_callback = None
        
        context = ExecutionContextWrapper(pipeline)
        
        # Execute pipeline
        results = await pipeline.execute(context)
        
        if results:
            # Get final images
            final_images = []
            print(f"🖼️ Checking {len(pipeline.components)} components for images...")
            for component_id, component in pipeline.components.items():
                print(f"📋 Component {component.id} ({type(component).__name__}) has outputs: {list(component.outputs.keys())}")
                if hasattr(component, 'outputs') and 'images' in component.outputs:
                    images = component.outputs['images']
                    print(f"🎯 Found {len(images) if images else 0} images in component {component.id}")
                    if images:
                        # Convert images to base64 for JSON response
                        import base64
                        import io
                        for i, img in enumerate(images):
                            if hasattr(img, 'save'):
                                buffer = io.BytesIO()
                                img.save(buffer, format='PNG')
                                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                final_images.append(img_b64)
                                print(f"✅ Converted image {i+1} to base64 ({len(img_b64)} chars)")
            print(f"🚀 Final result: {len(final_images)} images to return")
            
            await manager.broadcast_progress("Execution complete!")
            
            return {
                "success": True,
                "message": "Workflow executed successfully",
                "images": final_images,
                "execution_time": "unknown"  # Could add timing
            }
        else:
            raise HTTPException(status_code=500, detail="Pipeline execution failed")
            
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        print(f"❌ Full error details: {e}")
        import traceback
        traceback.print_exc()
        await manager.broadcast_progress(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/models")
async def get_available_models():
    """Get list of available models from config and filesystem scan"""
    try:
        config_path = Path(__file__).parent.parent / "sd_models_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Get configured models
        configured_models = {
            "flux_models": config.get("flux_models", {}),
            "vae_models": config.get("vae_models", {}),
            "popular_loras": config.get("popular_loras", {}),
            "default_settings": config.get("default_settings", {})
        }
        
        # Scan filesystem for additional models
        scanned_models = scan_model_directories(config)
        
        # Combine and organize all models for expandable dropdown
        all_models = organize_models_for_dropdown(configured_models, scanned_models)
        
        return {
            **configured_models,
            "all_models": all_models,
            "organized_models": {
                "Flux Models": all_models.get("flux", []),
                "SDXL Models": all_models.get("sdxl", []),
                "SD 1.5 Models": all_models.get("sd15", []),
                "Other Models": all_models.get("other", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model config: {e}")


def scan_model_directories(config: Dict) -> Dict[str, List]:
    """Scan model directories for all available models"""
    scanned = {
        "flux": [],
        "sdxl": [], 
        "sd15": [],
        "other": []
    }
    
    try:
        # Get model paths from config
        model_paths = config.get("model_paths", {})
        base_paths = [
            model_paths.get("checkpoints", "/home/alex/SwarmUI/Models/diffusion_models"),
            "/home/alex/SwarmUI/Models/Stable-Diffusion",
            "/home/alex/models",
            "./models"
        ]
        
        # Scan each directory
        for base_path in base_paths:
            base_path = Path(base_path)
            if not base_path.exists():
                continue
                
            # Look for model files
            model_extensions = ["*.safetensors", "*.ckpt", "*.bin", "*.pt"]
            for pattern in model_extensions:
                for model_file in base_path.rglob(pattern):
                    if model_file.is_file():
                        model_info = classify_model_file(model_file)
                        if model_info:
                            category = model_info["category"]
                            scanned[category].append(model_info)
        
        # Remove duplicates and sort
        for category in scanned:
            scanned[category] = sorted(list({m["path"]: m for m in scanned[category]}.values()), 
                                     key=lambda x: x["name"])
                                     
    except Exception as e:
        print(f"Error scanning model directories: {e}")
    
    return scanned


def classify_model_file(model_path: Path) -> Optional[Dict]:
    """Classify a model file based on filename and size"""
    try:
        name = model_path.name
        name_lower = name.lower()
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Determine category based on filename patterns
        category = "other"
        model_type = "unknown"
        
        if any(term in name_lower for term in ["flux", "flux1", "flux-"]):
            category = "flux"
            model_type = "flux"
        elif any(term in name_lower for term in ["xl", "sdxl", "stable-diffusion-xl"]):
            category = "sdxl" 
            model_type = "sdxl"
        elif any(term in name_lower for term in ["sd15", "sd-1.5", "stable-diffusion-v1-5", "v1-5"]):
            category = "sd15"
            model_type = "sd15"
        elif size_mb > 5000:  # Large files are likely SDXL
            category = "sdxl"
            model_type = "sdxl"
        elif 2000 < size_mb < 5000:  # Medium files likely SD 1.5
            category = "sd15" 
            model_type = "sd15"
        
        return {
            "name": name,
            "display_name": name.replace('.safetensors', '').replace('.ckpt', '').replace('_', ' '),
            "path": str(model_path),
            "category": category,
            "type": model_type,
            "size_mb": round(size_mb, 1),
            "extension": model_path.suffix
        }
        
    except Exception as e:
        print(f"Error classifying model {model_path}: {e}")
        return None


def organize_models_for_dropdown(configured: Dict, scanned: Dict) -> Dict:
    """Organize all models for frontend dropdown display"""
    organized = {
        "flux": [],
        "sdxl": [],
        "sd15": [],
        "other": []
    }
    
    # Add configured models first (with priority)
    for flux_key, flux_info in configured.get("flux_models", {}).items():
        organized["flux"].append({
            "id": flux_key,
            "name": flux_info.get("name", flux_key),
            "path": flux_info.get("path", ""),
            "configured": True,
            "steps": flux_info.get("steps", 4),
            "cfg": flux_info.get("cfg", 1.0),
            "description": flux_info.get("description", "")
        })
    
    # Add scanned models (avoiding duplicates)
    configured_paths = set()
    for category_models in configured.values():
        if isinstance(category_models, dict):
            for model_info in category_models.values():
                if isinstance(model_info, dict) and "path" in model_info:
                    configured_paths.add(model_info["path"])
    
    for category, models in scanned.items():
        for model in models:
            if model["path"] not in configured_paths:
                organized[category].append({
                    "id": f"scanned_{Path(model['path']).stem}",
                    "name": model["display_name"],
                    "path": model["path"],
                    "configured": False,
                    "size_mb": model["size_mb"],
                    "type": model["type"]
                })
    
    return organized


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/litegraph_nodes")
async def get_litegraph_nodes():
    """Convert SD Forms components to LiteGraph node definitions"""
    try:
        nodes = {}
        
        # Get components from registry
        components = component_registry._components if hasattr(component_registry, '_components') else {}
        
        for comp_type, comp_class in components.items():
            # Create LiteGraph node definition
            node_def = {
                "title": getattr(comp_class, 'display_name', comp_type),
                "category": f"SD Forms/{getattr(comp_class, 'category', 'General')}",
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets": []
            }
            
            # Add inputs
            if hasattr(comp_class, 'input_ports'):
                for port in comp_class.input_ports:
                    node_def["inputs"].append({
                        "name": port.name,
                        "type": getattr(port.type, '__name__', str(port.type)) if hasattr(port, 'type') else str(port.data_type)
                    })
            
            # Add outputs
            if hasattr(comp_class, 'output_ports'):
                for port in comp_class.output_ports:
                    node_def["outputs"].append({
                        "name": port.name,
                        "type": getattr(port.type, '__name__', str(port.type)) if hasattr(port, 'type') else str(port.data_type)
                    })
            
            # Add properties as widgets
            if hasattr(comp_class, 'property_definitions'):
                for prop in comp_class.property_definitions:
                    prop_type = str(prop.property_type).replace('PropertyType.', '') if hasattr(prop, 'property_type') else 'STRING'
                    
                    widget = {
                        "name": getattr(prop, 'name', getattr(prop, 'key', 'unknown')),
                        "type": prop_type,
                        "default": getattr(prop, 'default_value', getattr(prop, 'default', None)),
                        "options": getattr(prop, 'metadata', {})
                    }
                    node_def["widgets"].append(widget)
                    node_def["properties"][widget["name"]] = widget["default"]
            
            nodes[comp_type] = node_def
        
        return nodes
        
    except Exception as e:
        print(f"Error in litegraph_nodes: {e}")
        # Return some basic fallback nodes
        return {
            "model": {
                "title": "Model",
                "category": "SD Forms/Core",
                "inputs": [],
                "outputs": [{"name": "pipeline", "type": "pipeline"}, {"name": "conditioning", "type": "conditioning"}],
                "properties": {"model_path": "stabilityai/stable-diffusion-xl-base-1.0", "prompt": "A beautiful landscape"},
                "widgets": [
                    {"name": "model_path", "type": "STRING", "default": "stabilityai/stable-diffusion-xl-base-1.0"},
                    {"name": "prompt", "type": "STRING", "default": "A beautiful landscape"}
                ]
            },
            "sampler": {
                "title": "Sampler", 
                "category": "SD Forms/Core",
                "inputs": [{"name": "pipeline", "type": "pipeline"}, {"name": "conditioning", "type": "conditioning"}],
                "outputs": [{"name": "image", "type": "image"}],
                "properties": {"steps": 25, "cfg_scale": 7.0},
                "widgets": [
                    {"name": "steps", "type": "INTEGER", "default": 25},
                    {"name": "cfg_scale", "type": "FLOAT", "default": 7.0}
                ]
            },
            "output": {
                "title": "Output",
                "category": "SD Forms/Core", 
                "inputs": [{"name": "image", "type": "image"}],
                "outputs": [],
                "properties": {"format": "PNG"},
                "widgets": [{"name": "format", "type": "STRING", "default": "PNG"}]
            }
        }


# Serve static files and index
web_path = Path(__file__).parent.parent / "web"

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse(web_path / 'index.html')

# Serve static files
app.mount("/static", StaticFiles(directory=str(web_path)), name="static")

# Serve lib files if needed
@app.get("/lib/{file_path:path}")
async def serve_lib(file_path: str):
    return FileResponse(web_path / "lib" / file_path)

print(f"✅ Serving LiteGraph web frontend from: {web_path}")

# Mount additional static routes for better compatibility
@app.get("/js/{file_path:path}")
async def serve_js(file_path: str):
    return FileResponse(web_path / "js" / file_path)

@app.get("/css/{file_path:path}")
async def serve_css(file_path: str):
    return FileResponse(web_path / "css" / file_path)

@app.get("/assets/{file_path:path}")
async def serve_assets(file_path: str):
    return FileResponse(web_path / "assets" / file_path)


if __name__ == "__main__":
    print("🌐 Starting SD Forms FastAPI server...")
    print("🎨 Serving LiteGraph web frontend and API")
    print("🔗 Web UI: http://localhost:8001")
    print("🔗 API docs: http://localhost:8001/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )