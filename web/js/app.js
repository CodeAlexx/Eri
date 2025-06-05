/**
 * SD Forms Web UI - COMPLETELY REWRITTEN FOR WORKING WIDGETS
 */

console.log("LiteGraph version:", LiteGraph.VERSION || "Unknown");

// FORCE SMALL NODE SETTINGS GLOBALLY
LiteGraph.NODE_TEXT_SIZE = 10;
LiteGraph.NODE_WIDGET_HEIGHT = 16;
LiteGraph.NODE_TITLE_HEIGHT = 20;
LiteGraph.NODE_MIN_WIDTH = 150;

let graph, canvas;

window.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Starting SD Forms with working widgets...');
    
    // Create graph and canvas
    graph = new LGraph();
    canvas = new LGraphCanvas("#mycanvas", graph);
    
    // FIX BLURRY CANVAS - High-DPI support
    const canvasElement = canvas.canvas;
    const ctx = canvasElement.getContext('2d');
    const devicePixelRatio = window.devicePixelRatio || 1;
    
    // Set actual size in memory (scaled to account for extra pixel density)
    const rect = canvasElement.getBoundingClientRect();
    canvasElement.width = rect.width * devicePixelRatio;
    canvasElement.height = rect.height * devicePixelRatio;
    
    // Scale the drawing context so everything draws at the correct size
    ctx.scale(devicePixelRatio, devicePixelRatio);
    
    // Set display size (should be the same as rect size)
    canvasElement.style.width = rect.width + 'px';
    canvasElement.style.height = rect.height + 'px';
    
    // FIX ZOOM - Set proper initial scale
    canvas.ds.scale = 0.6; // Start zoomed out to see multiple nodes
    canvas.ds.offset = [0, 0]; // Center view
    
    // Enable interaction
    canvas.allow_dragcanvas = true;
    canvas.allow_dragnodes = true;
    canvas.allow_interaction = true;
    
    // Force sharp rendering
    ctx.imageSmoothingEnabled = false;
    canvas.render_canvas_border = false;
    
    console.log(`Canvas fixed: ${canvasElement.width}x${canvasElement.height}, DPR: ${devicePixelRatio}, scale: ${canvas.ds.scale}`);
    
    // Start graph
    graph.start();
    
    // Make global for debugging
    window.graph = graph;
    window.canvas = canvas;
    
    // Load components and create working nodes
    await loadAndRegisterComponents();
    
    // Setup UI
    setupEventHandlers();
    setupNodeSelection();
    setupComponentToolbar();
    
    console.log('✅ SD Forms initialized');
});

async function loadAndRegisterComponents() {
    try {
        const response = await fetch('/api/components');
        const components = await response.json();
        
        console.log(`Loading ${components.length} components...`);
        
        // Register each component with WORKING widgets
        components.forEach((comp, index) => {
            const nodeType = comp.id || comp.type;
            
            function SDFormsNode() {
                this.title = comp.display_name || nodeType;
                this.size = [220, 60]; // Start small, will grow
                this.properties = {};
                
                console.log(`Creating ${nodeType} node...`);
                
                // Add I/O ports
                if (comp.input_ports) {
                    comp.input_ports.forEach(port => {
                        this.addInput(port.name, port.type);
                    });
                }
                
                if (comp.output_ports) {
                    comp.output_ports.forEach(port => {
                        this.addOutput(port.name, port.type);
                    });
                }
                
                // Add WORKING widgets with proper callbacks
                if (comp.property_definitions) {
                    // FOR MODEL COMPONENTS: Fix defaults to use Flux instead of SD1.5
                    if (nodeType === 'model') {
                        // Override defaults for model component
                        const modelDefaults = {
                            checkpoint_type: 'preset',
                            flux_preset: 'flux1-schnell',
                            prompt: 'a beautiful landscape, highly detailed',
                            negative_prompt: '',
                            cfg_scale: 1.0  // Flux uses lower CFG
                        };
                        Object.assign(this.properties, modelDefaults);
                    }
                    
                    comp.property_definitions.slice(0, 5).forEach(prop => { // Limit to 5 widgets
                        let defaultValue = prop.default_value || "";
                        
                        // FORCE FLUX DEFAULTS for model component
                        if (nodeType === 'model') {
                            if (prop.name === 'checkpoint_type') defaultValue = 'preset';
                            if (prop.name === 'flux_preset') defaultValue = 'flux1-schnell';
                            if (prop.name === 'cfg_scale') defaultValue = 1.0;
                        }
                        
                        this.properties[prop.name] = defaultValue;
                        
                        console.log(`  Adding widget: ${prop.name} (${prop.type})`);
                        
                        try {
                            switch(prop.type.toLowerCase()) {
                                case 'choice':
                                    const choices = prop.metadata?.choices || ['option1', 'option2'];
                                    
                                    // Special handling for model-related dropdowns
                                    if (prop.name === 'flux_preset' || prop.name === 'checkpoint' || prop.name === 'model_variant') {
                                        this.addWidget("combo", prop.name, defaultValue, (value) => {
                                            this.properties[prop.name] = value;
                                            console.log(`${prop.name} = ${value}`);
                                            // Update model display name when model changes
                                            updateModelDisplayNames();
                                        }, { 
                                            values: choices,
                                            property: "expandable_model_selector"
                                        });
                                    } else {
                                        this.addWidget("combo", prop.name, defaultValue, (value) => {
                                            this.properties[prop.name] = value;
                                            console.log(`${prop.name} = ${value}`);
                                        }, { values: choices });
                                    }
                                    break;
                                    
                                case 'text':
                                case 'string':
                                    this.addWidget("text", prop.name, defaultValue, (value) => {
                                        this.properties[prop.name] = value;
                                        console.log(`${prop.name} = ${value}`);
                                    });
                                    break;
                                    
                                case 'integer':
                                    this.addWidget("number", prop.name, parseInt(defaultValue) || 0, (value) => {
                                        this.properties[prop.name] = parseInt(value) || 0;
                                        console.log(`${prop.name} = ${value}`);
                                    });
                                    break;
                                    
                                case 'float':
                                    this.addWidget("number", prop.name, parseFloat(defaultValue) || 0, (value) => {
                                        this.properties[prop.name] = parseFloat(value) || 0;
                                        console.log(`${prop.name} = ${value}`);
                                    });
                                    break;
                                    
                                case 'boolean':
                                    this.addWidget("toggle", prop.name, defaultValue || false, (value) => {
                                        this.properties[prop.name] = value;
                                        console.log(`${prop.name} = ${value}`);
                                    });
                                    break;
                                    
                                default:
                                    // Fallback to text
                                    this.addWidget("text", prop.name, defaultValue, (value) => {
                                        this.properties[prop.name] = value;
                                        console.log(`${prop.name} = ${value}`);
                                    });
                            }
                        } catch (error) {
                            console.warn(`Failed to add widget ${prop.name}:`, error);
                        }
                    });
                }
                
                // Auto-size but keep reasonable limits
                this.size = this.computeSize();
                if (this.size[0] > 300) this.size[0] = 300;
                if (this.size[1] > 200) this.size[1] = 200;
                
                console.log(`  ✅ ${nodeType} created: ${this.widgets?.length || 0} widgets, size [${this.size[0]}, ${this.size[1]}]`);
            }
            
            // Proper inheritance
            SDFormsNode.prototype = Object.create(LGraphNode.prototype);
            SDFormsNode.prototype.constructor = SDFormsNode;
            SDFormsNode.title = comp.display_name || nodeType;
            
            // Register with LiteGraph
            LiteGraph.registerNodeType(`sdforms/${nodeType}`, SDFormsNode);
            console.log(`✅ Registered: sdforms/${nodeType}`);
        });
        
        // Create test workflow
        setTimeout(createTestWorkflow, 1000);
        
    } catch (error) {
        console.error('Failed to load components:', error);
        createFallbackNodes();
    }
}

function createTestWorkflow() {
    console.log('Creating test workflow...');
    
    // Clear existing
    graph.clear();
    
    // Update status
    document.getElementById('statusText').textContent = 'Connected';
    document.getElementById('statusText').className = 'status-connected';
    
    // Create 4 test nodes: Model → Sampler → ImageDisplay → Output
    const modelNode = LiteGraph.createNode("sdforms/flux_model");
    if (modelNode) {
        modelNode.pos = [50, 50];
        graph.add(modelNode);
        console.log(`Model node: ${modelNode.widgets?.length || 0} widgets`);
    }
    
    const samplerNode = LiteGraph.createNode("sdforms/sampler");
    if (samplerNode) {
        samplerNode.pos = [300, 50];
        graph.add(samplerNode);
        console.log(`Sampler node: ${samplerNode.widgets?.length || 0} widgets`);
    }
    
    const displayNode = LiteGraph.createNode("sdforms/media_display");
    if (displayNode) {
        displayNode.pos = [550, 50];
        graph.add(displayNode);
        console.log(`MediaDisplay node: ${displayNode.widgets?.length || 0} widgets`);
    }
    
    const outputNode = LiteGraph.createNode("sdforms/output");
    if (outputNode) {
        outputNode.pos = [800, 50];
        graph.add(outputNode);
        console.log(`Output node: ${outputNode.widgets?.length || 0} widgets`);
    }
    
    // AUTO-CONNECT nodes after a delay
    setTimeout(() => {
        if (modelNode && samplerNode && displayNode && outputNode) {
            try {
                // Connect model outputs to sampler inputs
                if (modelNode.outputs && samplerNode.inputs) {
                    modelNode.connect(0, samplerNode, 0); // pipeline
                    if (modelNode.outputs.length > 1 && samplerNode.inputs.length > 1) {
                        modelNode.connect(1, samplerNode, 1); // conditioning
                    }
                }
                
                // Connect sampler output to display input
                if (samplerNode.outputs && displayNode.inputs) {
                    samplerNode.connect(0, displayNode, 0); // image
                }
                
                // Connect display output to output input
                if (displayNode.outputs && outputNode.inputs) {
                    displayNode.connect(0, outputNode, 0); // image
                }
                
                console.log('✅ Auto-connected workflow: Model → Sampler → MediaDisplay → Output');
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
        
        // Center view on workflow
        if (canvas.centerOnNode && modelNode) {
            canvas.centerOnNode(modelNode);
        }
    }, 500);
}

function createFallbackNodes() {
    console.log('Creating fallback test nodes...');
    
    function SimpleNode() {
        this.title = "Simple Test";
        this.size = [180, 100];
        
        this.addInput("input", "*");
        this.addOutput("output", "*");
        
        this.addWidget("text", "prompt", "test prompt", (v) => {
            this.properties.prompt = v;
            console.log("Prompt changed:", v);
        });
        
        this.addWidget("number", "steps", 20, (v) => {
            this.properties.steps = parseInt(v);
            console.log("Steps changed:", v);
        });
        
        this.properties = { prompt: "test prompt", steps: 20 };
    }
    
    SimpleNode.prototype = Object.create(LGraphNode.prototype);
    SimpleNode.title = "Simple Test";
    LiteGraph.registerNodeType("test/simple", SimpleNode);
    
    // Create test node
    const node = LiteGraph.createNode("test/simple");
    node.pos = [100, 100];
    graph.add(node);
}

function setupEventHandlers() {
    // Setup draggable object inspector
    setupDraggableInspector();
    
    document.getElementById('testBtn')?.addEventListener('click', () => {
        console.log('Creating manual test node...');
        const node = LiteGraph.createNode("sdforms/flux_model");
        if (node) {
            node.pos = [Math.random() * 200 + 100, Math.random() * 200 + 100];
            graph.add(node);
            if (canvas.centerOnNode) canvas.centerOnNode(node);
        }
    });
    
    document.getElementById('clearBtn')?.addEventListener('click', () => {
        graph.clear();
    });
    
    document.getElementById('centerBtn')?.addEventListener('click', () => {
        if (graph._nodes.length > 0 && canvas.centerOnNode) {
            canvas.centerOnNode(graph._nodes[0]);
        }
    });
    
    // Improved zoom controls
    document.getElementById('zoomInBtn')?.addEventListener('click', () => {
        if (canvas.ds) {
            canvas.ds.scale = Math.min(canvas.ds.scale * 1.2, 2.0);
            canvas.setDirty(true, true);
            console.log('Zoom:', canvas.ds.scale);
        }
    });
    
    document.getElementById('zoomOutBtn')?.addEventListener('click', () => {
        if (canvas.ds) {
            canvas.ds.scale = Math.max(canvas.ds.scale * 0.8, 0.2);
            canvas.setDirty(true, true);
            console.log('Zoom:', canvas.ds.scale);
        }
    });
    
    document.getElementById('zoomResetBtn')?.addEventListener('click', () => {
        if (canvas.ds) {
            canvas.ds.scale = 0.6; // Good default zoom level
            canvas.ds.offset = [0, 0];
            canvas.setDirty(true, true);
            console.log('Reset zoom to 0.6');
        }
    });
    
    // Scale button - make everything smaller
    document.getElementById('scaleBtn')?.addEventListener('click', () => {
        if (canvas.ds) {
            canvas.ds.scale = 0.4; // Very zoomed out
            canvas.setDirty(true, true);
            console.log('Scaled to 0.4x');
        }
    });
    
    // Fit content button
    document.getElementById('fitBtn')?.addEventListener('click', () => {
        if (canvas.ds && graph._nodes.length > 0) {
            // Calculate bounding box of all nodes
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            
            graph._nodes.forEach(node => {
                minX = Math.min(minX, node.pos[0]);
                minY = Math.min(minY, node.pos[1]);
                maxX = Math.max(maxX, node.pos[0] + node.size[0]);
                maxY = Math.max(maxY, node.pos[1] + node.size[1]);
            });
            
            // Calculate scale to fit all nodes
            const margin = 50;
            const canvasWidth = canvas.canvas.width / window.devicePixelRatio;
            const canvasHeight = canvas.canvas.height / window.devicePixelRatio;
            const contentWidth = maxX - minX + margin * 2;
            const contentHeight = maxY - minY + margin * 2;
            
            const scaleX = canvasWidth / contentWidth;
            const scaleY = canvasHeight / contentHeight;
            const scale = Math.min(scaleX, scaleY, 1.0); // Don't zoom in beyond 1:1
            
            canvas.ds.scale = scale;
            canvas.ds.offset = [
                (canvasWidth - contentWidth * scale) / 2 - minX * scale + margin * scale,
                (canvasHeight - contentHeight * scale) / 2 - minY * scale + margin * scale
            ];
            
            canvas.setDirty(true, true);
            console.log(`Fit content: scale ${scale.toFixed(2)}`);
        }
    });
}

// Enhanced execution with progress feedback
async function executeWorkflow() {
    console.log('🚀 Starting workflow execution...');
    
    // Show progress
    showProgress('Preparing workflow...', 0);
    
    const nodes = graph._nodes;
    if (nodes.length === 0) {
        hideProgress();
        alert('No nodes in workflow! Click "Load Standard Form" or add components.');
        return;
    }
    
    // Extract connections from the graph
    const connections = [];
    if (graph.links) {
        Object.values(graph.links).forEach(link => {
            if (link) {
                // Get the actual port names from the components
                const fromNode = nodes.find(n => n.id == link.origin_id);
                const toNode = nodes.find(n => n.id == link.target_id);
                
                let fromPortName = link.origin_slot.toString();
                let toPortName = link.target_slot.toString();
                
                // Try to get actual port names if available
                if (fromNode && fromNode.outputs && fromNode.outputs[link.origin_slot]) {
                    fromPortName = fromNode.outputs[link.origin_slot].name || fromPortName;
                }
                if (toNode && toNode.inputs && toNode.inputs[link.target_slot]) {
                    toPortName = toNode.inputs[link.target_slot].name || toPortName;
                }
                
                connections.push({
                    id: `conn_${link.id}`,
                    from_component: link.origin_id.toString(),
                    from_port: fromPortName,
                    to_component: link.target_id.toString(),
                    to_port: toPortName
                });
            }
        });
    }
    
    const workflow = {
        components: nodes.map(node => ({
            id: node.id.toString(),
            type: node.type.replace('sdforms/', ''),
            properties: node.properties || {},
            position: {
                x: node.pos[0] || 0,
                y: node.pos[1] || 0
            }
        })),
        connections: connections
    };
    
    try {
        showProgress('Connecting to server...', 10);
        
        const response = await fetch('/api/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workflow })
        });
        
        showProgress('Processing workflow...', 30);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Execution result:', result);
        
        if (result.success) {
            showProgress('Generation complete!', 100);
            
            // Display generated images directly in image components on canvas
            if (result.images && result.images.length > 0) {
                displayImagesInComponents(result.images);
                console.log(`🖼️ Generated ${result.images.length} images`);
            } else {
                console.warn('⚠️ No images returned from workflow');
            }
            
            setTimeout(hideProgress, 2000);
        } else {
            throw new Error(result.message || 'Workflow execution failed');
        }
        
    } catch (error) {
        console.error('❌ Execution failed:', error);
        showProgress(`Error: ${error.message}`, 0);
        setTimeout(hideProgress, 3000);
    }
}

function showProgress(message, percent) {
    const progressDiv = document.getElementById('progress');
    const progressText = document.getElementById('progressText');
    const progressFill = document.getElementById('progressFill');
    
    if (progressDiv) {
        progressDiv.style.display = 'block';
    }
    if (progressText) {
        progressText.textContent = message;
    }
    if (progressFill) {
        progressFill.style.width = `${percent}%`;
    }
    
    console.log(`📊 ${message} (${percent}%)`);
}

function hideProgress() {
    const progressDiv = document.getElementById('progress');
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
}

function showGeneratedMedia(mediaItems) {
    console.log(`🎬 Displaying ${mediaItems.length} generated media items`);
    
    // Show preview panel
    const preview = document.getElementById('preview');
    if (preview) {
        preview.style.display = 'block';
        
        // Clear existing content
        preview.innerHTML = '<div style="margin-bottom: 10px;">Generated Media:</div>';
        
        // Add media items
        mediaItems.forEach((item, index) => {
            if (item.type === 'image') {
                // Handle images
                const imgElement = document.createElement('img');
                imgElement.src = `data:image/png;base64,${item.data}`;
                imgElement.style.width = '100%';
                imgElement.style.marginBottom = '5px';
                imgElement.style.borderRadius = '4px';
                imgElement.style.cursor = 'pointer';
                imgElement.title = `Generated image ${index + 1}`;
                
                // Click to open in new tab
                imgElement.addEventListener('click', () => {
                    const newWindow = window.open();
                    newWindow.document.write(`<img src="data:image/png;base64,${item.data}" style="max-width: 100%; height: auto;">`);
                    newWindow.document.title = `Generated Image ${index + 1}`;
                });
                
                preview.appendChild(imgElement);
                
            } else if (item.type === 'video') {
                // Handle videos
                const videoElement = document.createElement('video');
                videoElement.src = `data:video/mp4;base64,${item.data}`;
                videoElement.style.width = '100%';
                videoElement.style.marginBottom = '5px';
                videoElement.style.borderRadius = '4px';
                videoElement.controls = true;
                videoElement.autoplay = item.autoplay || false;
                videoElement.loop = item.loop || false;
                videoElement.title = `Generated video ${index + 1}`;
                
                preview.appendChild(videoElement);
            }
        });
        
        console.log(`✅ Media displayed in preview panel`);
    }
}

function displayImagesInComponents(images) {
    console.log('🖼️ Displaying images in canvas components...');
    
    // Find image/media display components in the graph
    const imageComponents = graph._nodes.filter(node => 
        node.type === 'sdforms/image' || node.type === 'sdforms/media_display'
    );
    
    if (imageComponents.length === 0) {
        console.warn('⚠️ No image components found on canvas, showing in preview panel');
        showGeneratedImages(images);
        return;
    }
    
    // Display the first image in the first image component found
    const targetComponent = imageComponents[0];
    const firstImage = images[0];
    
    if (firstImage && targetComponent) {
        // Store the image data in the component
        targetComponent.generatedImage = firstImage;
        
        // Update the component's visual appearance to show the image
        updateComponentWithImage(targetComponent, firstImage);
        
        console.log(`✅ Image displayed in component: ${targetComponent.title}`);
    }
}

function updateComponentWithImage(component, imageB64) {
    // Store the image data in the component
    component.generatedImageData = imageB64;
    
    // Create an Image object for drawing
    if (!component.imageObj) {
        component.imageObj = new Image();
    }
    
    component.imageObj.onload = () => {
        // Force canvas redraw when image loads
        if (canvas) {
            canvas.setDirty(true, true);
        }
    };
    
    component.imageObj.src = `data:image/png;base64,${imageB64}`;
    
    // Get the resolution from workflow components
    const resolution = getWorkflowResolution();
    const targetWidth = resolution.width || 512;
    const targetHeight = resolution.height || 512;
    
    // Calculate component size based on resolution with proper scaling
    const maxDisplaySize = 400; // Maximum size for display
    const aspectRatio = targetWidth / targetHeight;
    
    let componentWidth, componentHeight;
    if (aspectRatio > 1) {
        // Landscape
        componentWidth = Math.min(maxDisplaySize, targetWidth * 0.3);
        componentHeight = componentWidth / aspectRatio;
    } else {
        // Portrait or square
        componentHeight = Math.min(maxDisplaySize, targetHeight * 0.3);
        componentWidth = componentHeight * aspectRatio;
    }
    
    // Ensure minimum size for usability
    componentWidth = Math.max(componentWidth, 200);
    componentHeight = Math.max(componentHeight, 150);
    
    // Add padding for title and border
    component.size = [componentWidth + 20, componentHeight + 40];
    
    console.log(`📏 Resized image component to ${component.size[0]}x${component.size[1]} for ${targetWidth}x${targetHeight} image`);
    
    // Override the onDrawForeground method to draw the image
    component.onDrawForeground = function(ctx) {
        if (this.imageObj && this.imageObj.complete) {
            // Calculate image position and size within the component
            const padding = 5;
            const titleHeight = 20;
            const drawX = padding;
            const drawY = titleHeight + padding;
            const drawWidth = this.size[0] - (padding * 2);
            const drawHeight = this.size[1] - titleHeight - (padding * 2);
            
            if (drawWidth > 0 && drawHeight > 0) {
                // Draw a background
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                
                // Calculate aspect ratio to fit image
                const imgAspect = this.imageObj.width / this.imageObj.height;
                const boxAspect = drawWidth / drawHeight;
                
                let imgDrawWidth, imgDrawHeight, imgDrawX, imgDrawY;
                
                if (imgAspect > boxAspect) {
                    // Image is wider than box
                    imgDrawWidth = drawWidth;
                    imgDrawHeight = drawWidth / imgAspect;
                    imgDrawX = drawX;
                    imgDrawY = drawY + (drawHeight - imgDrawHeight) / 2;
                } else {
                    // Image is taller than box
                    imgDrawHeight = drawHeight;
                    imgDrawWidth = drawHeight * imgAspect;
                    imgDrawX = drawX + (drawWidth - imgDrawWidth) / 2;
                    imgDrawY = drawY;
                }
                
                // Draw the image
                ctx.drawImage(this.imageObj, imgDrawX, imgDrawY, imgDrawWidth, imgDrawHeight);
                
                // Draw border
                ctx.strokeStyle = "#555";
                ctx.lineWidth = 1;
                ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
                
                // Draw resolution info in corner
                ctx.fillStyle = "#666";
                ctx.font = "10px Arial";
                ctx.fillText(`${targetWidth}x${targetHeight}`, drawX + 3, drawY + drawHeight - 3);
            }
        }
    };
    
    // Add click handler for full-size view
    component.onMouseDown = function(e, localpos) {
        if (this.generatedImageData && localpos[1] > 20) { // Below title area
            const newWindow = window.open();
            newWindow.document.write(`
                <html>
                    <head><title>Generated Image</title></head>
                    <body style="margin:0; background:#000; display:flex; align-items:center; justify-content:center;">
                        <img src="data:image/png;base64,${this.generatedImageData}" style="max-width:100%; max-height:100vh; object-fit:contain;" />
                    </body>
                </html>
            `);
            newWindow.document.title = 'Generated Image';
            return true; // Consume the event
        }
        return false;
    };
    
    // Force canvas redraw
    if (canvas) {
        canvas.setDirty(true, true);
    }
}

function getWorkflowResolution() {
    // Get resolution from sampler or model components in the current workflow
    let width = 1024, height = 1024; // Default fallback
    
    // Check all nodes for width/height properties
    if (graph && graph._nodes) {
        for (const node of graph._nodes) {
            if (node.properties) {
                // Check for width/height in properties
                if (node.properties.width && node.properties.height) {
                    width = parseInt(node.properties.width) || width;
                    height = parseInt(node.properties.height) || height;
                    console.log(`📐 Found resolution in ${node.title}: ${width}x${height}`);
                    break;
                }
            }
        }
    }
    
    return { width, height };
}

function showGeneratedImages(images) {
    // Convert legacy image format to new media format
    const mediaItems = images.map((imgB64, index) => ({
        type: 'image',
        data: imgB64,
        index: index
    }));
    showGeneratedMedia(mediaItems);
}

// Additional event handlers for save/load
function setupSaveLoad() {
    document.getElementById('saveBtn')?.addEventListener('click', () => {
        const data = graph.serialize();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `workflow_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        console.log('Workflow saved');
    });
    
    document.getElementById('loadBtn')?.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
    
    document.getElementById('fileInput')?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    graph.configure(data);
                    console.log('Workflow loaded');
                } catch (error) {
                    console.error('Failed to load workflow:', error);
                    alert('Failed to load workflow file');
                }
            };
            reader.readAsText(file);
        }
    });
}

// Connect all event handlers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('generateBtn')?.addEventListener('click', executeWorkflow);
    document.getElementById('basicWorkflowBtn')?.addEventListener('click', createBasicWorkflow);
    setupSaveLoad();
});

// Global variables for UI state
let selectedNode = null;
let availableComponents = [];
let allAvailableModels = {};
let organizedModels = {};

async function setupComponentToolbar() {
    console.log('Setting up component toolbar...');
    
    try {
        // Load components
        const response = await fetch('/api/components');
        availableComponents = await response.json();
        
        // Load models for expandable dropdowns
        await loadAllAvailableModels();
        
        // Categorize components properly based on their actual capabilities
        console.log('🔍 Available component IDs:', availableComponents.map(c => c.id));
        const coreImageModels = availableComponents.filter(comp => 
            ['sdxl_model', 'flux_model', 'hidream_i1', 'lumina_control', 'sampler', 'output', 'sd15', 'sd35', 'chroma', 'omnigen'].includes(comp.id)
        );
        console.log('🎯 Core image models found:', coreImageModels.map(c => c.id));
        const videoModelComponents = availableComponents.filter(comp => 
            ['wan_vace', 'animatediff', 'ltx_video'].includes(comp.id)
        );
        const displayComponents = availableComponents.filter(comp => 
            ['image', 'media_display'].includes(comp.id)
        );
        const advancedComponents = availableComponents.filter(comp => 
            !['sdxl_model', 'flux_model', 'lumina_control', 'sampler', 'image', 'media_display', 'output', 'vae', 'hidream_i1', 'wan_vace', 'animatediff', 'ltx_video', 'sd15', 'sd35', 'chroma', 'omnigen'].includes(comp.id)
        );
        
        // Populate image model components
        const coreContainer = document.getElementById('coreComponents');
        if (coreContainer) {
            coreContainer.innerHTML = '';
            coreImageModels.forEach(comp => {
                const button = document.createElement('button');
                button.className = 'component-btn';
                button.textContent = `${comp.icon || '🔧'} ${comp.display_name}`;
                button.title = comp.display_name;
                button.addEventListener('click', () => createNodeFromComponent(comp));
                coreContainer.appendChild(button);
            });
        }
        
        // Populate display components
        const displayContainer = document.getElementById('displayComponents');
        if (displayContainer) {
            displayContainer.innerHTML = '';
            displayComponents.forEach(comp => {
                const button = document.createElement('button');
                button.className = 'component-btn';
                button.textContent = `${comp.icon || '📺'} ${comp.display_name}`;
                button.title = comp.display_name;
                button.addEventListener('click', () => createNodeFromComponent(comp));
                displayContainer.appendChild(button);
            });
        }
        
        // Populate video model components
        const videoContainer = document.getElementById('videoModelComponents');
        if (videoContainer) {
            videoContainer.innerHTML = '';
            videoModelComponents.forEach(comp => {
                const button = document.createElement('button');
                button.className = 'component-btn';
                button.textContent = `${comp.icon || '🎬'} ${comp.display_name}`;
                button.title = comp.display_name;
                button.addEventListener('click', () => createNodeFromComponent(comp));
                videoContainer.appendChild(button);
            });
        }
        
        // Populate advanced components
        const advancedContainer = document.getElementById('advancedComponents');
        if (advancedContainer) {
            advancedContainer.innerHTML = '';
            advancedComponents.forEach(comp => {
                const button = document.createElement('button');
                button.className = 'component-btn';
                button.textContent = `${comp.icon || '🔧'} ${comp.display_name}`;
                button.title = comp.display_name;
                button.addEventListener('click', () => createNodeFromComponent(comp));
                advancedContainer.appendChild(button);
            });
        }
        
        console.log(`✅ Component toolbar populated: ${coreImageModels.length} image models, ${videoModelComponents.length} video models, ${displayComponents.length} display, ${advancedComponents.length} advanced`);
        
    } catch (error) {
        console.error('Failed to setup component toolbar:', error);
    }
}

function createNodeFromComponent(comp) {
    console.log(`Creating node: ${comp.display_name}`);
    
    const node = LiteGraph.createNode(`sdforms/${comp.id}`);
    if (node) {
        // Position node at center of canvas with some randomness
        const canvasRect = canvas.canvas.getBoundingClientRect();
        const centerX = (canvasRect.width / 2 - canvas.ds.offset[0]) / canvas.ds.scale;
        const centerY = (canvasRect.height / 2 - canvas.ds.offset[1]) / canvas.ds.scale;
        
        node.pos = [
            centerX + (Math.random() - 0.5) * 100,
            centerY + (Math.random() - 0.5) * 100
        ];
        
        graph.add(node);
        console.log(`✅ Created ${comp.display_name} node`);
        
        // Auto-connect logic based on component type
        if (comp.id === 'sampler') {
            autoConnectSamplerToModel(node);
        } else if (comp.id === 'image') {
            autoConnectImageToSampler(node);
        } else if (comp.id === 'media_display') {
            autoConnectMediaDisplayToSampler(node);
        } else if (comp.id === 'output') {
            autoConnectOutputToSampler(node); // Output connects directly to Sampler
        } else if (comp.id === 'upscaler') {
            autoConnectUpscalerToImage(node);
        } else if (comp.id === 'animatediff') {
            autoConnectAnimateDiffToModel(node);
        } else if (comp.id === 'ltx_video') {
            // LTX Video is standalone, no auto-connect needed
        }
        
        // Select the new node
        graph.selectNode(node);
        updateObjectInspector(node);
        updateModelDisplayNames();
    } else {
        console.error(`Failed to create node: ${comp.id}`);
    }
}

function autoConnectSamplerToModel(samplerNode) {
    // Find the closest model node
    const modelNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sdxl_model' || n.type === 'sdforms/flux_model'
    );
    
    if (modelNodes.length > 0) {
        const closestModel = findClosestNode(samplerNode, modelNodes);
        if (closestModel && closestModel.outputs && samplerNode.inputs) {
            try {
                // Connect model pipeline to sampler
                if (closestModel.outputs.length > 0 && samplerNode.inputs.length > 0) {
                    closestModel.connect(0, samplerNode, 0); // pipeline
                    console.log(`✅ Auto-connected ${closestModel.title} → ${samplerNode.title} (pipeline)`);
                }
                
                // Connect conditioning if available
                if (closestModel.outputs.length > 1 && samplerNode.inputs.length > 1) {
                    closestModel.connect(1, samplerNode, 1); // conditioning
                    console.log(`✅ Auto-connected ${closestModel.title} → ${samplerNode.title} (conditioning)`);
                }
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
    }
}

function autoConnectMediaDisplayToSampler(displayNode) {
    // Find the closest sampler node
    const samplerNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sampler'
    );
    
    if (samplerNodes.length > 0) {
        const closestSampler = findClosestNode(displayNode, samplerNodes);
        if (closestSampler && closestSampler.outputs && displayNode.inputs) {
            try {
                closestSampler.connect(0, displayNode, 0); // image
                console.log(`✅ Auto-connected ${closestSampler.title} → ${displayNode.title}`);
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
    }
}

function autoConnectImageToSampler(imageNode) {
    // Find the closest sampler node
    const samplerNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sampler'
    );
    
    if (samplerNodes.length > 0) {
        const closestSampler = findClosestNode(imageNode, samplerNodes);
        if (closestSampler && closestSampler.outputs && imageNode.inputs) {
            try {
                closestSampler.connect(0, imageNode, 0); // image
                console.log(`✅ Auto-connected ${closestSampler.title} → ${imageNode.title}`);
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
    }
}

function autoConnectOutputToSampler(outputNode) {
    // Output connects directly to Sampler (not through media display)
    const samplerNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sampler'
    );
    
    if (samplerNodes.length > 0) {
        const closestSampler = findClosestNode(outputNode, samplerNodes);
        if (closestSampler && closestSampler.outputs && outputNode.inputs) {
            try {
                closestSampler.connect(0, outputNode, 0); // image
                console.log(`✅ Auto-connected ${closestSampler.title} → ${outputNode.title} (direct)`);
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
    }
}

function autoConnectUpscalerToImage(upscalerNode) {
    // Upscaler connects to Image component or Sampler
    let sourceNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/image'
    );
    
    // If no Image component, connect to Sampler
    if (sourceNodes.length === 0) {
        sourceNodes = graph._nodes.filter(n => 
            n.type === 'sdforms/sampler'
        );
    }
    
    if (sourceNodes.length > 0) {
        const closestSource = findClosestNode(upscalerNode, sourceNodes);
        if (closestSource && closestSource.outputs && upscalerNode.inputs) {
            try {
                closestSource.connect(0, upscalerNode, 0); // image
                console.log(`✅ Auto-connected ${closestSource.title} → ${upscalerNode.title}`);
            } catch (error) {
                console.warn('Auto-connect failed:', error);
            }
        }
    }
}

function autoConnectAnimateDiffToModel(animateDiffNode) {
    // AnimateDiff connects to Model for pipeline and conditioning
    const modelNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sdxl_model' || n.type === 'sdforms/flux_model'
    );
    
    if (modelNodes.length > 0) {
        const closestModel = findClosestNode(animateDiffNode, modelNodes);
        if (closestModel && closestModel.outputs && animateDiffNode.inputs) {
            try {
                // Connect model pipeline to AnimateDiff
                if (closestModel.outputs.length > 0 && animateDiffNode.inputs.length > 0) {
                    closestModel.connect(0, animateDiffNode, 0); // pipeline
                    console.log(`✅ Auto-connected ${closestModel.title} → ${animateDiffNode.title} (pipeline)`);
                }
                
                // Connect conditioning if available
                if (closestModel.outputs.length > 1 && animateDiffNode.inputs.length > 1) {
                    closestModel.connect(1, animateDiffNode, 1); // conditioning
                    console.log(`✅ Auto-connected ${closestModel.title} → ${animateDiffNode.title} (conditioning)`);
                }
            } catch (error) {
                console.warn('AnimateDiff auto-connect failed:', error);
            }
        }
    }
}

function findClosestNode(targetNode, candidateNodes) {
    if (candidateNodes.length === 0) return null;
    
    let closest = candidateNodes[0];
    let minDistance = getNodeDistance(targetNode, closest);
    
    for (let i = 1; i < candidateNodes.length; i++) {
        const distance = getNodeDistance(targetNode, candidateNodes[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closest = candidateNodes[i];
        }
    }
    
    return closest;
}

function getNodeDistance(node1, node2) {
    const dx = node1.pos[0] - node2.pos[0];
    const dy = node1.pos[1] - node2.pos[1];
    return Math.sqrt(dx * dx + dy * dy);
}

function updateModelDisplayNames() {
    // Update model nodes to display their selected model type
    const modelNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sdxl_model' || n.type === 'sdforms/flux_model'
    );
    
    modelNodes.forEach(node => {
        if (node.widgets) {
            const fluxPresetWidget = node.widgets.find(w => w.name === 'flux_preset');
            const checkpointWidget = node.widgets.find(w => w.name === 'checkpoint');
            const checkpointTypeWidget = node.widgets.find(w => w.name === 'checkpoint_type');
            
            let modelName = 'Model';
            
            if (checkpointTypeWidget && checkpointTypeWidget.value === 'preset' && fluxPresetWidget) {
                modelName = `Model (${fluxPresetWidget.value})`;
            } else if (checkpointWidget) {
                const shortName = checkpointWidget.value.split('/').pop().replace('-', ' ');
                modelName = `Model (${shortName})`;
            }
            
            if (node.title !== modelName) {
                node.title = modelName;
                console.log(`📋 Updated model display: ${modelName}`);
            }
        }
    });
    
    // Force redraw
    if (canvas) {
        canvas.setDirty(true, true);
    }
}

function createBasicWorkflow() {
    console.log('🚀 Creating basic workflow based on existing model...');
    
    // Check if there are existing model nodes
    const existingModelNodes = graph._nodes.filter(n => 
        n.type === 'sdforms/sdxl_model' ||
        n.type === 'sdforms/flux_model' ||
        n.type === 'sdforms/hidream_i1' ||
        n.type === 'sdforms/lumina_control' ||
        n.type === 'sdforms/wan_vace' ||
        n.type === 'sdforms/animatediff' ||
        n.type === 'sdforms/ltx_video' ||
        n.type === 'sdforms/chroma' ||
        n.type === 'sdforms/sd15' ||
        n.type === 'sdforms/sd35' ||
        n.type === 'sdforms/omnigen'
    );
    
    if (existingModelNodes.length === 0) {
        alert('Please add a model component first, then click "Basic Workflow" to complete the pipeline.');
        return;
    }
    
    // Detect model type from existing nodes
    const modelNode = existingModelNodes[0];
    let workflowType = detectWorkflowType(modelNode);
    
    console.log(`📋 Detected workflow type: ${workflowType} from ${modelNode.title}`);
    
    // Create workflow based on detected type
    if (workflowType === 'image') {
        createImageWorkflowAroundModel(modelNode);
    } else if (workflowType === 'video') {
        createVideoWorkflowAroundModel(modelNode);
    } else if (workflowType === 'animatediff') {
        createAnimatedDiffWorkflowAroundModel(modelNode);
    } else if (workflowType === 'direct_output') {
        createDirectOutputWorkflowAroundModel(modelNode);
    }
    
    console.log(`✅ Basic workflow created for: ${workflowType}`);
}

function detectWorkflowType(modelNode) {
    const nodeType = modelNode.type;
    
    // Check for TRUE video-specific models (based on actual properties)
    if (nodeType === 'sdforms/wan_vace' || 
        nodeType === 'sdforms/animatediff' || 
        nodeType === 'sdforms/ltx_video') {
        return 'video';
    }
    
    // Check for direct-output models (don't need sampler)
    if (nodeType === 'sdforms/hidream_i1' ||
        nodeType === 'sdforms/lumina_control' ||
        nodeType === 'sdforms/omnigen') {
        return 'direct_output';
    }
    
    // Check for standard diffusion models (need sampler)
    if (nodeType === 'sdforms/sdxl_model' ||
        nodeType === 'sdforms/flux_model' ||
        nodeType === 'sdforms/chroma' ||
        nodeType === 'sdforms/sd15' ||
        nodeType === 'sdforms/sd35') {
        
        // For standard SD models, check if user wants AnimateDiff
        if (nodeType === 'sdforms/sdxl_model') {
            if (modelNode.widgets) {
                const checkpointWidget = modelNode.widgets.find(w => w.name === 'checkpoint');
                
                // If using SD 1.5 model, ask user for AnimateDiff
                if (checkpointWidget && checkpointWidget.value.includes('stable-diffusion-v1-5')) {
                    const useAnimateDiff = confirm('This is a Stable Diffusion 1.5 model. Would you like to create an AnimateDiff video workflow instead of image workflow?');
                    return useAnimateDiff ? 'animatediff' : 'image';
                }
            }
        }
        
        // All other image models default to image generation
        return 'image';
    }
    
    return 'image'; // Default fallback
}

function createImageWorkflowAroundModel(modelNode) {
    console.log('🖼️ Creating image generation workflow...');
    
    // Check if we already have the required components
    const existingSampler = graph._nodes.find(n => n.type === 'sdforms/sampler');
    const existingImage = graph._nodes.find(n => n.type === 'sdforms/image');
    const existingOutput = graph._nodes.find(n => n.type === 'sdforms/output');
    
    // Calculate positions relative to model
    const baseX = modelNode.pos[0];
    const baseY = modelNode.pos[1];
    
    let samplerNode = existingSampler;
    if (!samplerNode) {
        samplerNode = LiteGraph.createNode("sdforms/sampler");
        if (samplerNode) {
            samplerNode.pos = [baseX + 300, baseY];
            graph.add(samplerNode);
            console.log('➕ Added Sampler component');
        }
    }
    
    let imageNode = existingImage;
    if (!imageNode) {
        imageNode = LiteGraph.createNode("sdforms/image");
        if (imageNode) {
            imageNode.pos = [baseX + 600, baseY];
            graph.add(imageNode);
            console.log('➕ Added Image component');
        }
    }
    
    let outputNode = existingOutput;
    if (!outputNode) {
        outputNode = LiteGraph.createNode("sdforms/output");
        if (outputNode) {
            outputNode.pos = [baseX + 600, baseY + 150];
            graph.add(outputNode);
            console.log('➕ Added Output component');
        }
    }
    
    // Auto-connect the workflow
    setTimeout(() => {
        try {
            // Model → Sampler
            if (modelNode && samplerNode && modelNode.outputs && samplerNode.inputs) {
                if (!isAlreadyConnected(modelNode, 0, samplerNode, 0)) {
                    modelNode.connect(0, samplerNode, 0); // pipeline
                    console.log('🔗 Connected Model → Sampler (pipeline)');
                }
                if (modelNode.outputs.length > 1 && samplerNode.inputs.length > 1) {
                    if (!isAlreadyConnected(modelNode, 1, samplerNode, 1)) {
                        modelNode.connect(1, samplerNode, 1); // conditioning
                        console.log('🔗 Connected Model → Sampler (conditioning)');
                    }
                }
            }
            
            // Sampler → Image
            if (samplerNode && imageNode && samplerNode.outputs && imageNode.inputs) {
                if (!isAlreadyConnected(samplerNode, 0, imageNode, 0)) {
                    samplerNode.connect(0, imageNode, 0);
                    console.log('🔗 Connected Sampler → Image');
                }
            }
            
            // Sampler → Output (direct connection)
            if (samplerNode && outputNode && samplerNode.outputs && outputNode.inputs) {
                if (!isAlreadyConnected(samplerNode, 0, outputNode, 0)) {
                    samplerNode.connect(0, outputNode, 0);
                    console.log('🔗 Connected Sampler → Output');
                }
            }
            
            console.log('✅ Image workflow connected: Model → Sampler → Image + Output');
        } catch (error) {
            console.warn('Connection failed:', error);
        }
    }, 100);
}

function createVideoWorkflowAroundModel(modelNode) {
    console.log('🎬 Creating video generation workflow...');
    
    // For direct video models (HiDream, WAN-VACE, LTX), just add display and output
    const existingDisplay = graph._nodes.find(n => n.type === 'sdforms/media_display');
    const existingOutput = graph._nodes.find(n => n.type === 'sdforms/output');
    
    const baseX = modelNode.pos[0];
    const baseY = modelNode.pos[1];
    
    let displayNode = existingDisplay;
    if (!displayNode) {
        displayNode = LiteGraph.createNode("sdforms/media_display");
        if (displayNode) {
            displayNode.pos = [baseX + 400, baseY];
            graph.add(displayNode);
            console.log('➕ Added MediaDisplay component');
        }
    }
    
    let outputNode = existingOutput;
    if (!outputNode) {
        outputNode = LiteGraph.createNode("sdforms/output");
        if (outputNode) {
            outputNode.pos = [baseX + 700, baseY];
            graph.add(outputNode);
            console.log('➕ Added Output component');
        }
    }
    
    // Auto-connect
    setTimeout(() => {
        try {
            // Video Model → MediaDisplay
            if (modelNode && displayNode && modelNode.outputs && displayNode.inputs) {
                if (!isAlreadyConnected(modelNode, 0, displayNode, 1)) {
                    modelNode.connect(0, displayNode, 1); // video input
                    console.log('🔗 Connected VideoModel → MediaDisplay');
                }
            }
            
            // Video Model → Output (direct)
            if (modelNode && outputNode && modelNode.outputs && outputNode.inputs) {
                if (!isAlreadyConnected(modelNode, 0, outputNode, 0)) {
                    modelNode.connect(0, outputNode, 0);
                    console.log('🔗 Connected VideoModel → Output');
                }
            }
            
            console.log('✅ Video workflow connected');
        } catch (error) {
            console.warn('Video connection failed:', error);
        }
    }, 100);
}

function createAnimatedDiffWorkflowAroundModel(modelNode) {
    console.log('🎞️ Creating AnimateDiff workflow...');
    
    // Model → AnimateDiff → MediaDisplay + Output
    const existingAnimateDiff = graph._nodes.find(n => n.type === 'sdforms/animatediff');
    const existingDisplay = graph._nodes.find(n => n.type === 'sdforms/media_display');
    const existingOutput = graph._nodes.find(n => n.type === 'sdforms/output');
    
    const baseX = modelNode.pos[0];
    const baseY = modelNode.pos[1];
    
    let animateDiffNode = existingAnimateDiff;
    if (!animateDiffNode) {
        animateDiffNode = LiteGraph.createNode("sdforms/animatediff");
        if (animateDiffNode) {
            animateDiffNode.pos = [baseX + 300, baseY];
            graph.add(animateDiffNode);
            console.log('➕ Added AnimateDiff component');
        }
    }
    
    let displayNode = existingDisplay;
    if (!displayNode) {
        displayNode = LiteGraph.createNode("sdforms/media_display");
        if (displayNode) {
            displayNode.pos = [baseX + 600, baseY];
            graph.add(displayNode);
            console.log('➕ Added MediaDisplay component');
        }
    }
    
    let outputNode = existingOutput;
    if (!outputNode) {
        outputNode = LiteGraph.createNode("sdforms/output");
        if (outputNode) {
            outputNode.pos = [baseX + 600, baseY + 150];
            graph.add(outputNode);
            console.log('➕ Added Output component');
        }
    }
    
    // Auto-connect
    setTimeout(() => {
        try {
            // Model → AnimateDiff
            if (modelNode && animateDiffNode && modelNode.outputs && animateDiffNode.inputs) {
                if (!isAlreadyConnected(modelNode, 0, animateDiffNode, 0)) {
                    modelNode.connect(0, animateDiffNode, 0); // pipeline
                    console.log('🔗 Connected Model → AnimateDiff (pipeline)');
                }
                if (modelNode.outputs.length > 1 && animateDiffNode.inputs.length > 1) {
                    if (!isAlreadyConnected(modelNode, 1, animateDiffNode, 1)) {
                        modelNode.connect(1, animateDiffNode, 1); // conditioning
                        console.log('🔗 Connected Model → AnimateDiff (conditioning)');
                    }
                }
            }
            
            // AnimateDiff → MediaDisplay
            if (animateDiffNode && displayNode && animateDiffNode.outputs && displayNode.inputs) {
                if (!isAlreadyConnected(animateDiffNode, 0, displayNode, 1)) {
                    animateDiffNode.connect(0, displayNode, 1); // video
                    console.log('🔗 Connected AnimateDiff → MediaDisplay');
                }
            }
            
            // AnimateDiff → Output
            if (animateDiffNode && outputNode && animateDiffNode.outputs && outputNode.inputs) {
                if (!isAlreadyConnected(animateDiffNode, 0, outputNode, 0)) {
                    animateDiffNode.connect(0, outputNode, 0);
                    console.log('🔗 Connected AnimateDiff → Output');
                }
            }
            
            console.log('✅ AnimateDiff workflow connected: Model → AnimateDiff → MediaDisplay + Output');
        } catch (error) {
            console.warn('AnimateDiff connection failed:', error);
        }
    }, 100);
}

function createDirectOutputWorkflowAroundModel(modelNode) {
    console.log('🎯 Creating direct output workflow (no sampler needed)...');
    
    // Direct output models (OmniGen, HiDream, Lumina) output images directly
    // They only need Image display and Output components
    
    // Check if we already have the required components
    const existingImage = graph._nodes.find(n => n.type === 'sdforms/image');
    const existingOutput = graph._nodes.find(n => n.type === 'sdforms/output');
    
    // Calculate positions relative to model
    const baseX = modelNode.pos[0];
    const baseY = modelNode.pos[1];
    
    let imageNode = existingImage;
    if (!imageNode) {
        imageNode = LiteGraph.createNode("sdforms/image");
        if (imageNode) {
            imageNode.pos = [baseX + 300, baseY];
            graph.add(imageNode);
            console.log('➕ Added Image component');
        }
    }
    
    let outputNode = existingOutput;
    if (!outputNode) {
        outputNode = LiteGraph.createNode("sdforms/output");
        if (outputNode) {
            outputNode.pos = [baseX + 300, baseY + 150];
            graph.add(outputNode);
            console.log('➕ Added Output component');
        }
    }
    
    // Auto-connect the workflow
    setTimeout(() => {
        try {
            // Model → Image (direct images output)
            if (modelNode && imageNode && modelNode.outputs && imageNode.inputs) {
                // Find the "images" output port
                let imagesOutputIdx = -1;
                for (let i = 0; i < modelNode.outputs.length; i++) {
                    if (modelNode.outputs[i].name === 'images') {
                        imagesOutputIdx = i;
                        break;
                    }
                }
                
                if (imagesOutputIdx >= 0 && !isAlreadyConnected(modelNode, imagesOutputIdx, imageNode, 0)) {
                    modelNode.connect(imagesOutputIdx, imageNode, 0);
                    console.log('🔗 Connected Model → Image (images)');
                }
            }
            
            // Model → Output (direct connection)
            if (modelNode && outputNode && modelNode.outputs && outputNode.inputs) {
                // Find the "images" output port
                let imagesOutputIdx = -1;
                for (let i = 0; i < modelNode.outputs.length; i++) {
                    if (modelNode.outputs[i].name === 'images') {
                        imagesOutputIdx = i;
                        break;
                    }
                }
                
                if (imagesOutputIdx >= 0 && !isAlreadyConnected(modelNode, imagesOutputIdx, outputNode, 0)) {
                    modelNode.connect(imagesOutputIdx, outputNode, 0);
                    console.log('🔗 Connected Model → Output (images)');
                }
            }
            
            console.log('✅ Direct output workflow connected: Model → Image + Output');
        } catch (error) {
            console.warn('Connection failed:', error);
        }
    }, 100);
}

function isAlreadyConnected(fromNode, fromSlot, toNode, toSlot) {
    // Check if nodes are already connected
    if (!fromNode.outputs || !fromNode.outputs[fromSlot]) return false;
    
    const output = fromNode.outputs[fromSlot];
    if (!output.links) return false;
    
    for (let linkId of output.links) {
        const link = graph.links[linkId];
        if (link && link.target_id === toNode.id && link.target_slot === toSlot) {
            return true;
        }
    }
    
    return false;
}

function createFluxStandardForm() {
    // Create Model → Sampler → Image → Output workflow (proper flow)
    const modelNode = LiteGraph.createNode("sdforms/flux_model");
    if (modelNode) {
        modelNode.pos = [50, 100];
        modelNode.title = "Model (flux1-schnell)";
        graph.add(modelNode);
    }
    
    const samplerNode = LiteGraph.createNode("sdforms/sampler");
    if (samplerNode) {
        samplerNode.pos = [350, 100];
        graph.add(samplerNode);
    }
    
    const imageNode = LiteGraph.createNode("sdforms/image");
    if (imageNode) {
        imageNode.pos = [650, 100];
        graph.add(imageNode);
    }
    
    const outputNode = LiteGraph.createNode("sdforms/output");
    if (outputNode) {
        outputNode.pos = [950, 100];
        graph.add(outputNode);
    }
    
    // Connect them
    setTimeout(() => {
        try {
            if (modelNode && samplerNode) {
                modelNode.connect(0, samplerNode, 0); // pipeline
                if (modelNode.outputs.length > 1) {
                    modelNode.connect(1, samplerNode, 1); // conditioning
                }
            }
            if (samplerNode && imageNode) {
                samplerNode.connect(0, imageNode, 0); // image to Image component
            }
            if (samplerNode && outputNode) {
                samplerNode.connect(0, outputNode, 0); // image directly to Output
            }
            console.log('✅ Standard form connected: Model → Sampler → Image + Output');
        } catch (error) {
            console.warn('Standard form connection failed:', error);
        }
    }, 100);
}

function createVideoStandardForm() {
    // Create Model → AnimateDiff → MediaDisplay → Output workflow
    const modelNode = LiteGraph.createNode("sdforms/flux_model");
    if (modelNode) {
        modelNode.pos = [50, 100];
        modelNode.title = "Model (SD 1.5)";
        graph.add(modelNode);
    }
    
    const animateDiffNode = LiteGraph.createNode("sdforms/animatediff");
    if (animateDiffNode) {
        animateDiffNode.pos = [350, 100];
        graph.add(animateDiffNode);
    }
    
    const displayNode = LiteGraph.createNode("sdforms/media_display");
    if (displayNode) {
        displayNode.pos = [650, 100];
        graph.add(displayNode);
    }
    
    const outputNode = LiteGraph.createNode("sdforms/output");
    if (outputNode) {
        outputNode.pos = [950, 100];
        graph.add(outputNode);
    }
    
    // Connect them
    setTimeout(() => {
        try {
            if (modelNode && animateDiffNode) {
                modelNode.connect(0, animateDiffNode, 0); // pipeline
                if (modelNode.outputs.length > 1) {
                    modelNode.connect(1, animateDiffNode, 1); // conditioning
                }
            }
            if (animateDiffNode && displayNode) {
                animateDiffNode.connect(0, displayNode, 1); // video to video input
            }
            if (animateDiffNode && outputNode) {
                animateDiffNode.connect(0, outputNode, 0); // video to output
            }
            console.log('✅ Video standard form connected: Model → AnimateDiff → MediaDisplay + Output');
        } catch (error) {
            console.warn('Video standard form connection failed:', error);
        }
    }, 100);
}

function setupNodeSelection() {
    console.log('Setting up node selection...');
    
    // Override LiteGraph's selection handling
    const originalSelectNode = graph.selectNode;
    graph.selectNode = function(node) {
        // Call original method
        if (originalSelectNode) {
            originalSelectNode.call(this, node);
        }
        
        // Update our selection state
        selectedNode = node;
        updateObjectInspector(node);
    };
    
    // Handle canvas clicks to deselect
    canvas.canvas.addEventListener('click', (e) => {
        setTimeout(() => {
            if (!graph.selected_nodes || graph.selected_nodes.length === 0) {
                selectedNode = null;
                updateObjectInspector(null);
            }
        }, 10);
    });
}

function updateObjectInspector(node) {
    const inspector = document.getElementById('inspectorContent');
    if (!inspector) return;
    
    if (!node) {
        inspector.innerHTML = '<div class="no-selection">Select a node to view properties</div>';
        return;
    }
    
    console.log(`Updating object inspector for: ${node.title}`);
    
    let html = `<div class="property-group">
        <div class="property-label">Node Type</div>
        <div style="color: #4CAF50; font-weight: bold;">${node.title}</div>
    </div>`;
    
    // Group properties by category
    const propsByCategory = {};
    
    if (node.widgets) {
        node.widgets.forEach(widget => {
            const category = 'Properties'; // Default category
            if (!propsByCategory[category]) {
                propsByCategory[category] = [];
            }
            propsByCategory[category].push(widget);
        });
    }
    
    // Render properties by category
    Object.keys(propsByCategory).forEach(category => {
        if (category !== 'Properties') {
            html += `<div class="property-category">${category}</div>`;
        }
        
        propsByCategory[category].forEach(widget => {
            html += `<div class="property-group">
                <div class="property-label">${widget.name}</div>`;
            
            if (widget.type === 'combo') {
                html += `<select class="property-select" onchange="updateNodeProperty('${widget.name}', this.value)">`;
                
                // Special handling for model-related dropdowns
                if (widget.name === 'flux_preset' && allAvailableModels.flux) {
                    // Create expandable Flux model dropdown
                    html += `<optgroup label="Configured Models">`;
                    allAvailableModels.flux.filter(m => m.configured).forEach(model => {
                        const selected = widget.value === model.id ? 'selected' : '';
                        html += `<option value="${model.id}" ${selected}>${model.name}</option>`;
                    });
                    html += `</optgroup>`;
                    
                    if (allAvailableModels.flux.filter(m => !m.configured).length > 0) {
                        html += `<optgroup label="Available Models">`;
                        allAvailableModels.flux.filter(m => !m.configured).forEach(model => {
                            const selected = widget.value === model.id ? 'selected' : '';
                            html += `<option value="${model.id}" ${selected}>${model.name} (${model.size_mb}MB)</option>`;
                        });
                        html += `</optgroup>`;
                    }
                } else if (widget.name === 'checkpoint' && organizedModels) {
                    // Create expandable checkpoint dropdown with categories
                    Object.entries(organizedModels).forEach(([category, models]) => {
                        if (models.length > 0) {
                            html += `<optgroup label="${category}">`;
                            models.forEach(model => {
                                const selected = widget.value === model.id ? 'selected' : '';
                                const displayName = model.configured ? model.name : `${model.name} (${model.size_mb}MB)`;
                                html += `<option value="${model.id}" ${selected}>${displayName}</option>`;
                            });
                            html += `</optgroup>`;
                        }
                    });
                } else {
                    // Standard dropdown
                    (widget.options?.values || ['option1', 'option2']).forEach(value => {
                        const selected = widget.value === value ? 'selected' : '';
                        html += `<option value="${value}" ${selected}>${value}</option>`;
                    });
                }
                html += `</select>`;
            } else if (widget.type === 'toggle') {
                const checked = widget.value ? 'checked' : '';
                html += `<label><input type="checkbox" class="property-checkbox" ${checked} onchange="updateNodeProperty('${widget.name}', this.checked)"> ${widget.value ? 'Enabled' : 'Disabled'}</label>`;
            } else if (widget.type === 'number') {
                html += `<input type="number" class="property-input" value="${widget.value}" onchange="updateNodeProperty('${widget.name}', parseFloat(this.value) || 0)">`;
            } else {
                html += `<input type="text" class="property-input" value="${widget.value || ''}" onchange="updateNodeProperty('${widget.name}', this.value)">`;
            }
            
            html += `</div>`;
        });
    });
    
    inspector.innerHTML = html;
}

function updateNodeProperty(propertyName, value) {
    if (!selectedNode) return;
    
    console.log(`Updating property ${propertyName} = ${value}`);
    
    // Update the widget value
    if (selectedNode.widgets) {
        const widget = selectedNode.widgets.find(w => w.name === propertyName);
        if (widget) {
            widget.value = value;
            // Trigger the widget's callback if it exists
            if (widget.callback) {
                widget.callback(value);
            }
        }
    }
    
    // Update the node's properties
    if (!selectedNode.properties) {
        selectedNode.properties = {};
    }
    selectedNode.properties[propertyName] = value;
    
    // Update model display names if this is a model property
    if (['flux_preset', 'checkpoint', 'checkpoint_type'].includes(propertyName)) {
        updateModelDisplayNames();
    }
    
    // Redraw the node
    if (canvas) {
        canvas.setDirty(true, true);
    }
}

// Make function global for onclick handlers
window.updateNodeProperty = updateNodeProperty;

async function loadAllAvailableModels() {
    console.log('🔍 Loading all available models...');
    
    try {
        const response = await fetch('/api/models');
        const modelData = await response.json();
        
        allAvailableModels = modelData.all_models || {};
        organizedModels = modelData.organized_models || {};
        
        console.log(`✅ Loaded models:`, {
            flux: allAvailableModels.flux?.length || 0,
            sdxl: allAvailableModels.sdxl?.length || 0,
            sd15: allAvailableModels.sd15?.length || 0,
            other: allAvailableModels.other?.length || 0
        });
        
        // Update any existing model dropdowns
        updateModelDropdowns();
        
    } catch (error) {
        console.error('❌ Failed to load models:', error);
    }
}

function updateModelDropdowns() {
    console.log('🔄 Updating model dropdowns with expanded options...');
    
    // Find all nodes with model widgets
    if (graph && graph._nodes) {
        graph._nodes.forEach(node => {
            if (node.widgets) {
                node.widgets.forEach(widget => {
                    if (widget.name === 'flux_preset' && allAvailableModels.flux) {
                        // Update flux preset dropdown with all available Flux models
                        const fluxChoices = allAvailableModels.flux.map(model => model.id);
                        widget.options = { values: fluxChoices };
                        console.log(`📝 Updated flux_preset with ${fluxChoices.length} options`);
                    } else if (widget.name === 'checkpoint' && organizedModels) {
                        // Create expandable choices for checkpoint dropdown
                        const allModelChoices = [];
                        
                        // Add organized model categories
                        Object.entries(organizedModels).forEach(([category, models]) => {
                            if (models.length > 0) {
                                allModelChoices.push(`--- ${category} ---`);
                                models.forEach(model => {
                                    allModelChoices.push(model.id);
                                });
                            }
                        });
                        
                        widget.options = { values: allModelChoices };
                        console.log(`📝 Updated checkpoint with ${allModelChoices.length} organized options`);
                    }
                });
            }
        });
        
        // Force canvas redraw
        if (canvas) {
            canvas.setDirty(true, true);
        }
    }
}

function getModelDisplayName(modelId, category = 'flux') {
    if (!allAvailableModels[category]) return modelId;
    
    const model = allAvailableModels[category].find(m => m.id === modelId);
    return model ? model.name : modelId;
}

function getModelInfo(modelId, category = 'flux') {
    if (!allAvailableModels[category]) return null;
    
    return allAvailableModels[category].find(m => m.id === modelId);
}

function setupDraggableInspector() {
    const inspector = document.getElementById('objectInspector');
    if (!inspector) return;
    
    let isDragging = false;
    let dragOffsetX = 0;
    let dragOffsetY = 0;
    
    inspector.addEventListener('mousedown', (e) => {
        // Only start dragging if clicking on the header or empty space, not on controls
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        isDragging = true;
        inspector.classList.add('dragging');
        
        const rect = inspector.getBoundingClientRect();
        dragOffsetX = e.clientX - rect.left;
        dragOffsetY = e.clientY - rect.top;
        
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        const x = e.clientX - dragOffsetX;
        const y = e.clientY - dragOffsetY;
        
        // Keep within viewport bounds
        const maxX = window.innerWidth - inspector.offsetWidth;
        const maxY = window.innerHeight - inspector.offsetHeight;
        
        inspector.style.left = Math.max(0, Math.min(x, maxX)) + 'px';
        inspector.style.top = Math.max(0, Math.min(y, maxY)) + 'px';
        inspector.style.right = 'auto';
        inspector.style.bottom = 'auto';
        
        e.preventDefault();
    });
    
    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            inspector.classList.remove('dragging');
        }
    });
    
    console.log('📦 Object inspector is now draggable');
}

console.log('App.js loaded - waiting for DOM...');