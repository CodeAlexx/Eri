/**
 * Sampler Node - Specialized LiteGraph node for SD Forms Sampler component
 * Handles generation parameters, scheduler selection, and live preview via WebSocket
 */

// Available schedulers matching sd_forms/components/standard.py
const SCHEDULERS = [
    "DPMSolverMultistep",
    "DPMSolverSinglestep", 
    "Euler",
    "EulerAncestral",
    "DDIM",
    "PNDM",
    "HeunDiscrete",
    "KDPM2Discrete",
    "KDPM2AncestralDiscrete",
    "LMSDiscrete"
];

// Common resolution presets
const RESOLUTION_PRESETS = [
    { name: "512x512 (SD 1.5)", width: 512, height: 512 },
    { name: "768x768 (SD 2.1)", width: 768, height: 768 },
    { name: "1024x1024 (SDXL/Flux)", width: 1024, height: 1024 },
    { name: "1152x896 (SDXL Landscape)", width: 1152, height: 896 },
    { name: "896x1152 (SDXL Portrait)", width: 896, height: 1152 },
    { name: "1344x768 (Wide)", width: 1344, height: 768 },
    { name: "768x1344 (Tall)", width: 768, height: 1344 }
];

/**
 * Specialized Sampler Node with generation controls and live preview
 */
class SamplerNode extends SDFormsNode {
    constructor() {
        // Create mock component info for Sampler
        const samplerComponentInfo = {
            id: 'sampler_component',
            type: 'sampler',
            display_name: 'Sampler',
            category: 'Core',
            icon: '🎲',
            input_ports: [
                { name: 'pipeline', type: 'pipeline', direction: 'INPUT' },
                { name: 'conditioning', type: 'conditioning', direction: 'INPUT' },
                { name: 'resolution', type: 'any', direction: 'INPUT', optional: true },
                { name: 'seed', type: 'any', direction: 'INPUT', optional: true }
            ],
            output_ports: [
                { name: 'images', type: 'image', direction: 'OUTPUT' },
                { name: 'latents', type: 'latent', direction: 'OUTPUT', optional: true }
            ],
            property_definitions: [] // We'll create custom widgets
        };
        
        super(samplerComponentInfo);
        
        // Custom initialization
        this.setupSamplerNode();
        
        // Preview state
        this.isGenerating = false;
        this.currentStep = 0;
        this.totalSteps = 20;
        this.previewImage = null;
        this.generationStartTime = null;
        
        // WebSocket callback for live updates
        this.wsCallback = this.handleWebSocketMessage.bind(this);
        
        // Register for WebSocket updates
        if (typeof apiService !== 'undefined') {
            apiService.addWebSocketCallback(this.wsCallback);
        }
    }
    
    setupSamplerNode() {
        // Clear default widgets and create custom ones
        this.widgets = [];
        this.properties = {};
        this.propertyValues = {};
        
        // Set node appearance
        this.title = "🎲 Sampler";
        this.size = [300, 500]; // Larger to accommodate preview
        this.color = "#e74c3c";
        this.bgcolor = "#c0392b";
        
        // Initialize properties
        this.initializeProperties();
        
        // Create custom widgets
        this.createPromptWidgets();
        this.createSamplingWidgets();
        this.createImageSizeWidgets();
        this.createAdvancedWidgets();
        this.createControlButtons();
        this.createPreviewArea();
        
        // Set up auto-configuration listener
        this.setupAutoConfigListener();
    }
    
    initializeProperties() {
        const defaults = {
            prompt: "a professional photo of a beautiful landscape at golden hour, highly detailed, photorealistic",
            negative_prompt: "",
            scheduler: "DPMSolverMultistep",
            steps: 20,
            cfg_scale: 7.5,
            width: 1024,
            height: 1024,
            seed: -1,
            batch_size: 1,
            preview_interval: 5,
            enable_preview: true
        };
        
        for (const [key, value] of Object.entries(defaults)) {
            this.properties[key] = value;
            this.propertyValues[key] = value;
        }
    }
    
    createPromptWidgets() {
        // Positive prompt - multiline text area
        this.promptWidget = this.addWidget(
            "textarea",
            "Prompt",
            this.properties.prompt,
            (value) => this.onPropertyChanged("prompt", value),
            { multiline: true, height: 60 }
        );
        this.promptWidget.property_name = "prompt";
        
        // Negative prompt - multiline text area
        this.negativePromptWidget = this.addWidget(
            "textarea",
            "Negative Prompt",
            this.properties.negative_prompt,
            (value) => this.onPropertyChanged("negative_prompt", value),
            { multiline: true, height: 40 }
        );
        this.negativePromptWidget.property_name = "negative_prompt";
    }
    
    createSamplingWidgets() {
        // Scheduler selection
        this.schedulerWidget = this.addWidget(
            "combo",
            "Scheduler",
            this.properties.scheduler,
            (value) => this.onPropertyChanged("scheduler", value),
            { values: SCHEDULERS }
        );
        this.schedulerWidget.property_name = "scheduler";
        
        // Steps slider
        this.stepsWidget = this.addWidget(
            "slider",
            "Steps",
            this.properties.steps,
            (value) => this.onPropertyChanged("steps", Math.round(value)),
            { min: 1, max: 150, step: 1 }
        );
        this.stepsWidget.property_name = "steps";
        
        // CFG Scale slider
        this.cfgScaleWidget = this.addWidget(
            "slider",
            "CFG Scale",
            this.properties.cfg_scale,
            (value) => this.onPropertyChanged("cfg_scale", Math.round(value * 10) / 10),
            { min: 1.0, max: 20.0, step: 0.1, precision: 1 }
        );
        this.cfgScaleWidget.property_name = "cfg_scale";
    }
    
    createImageSizeWidgets() {
        // Resolution preset dropdown
        this.resolutionPresetWidget = this.addWidget(
            "combo",
            "Resolution Preset",
            "1024x1024 (SDXL/Flux)",
            (value) => this.applyResolutionPreset(value),
            { values: RESOLUTION_PRESETS.map(p => p.name) }
        );
        
        // Width slider
        this.widthWidget = this.addWidget(
            "slider",
            "Width",
            this.properties.width,
            (value) => this.onPropertyChanged("width", Math.round(value / 64) * 64),
            { min: 64, max: 2048, step: 64 }
        );
        this.widthWidget.property_name = "width";
        
        // Height slider
        this.heightWidget = this.addWidget(
            "slider", 
            "Height",
            this.properties.height,
            (value) => this.onPropertyChanged("height", Math.round(value / 64) * 64),
            { min: 64, max: 2048, step: 64 }
        );
        this.heightWidget.property_name = "height";
    }
    
    createAdvancedWidgets() {
        // Seed input (-1 for random)
        this.seedWidget = this.addWidget(
            "number",
            "Seed (-1 = random)",
            this.properties.seed,
            (value) => this.onPropertyChanged("seed", Math.floor(value)),
            { min: -1, max: 2147483647, step: 1 }
        );
        this.seedWidget.property_name = "seed";
        
        // Batch size
        this.batchSizeWidget = this.addWidget(
            "slider",
            "Batch Size",
            this.properties.batch_size,
            (value) => this.onPropertyChanged("batch_size", Math.round(value)),
            { min: 1, max: 8, step: 1 }
        );
        this.batchSizeWidget.property_name = "batch_size";
        
        // Preview settings
        this.enablePreviewWidget = this.addWidget(
            "toggle",
            "Enable Preview",
            this.properties.enable_preview,
            (value) => this.onPropertyChanged("enable_preview", value)
        );
        this.enablePreviewWidget.property_name = "enable_preview";
        
        this.previewIntervalWidget = this.addWidget(
            "slider",
            "Preview Every N Steps",
            this.properties.preview_interval,
            (value) => this.onPropertyChanged("preview_interval", Math.round(value)),
            { min: 1, max: 50, step: 1 }
        );
        this.previewIntervalWidget.property_name = "preview_interval";
    }
    
    createControlButtons() {
        // Generate button
        this.generateButton = this.addWidget(
            "button",
            "🎨 Generate",
            null,
            () => this.startGeneration()
        );
        
        // Stop button
        this.stopButton = this.addWidget(
            "button",
            "⏹️ Stop",
            null,
            () => this.stopGeneration()
        );
        this.stopButton.hidden = true;
        
        // Random seed button
        this.randomSeedButton = this.addWidget(
            "button",
            "🎲 Random Seed",
            null,
            () => this.generateRandomSeed()
        );
    }
    
    createPreviewArea() {
        // Progress display
        this.progressWidget = this.addWidget(
            "text",
            "Progress",
            "Ready",
            null,
            { disabled: true }
        );
        
        // Canvas for preview image will be drawn in onDrawForeground
        this.previewSize = 150; // Preview image size
    }
    
    setupAutoConfigListener() {
        // Listen for auto-configuration from connected Model nodes
        this.addPropertyChangeHandler("__auto_config", (config) => {
            if (config && typeof config === 'object') {
                console.log('🔧 Applying auto-configuration:', config);
                
                if (config.steps) this.setProperty('steps', config.steps);
                if (config.cfg) this.setProperty('cfg_scale', config.cfg);
                if (config.optimal_size) {
                    this.setProperty('width', config.optimal_size[0]);
                    this.setProperty('height', config.optimal_size[1]);
                }
                if (config.sampler) this.setProperty('scheduler', config.sampler);
            }
        });
    }
    
    applyResolutionPreset(presetName) {
        const preset = RESOLUTION_PRESETS.find(p => p.name === presetName);
        if (preset) {
            this.setProperty('width', preset.width);
            this.setProperty('height', preset.height);
        }
    }
    
    generateRandomSeed() {
        const randomSeed = Math.floor(Math.random() * 2147483647);
        this.setProperty('seed', randomSeed);
    }
    
    async startGeneration() {
        if (this.isGenerating) {
            console.log('Generation already in progress');
            return;
        }
        
        try {
            // Prepare execution request
            const workflow = this.createWorkflowFromGraph();
            const executionRequest = new ExecutionRequest(workflow, {
                preview_enabled: this.properties.enable_preview,
                preview_interval: this.properties.preview_interval
            });
            
            // Update UI
            this.isGenerating = true;
            this.generationStartTime = Date.now();
            this.currentStep = 0;
            this.totalSteps = this.properties.steps;
            this.previewImage = null;
            
            this.generateButton.hidden = true;
            this.stopButton.hidden = false;
            this.progressWidget.value = "Starting generation...";
            
            this.setStatus('processing', 'Generating image...');
            this.setDirtyCanvas(true, false);
            
            // Execute workflow via API
            const result = await apiService.executeWorkflow(executionRequest);
            
            if (result.success) {
                this.onGenerationComplete(result);
            } else {
                this.onGenerationError(result.message);
            }
            
        } catch (error) {
            console.error('Generation failed:', error);
            this.onGenerationError(error.message);
        }
    }
    
    async stopGeneration() {
        if (!this.isGenerating) return;
        
        try {
            // Call API to cancel execution
            await apiService.cancelExecution();
            this.onGenerationStopped();
        } catch (error) {
            console.error('Failed to stop generation:', error);
            this.onGenerationError('Failed to stop generation');
        }
    }
    
    createWorkflowFromGraph() {
        // Create workflow from the current graph state
        const components = [];
        const connections = [];
        
        // Traverse the graph to build workflow
        if (this.graph && this.graph.nodes) {
            for (const node of this.graph.nodes) {
                if (node.getComponentInstance) {
                    const componentInstance = node.getComponentInstance();
                    if (componentInstance) {
                        components.push(componentInstance);
                    }
                }
            }
            
            // Build connections from graph links
            if (this.graph.links) {
                for (const [linkId, link] of Object.entries(this.graph.links)) {
                    if (link && link.origin_node && link.target_node) {
                        const originNode = this.graph.getNodeById(link.origin_node);
                        const targetNode = this.graph.getNodeById(link.target_node);
                        
                        if (originNode && targetNode) {
                            const connection = new Connection(
                                `conn_${linkId}`,
                                originNode.componentId || originNode.id,
                                originNode.outputs[link.origin_slot]?.name || `output_${link.origin_slot}`,
                                targetNode.componentId || targetNode.id,
                                targetNode.inputs[link.target_slot]?.name || `input_${link.target_slot}`
                            );
                            connections.push(connection);
                        }
                    }
                }
            }
        }
        
        return new Workflow(components, connections);
    }
    
    handleWebSocketMessage(data) {
        if (!this.isGenerating) return;
        
        const { type, ...payload } = data;
        
        switch (type) {
            case 'progress':
                this.onProgressUpdate(payload);
                break;
                
            case 'preview':
                this.onPreviewUpdate(payload);
                break;
                
            case 'result':
                this.onGenerationComplete(payload);
                break;
                
            case 'error':
                this.onGenerationError(payload.error);
                break;
                
            case 'component_status':
                // Check if it's our component
                if (payload.component_id === this.componentId) {
                    this.onComponentStatusUpdate(payload);
                }
                break;
        }
    }
    
    onProgressUpdate(payload) {
        this.currentStep = payload.step || 0;
        this.totalSteps = payload.total || this.properties.steps;
        
        const percentage = payload.percentage || Math.round((this.currentStep / this.totalSteps) * 100);
        const message = payload.message || `Step ${this.currentStep}/${this.totalSteps}`;
        
        this.progressWidget.value = `${message} (${percentage}%)`;
        this.setStatus('processing', message);
        this.setDirtyCanvas(true, false);
    }
    
    onPreviewUpdate(payload) {
        if (payload.image) {
            // Store base64 preview image
            this.previewImage = payload.image;
            this.setDirtyCanvas(true, false);
        }
    }
    
    onGenerationComplete(result) {
        this.isGenerating = false;
        this.generateButton.hidden = false;
        this.stopButton.hidden = true;
        
        const duration = this.generationStartTime ? 
            ((Date.now() - this.generationStartTime) / 1000).toFixed(1) : '?';
        
        this.progressWidget.value = `Complete (${duration}s)`;
        this.setStatus('complete', `Generated in ${duration}s`);
        
        // Store result images in outputs
        if (result.images && result.images.length > 0) {
            this.setOutputData(0, result.images); // images output
            
            // Show first image as preview
            this.previewImage = result.images[0];
        }
        
        this.setDirtyCanvas(true, false);
    }
    
    onGenerationError(error) {
        this.isGenerating = false;
        this.generateButton.hidden = false;
        this.stopButton.hidden = true;
        
        this.progressWidget.value = "Error occurred";
        this.setStatus('error', error);
        this.setDirtyCanvas(true, false);
    }
    
    onGenerationStopped() {
        this.isGenerating = false;
        this.generateButton.hidden = false;
        this.stopButton.hidden = true;
        
        this.progressWidget.value = "Stopped";
        this.setStatus('idle', 'Generation stopped');
        this.setDirtyCanvas(true, false);
    }
    
    onComponentStatusUpdate(payload) {
        // Update component-specific status
        if (payload.status === 'processing') {
            this.setStatus('processing', payload.message);
        } else if (payload.status === 'complete') {
            this.setStatus('complete', payload.message);
        } else if (payload.status === 'error') {
            this.setStatus('error', payload.message);
        }
    }
    
    // Custom drawing for preview and progress
    onDrawForeground(ctx) {
        super.onDrawForeground(ctx);
        
        if (this.flags.collapsed) return;
        
        // Draw preview area
        const previewY = this.size[1] - this.previewSize - 20;
        const previewX = 10;
        
        // Preview background
        ctx.save();
        ctx.fillStyle = "#222";
        ctx.fillRect(previewX, previewY, this.previewSize, this.previewSize);
        ctx.strokeStyle = "#666";
        ctx.strokeRect(previewX, previewY, this.previewSize, this.previewSize);
        
        // Draw preview image if available
        if (this.previewImage) {
            this.drawPreviewImage(ctx, previewX, previewY, this.previewSize, this.previewSize);
        } else {
            // Draw placeholder
            ctx.fillStyle = "#555";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Preview", previewX + this.previewSize/2, previewY + this.previewSize/2);
        }
        
        // Draw progress bar if generating
        if (this.isGenerating && this.totalSteps > 0) {
            const progress = this.currentStep / this.totalSteps;
            const barWidth = this.previewSize;
            const barHeight = 4;
            const barY = previewY - 10;
            
            // Background
            ctx.fillStyle = "#333";
            ctx.fillRect(previewX, barY, barWidth, barHeight);
            
            // Progress
            ctx.fillStyle = "#4CAF50";
            ctx.fillRect(previewX, barY, barWidth * progress, barHeight);
        }
        
        ctx.restore();
    }
    
    drawPreviewImage(ctx, x, y, width, height) {
        try {
            // Create image element from base64
            const img = new Image();
            img.onload = () => {
                ctx.save();
                ctx.drawImage(img, x, y, width, height);
                ctx.restore();
                this.setDirtyCanvas(true, false);
            };
            
            // Handle base64 data
            if (this.previewImage.startsWith('data:')) {
                img.src = this.previewImage;
            } else {
                img.src = `data:image/png;base64,${this.previewImage}`;
            }
        } catch (error) {
            console.error('Failed to draw preview image:', error);
        }
    }
    
    // Handle input connections for auto-configuration
    onConnectionsChange(type, slotIndex, isConnected, link_info, slot_info) {
        super.onConnectionsChange(type, slotIndex, isConnected, link_info, slot_info);
        
        if (type === LiteGraph.INPUT && isConnected) {
            // Check if connected to a model node for auto-configuration
            setTimeout(() => {
                this.checkForAutoConfiguration();
            }, 100);
        }
    }
    
    checkForAutoConfiguration() {
        // Look for connected model nodes with auto-configuration
        if (this.inputs && this.inputs.length > 0) {
            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.link) {
                    const link = this.graph.links[input.link];
                    if (link) {
                        const sourceNode = this.graph.getNodeById(link.origin_node);
                        if (sourceNode && sourceNode.componentType === 'model' && 
                            sourceNode.modelConfig && sourceNode.properties?.auto_configure) {
                            // Apply model configuration
                            this.onPropertyChanged('__auto_config', sourceNode.modelConfig);
                        }
                    }
                }
            }
        }
    }
    
    getTitle() {
        if (this.isGenerating) {
            return `🎲 Sampler (${this.currentStep}/${this.totalSteps})`;
        }
        return "🎲 Sampler";
    }
    
    // Cleanup on node removal
    onRemoved() {
        if (typeof apiService !== 'undefined') {
            apiService.removeWebSocketCallback(this.wsCallback);
        }
        super.onRemoved && super.onRemoved();
    }
    
    // Override to handle sampler-specific serialization
    serialize() {
        const data = super.serialize();
        data.previewImage = this.previewImage;
        data.isGenerating = this.isGenerating;
        return data;
    }
    
    configure(data) {
        super.configure(data);
        if (data.previewImage) {
            this.previewImage = data.previewImage;
        }
        this.isGenerating = data.isGenerating || false;
        
        // Update button visibility
        this.generateButton.hidden = this.isGenerating;
        this.stopButton.hidden = !this.isGenerating;
    }
}

// Register the specialized Sampler node
if (typeof LiteGraph !== 'undefined') {
    LiteGraph.registerNodeType("SD Forms/Core/Sampler", SamplerNode);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SamplerNode;
} else {
    window.SamplerNode = SamplerNode;
}