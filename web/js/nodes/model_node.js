/**
 * Model Node - Specialized LiteGraph node for SD Forms Model component
 * Implements custom UI for model selection, presets, and auto-configuration
 */

// Model configurations matching sd_forms/components/standard.py
const MODEL_CONFIGS = {
    'sd15': {
        'name': 'Stable Diffusion 1.5',
        'steps': 20,
        'cfg': 7.5,
        'optimal_size': [512, 512],
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sd-v1-5', 'v1-5', 'stable-diffusion-v1-5']
    },
    'sd21': {
        'name': 'Stable Diffusion 2.1',
        'steps': 25,
        'cfg': 7.0,
        'optimal_size': [768, 768],
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sd-v2-1', 'v2-1', 'stable-diffusion-2-1']
    },
    'sdxl': {
        'name': 'Stable Diffusion XL',
        'steps': 25,
        'cfg': 7.0,
        'optimal_size': [1024, 1024],
        'sampler': 'DPMSolverMultistep',
        'identifiers': ['sdxl', 'xl', 'stable-diffusion-xl']
    },
    'flux': {
        'name': 'Flux',
        'steps': 4,
        'cfg': 1.0,
        'optimal_size': [1024, 1024],
        'sampler': 'euler',
        'identifiers': ['flux', 'flux-dev', 'flux-schnell']
    }
};

// Preset model options
const FLUX_PRESETS = [
    "flux1-schnell",
    "flux1-dev", 
    "flux-lite-8b",
    "flux-dev-distilled",
    "flux-sigma-vision",
    "getphat-reality-v20",
    "pixelwave-flux"
];

const SD_PRESETS = [
    "stable-diffusion-v1-5",
    "stable-diffusion-v2-1",
    "stable-diffusion-xl-base-1.0"
];

/**
 * Specialized Model Node with custom UI and auto-configuration
 */
class ModelNode extends SDFormsNode {
    constructor() {
        // Create mock component info for Model
        const modelComponentInfo = {
            id: 'model_component',
            type: 'model',
            display_name: 'Model',
            category: 'Core',
            icon: '📦',
            input_ports: [],
            output_ports: [
                { name: 'pipeline', type: 'pipeline', direction: 'OUTPUT' },
                { name: 'conditioning', type: 'conditioning', direction: 'OUTPUT' }
            ],
            property_definitions: [] // We'll create custom widgets instead
        };
        
        super(modelComponentInfo);
        
        // Custom initialization
        this.setupModelNode();
    }
    
    setupModelNode() {
        // Clear default widgets and create custom ones
        this.widgets = [];
        this.properties = {};
        this.propertyValues = {};
        
        // Set node appearance
        this.title = "📦 Model";
        this.size = [280, 400]; // Larger to accommodate all widgets
        this.color = "#3498db";
        this.bgcolor = "#2c3e50";
        
        // Initialize properties
        this.initializeProperties();
        
        // Create custom widgets
        this.createModelSourceWidget();
        this.createModelSelectionWidgets();
        this.createVAEWidgets();
        this.createPromptWidgets();
        this.createLoRAWidget();
        this.createAdvancedWidgets();
        this.createInfoWidgets();
        
        // Set up property dependencies
        this.setupPropertyDependencies();
        
        // Initial UI update
        this.updateWidgetVisibility();
    }
    
    initializeProperties() {
        const defaults = {
            checkpoint_type: "preset",
            flux_preset: "flux1-schnell",
            sd_preset: "stable-diffusion-v1-5",
            checkpoint: "runwayml/stable-diffusion-v1-5",
            checkpoint_path: "",
            vae_type: "auto",
            vae_path: "",
            clip_skip: 1,
            prompt: "a professional photo of a beautiful landscape at golden hour, highly detailed, photorealistic",
            negative_prompt: "",
            loras: "",
            enable_freeu: false,
            enable_clip_fix: false,
            detected_model_type: "Unknown",
            auto_configure: true
        };
        
        for (const [key, value] of Object.entries(defaults)) {
            this.properties[key] = value;
            this.propertyValues[key] = value;
        }
    }
    
    createModelSourceWidget() {
        // Model Source dropdown
        this.checkpointTypeWidget = this.addWidget(
            "combo",
            "Model Source",
            this.properties.checkpoint_type,
            (value) => this.onPropertyChanged("checkpoint_type", value),
            { values: ["preset", "local", "huggingface"] }
        );
        this.checkpointTypeWidget.property_name = "checkpoint_type";
    }
    
    createModelSelectionWidgets() {
        // Flux Preset dropdown (shown when checkpoint_type is "preset")
        this.fluxPresetWidget = this.addWidget(
            "combo",
            "Flux Model",
            this.properties.flux_preset,
            (value) => this.onPropertyChanged("flux_preset", value),
            { values: FLUX_PRESETS }
        );
        this.fluxPresetWidget.property_name = "flux_preset";
        
        // SD Preset dropdown (alternative to flux)
        this.sdPresetWidget = this.addWidget(
            "combo", 
            "SD Model",
            this.properties.sd_preset,
            (value) => this.onPropertyChanged("sd_preset", value),
            { values: SD_PRESETS }
        );
        this.sdPresetWidget.property_name = "sd_preset";
        
        // HuggingFace model input (shown when checkpoint_type is "huggingface")
        this.checkpointWidget = this.addWidget(
            "text",
            "HuggingFace Model",
            this.properties.checkpoint,
            (value) => this.onPropertyChanged("checkpoint", value)
        );
        this.checkpointWidget.property_name = "checkpoint";
        
        // Local file path (shown when checkpoint_type is "local")
        this.checkpointPathWidget = this.addWidget(
            "text",
            "Local Model Path",
            this.properties.checkpoint_path,
            (value) => this.onPropertyChanged("checkpoint_path", value)
        );
        this.checkpointPathWidget.property_name = "checkpoint_path";
        
        // File browser button for local models
        this.browseButton = this.addWidget(
            "button",
            "Browse...",
            null,
            () => this.openFileBrowser()
        );
    }
    
    createVAEWidgets() {
        // VAE Type selection
        this.vaeTypeWidget = this.addWidget(
            "combo",
            "VAE Type",
            this.properties.vae_type,
            (value) => this.onPropertyChanged("vae_type", value),
            { values: ["auto", "included", "external"] }
        );
        this.vaeTypeWidget.property_name = "vae_type";
        
        // External VAE path (shown when vae_type is "external")
        this.vaePathWidget = this.addWidget(
            "text",
            "External VAE Path",
            this.properties.vae_path,
            (value) => this.onPropertyChanged("vae_path", value)
        );
        this.vaePathWidget.property_name = "vae_path";
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
    
    createLoRAWidget() {
        // LoRA collection - custom text input with helper button
        this.lorasWidget = this.addWidget(
            "text",
            "LoRA Collection",
            this.properties.loras,
            (value) => this.onPropertyChanged("loras", value)
        );
        this.lorasWidget.property_name = "loras";
        
        // LoRA helper button
        this.loraButton = this.addWidget(
            "button",
            "Manage LoRAs...",
            null,
            () => this.openLoRAManager()
        );
    }
    
    createAdvancedWidgets() {
        // CLIP Skip slider
        this.clipSkipWidget = this.addWidget(
            "slider",
            "CLIP Skip",
            this.properties.clip_skip,
            (value) => this.onPropertyChanged("clip_skip", Math.round(value)),
            { min: 1, max: 12, step: 1 }
        );
        this.clipSkipWidget.property_name = "clip_skip";
        
        // Enable FreeU checkbox
        this.enableFreeUWidget = this.addWidget(
            "toggle",
            "Enable FreeU",
            this.properties.enable_freeu,
            (value) => this.onPropertyChanged("enable_freeu", value)
        );
        this.enableFreeUWidget.property_name = "enable_freeu";
        
        // Enable CLIP Fix checkbox
        this.enableClipFixWidget = this.addWidget(
            "toggle",
            "Enable CLIP Fix",
            this.properties.enable_clip_fix,
            (value) => this.onPropertyChanged("enable_clip_fix", value)
        );
        this.enableClipFixWidget.property_name = "enable_clip_fix";
        
        // Auto-configure checkbox
        this.autoConfigureWidget = this.addWidget(
            "toggle",
            "Auto-Configure Sampler",
            this.properties.auto_configure,
            (value) => this.onPropertyChanged("auto_configure", value)
        );
        this.autoConfigureWidget.property_name = "auto_configure";
    }
    
    createInfoWidgets() {
        // Detected model type (read-only)
        this.detectedModelWidget = this.addWidget(
            "text",
            "Detected Model",
            this.properties.detected_model_type,
            null,
            { disabled: true }
        );
        this.detectedModelWidget.property_name = "detected_model_type";
        
        // Model detection button
        this.detectButton = this.addWidget(
            "button",
            "Detect Model Type",
            null,
            () => this.detectModelType()
        );
    }
    
    setupPropertyDependencies() {
        // Update UI when checkpoint_type changes
        this.addPropertyChangeHandler("checkpoint_type", (value) => {
            this.updateWidgetVisibility();
            if (value === "preset") {
                this.autoDetectPresetModel();
            } else if (value === "local" && this.properties.checkpoint_path) {
                this.detectModelType();
            }
        });
        
        // Update UI when vae_type changes
        this.addPropertyChangeHandler("vae_type", () => {
            this.updateWidgetVisibility();
        });
        
        // Auto-detect when local path changes
        this.addPropertyChangeHandler("checkpoint_path", (value) => {
            if (value && this.properties.checkpoint_type === "local") {
                this.detectModelType();
            }
        });
        
        // Auto-configure when model selection changes
        this.addPropertyChangeHandler("flux_preset", () => {
            if (this.properties.auto_configure) {
                this.autoConfigureSampler();
            }
        });
    }
    
    updateWidgetVisibility() {
        const checkpointType = this.properties.checkpoint_type;
        const vaeType = this.properties.vae_type;
        
        // Show/hide model selection widgets based on checkpoint_type
        if (this.fluxPresetWidget) {
            this.fluxPresetWidget.hidden = checkpointType !== "preset";
        }
        if (this.sdPresetWidget) {
            this.sdPresetWidget.hidden = checkpointType !== "preset";
        }
        if (this.checkpointWidget) {
            this.checkpointWidget.hidden = checkpointType !== "huggingface";
        }
        if (this.checkpointPathWidget) {
            this.checkpointPathWidget.hidden = checkpointType !== "local";
        }
        if (this.browseButton) {
            this.browseButton.hidden = checkpointType !== "local";
        }
        
        // Show/hide VAE path based on vae_type
        if (this.vaePathWidget) {
            this.vaePathWidget.hidden = vaeType !== "external";
        }
        
        this.setDirtyCanvas(true, false);
    }
    
    onPropertyChanged(propertyName, value, widget) {
        // Call parent implementation
        super.onPropertyChanged(propertyName, value, widget);
        
        // Handle special model node logic
        if (propertyName === "checkpoint_type") {
            this.updateWidgetVisibility();
            if (value === "preset") {
                this.autoDetectPresetModel();
            }
        } else if (propertyName === "vae_type") {
            this.updateWidgetVisibility();
        } else if (propertyName === "checkpoint_path" && value) {
            this.detectModelType();
        } else if (propertyName === "flux_preset" || propertyName === "sd_preset") {
            if (this.properties.auto_configure) {
                this.autoConfigureSampler();
            }
        }
    }
    
    autoDetectPresetModel() {
        // Set detected model type based on preset selection
        const preset = this.properties.flux_preset || this.properties.sd_preset;
        
        if (preset.includes("flux")) {
            this.setProperty("detected_model_type", "Flux");
        } else if (preset.includes("xl")) {
            this.setProperty("detected_model_type", "Stable Diffusion XL");
        } else if (preset.includes("v2")) {
            this.setProperty("detected_model_type", "Stable Diffusion 2.1");
        } else {
            this.setProperty("detected_model_type", "Stable Diffusion 1.5");
        }
    }
    
    async detectModelType() {
        const checkpointPath = this.properties.checkpoint_path;
        if (!checkpointPath) return;
        
        try {
            this.setStatus('processing', 'Detecting model type...');
            
            // Simple filename-based detection (client-side)
            const filename = checkpointPath.toLowerCase();
            let detectedType = "Unknown";
            
            for (const [modelKey, config] of Object.entries(MODEL_CONFIGS)) {
                for (const identifier of config.identifiers) {
                    if (filename.includes(identifier)) {
                        detectedType = config.name;
                        break;
                    }
                }
                if (detectedType !== "Unknown") break;
            }
            
            // Fallback size-based detection could be done via API call
            if (detectedType === "Unknown") {
                // Could call backend API for more sophisticated detection
                detectedType = await this.detectModelTypeViaAPI(checkpointPath);
            }
            
            this.setProperty("detected_model_type", detectedType);
            this.setStatus('complete', 'Model type detected');
            
            // Auto-configure if enabled
            if (this.properties.auto_configure) {
                this.autoConfigureSampler();
            }
            
        } catch (error) {
            console.error('Model detection failed:', error);
            this.setProperty("detected_model_type", "Detection Failed");
            this.setStatus('error', 'Failed to detect model type');
        }
    }
    
    async detectModelTypeViaAPI(checkpointPath) {
        // This would call the backend API for model detection
        // For now, return a placeholder
        return "Unknown (API detection not implemented)";
    }
    
    autoConfigureSampler() {
        if (!this.properties.auto_configure) return;
        
        try {
            // Get model configuration
            let modelConfig = null;
            const detectedType = this.properties.detected_model_type.toLowerCase();
            
            for (const [key, config] of Object.entries(MODEL_CONFIGS)) {
                if (detectedType.includes(key) || detectedType.includes(config.name.toLowerCase())) {
                    modelConfig = config;
                    break;
                }
            }
            
            if (!modelConfig) {
                // Default to flux config for preset models
                if (this.properties.checkpoint_type === "preset") {
                    modelConfig = MODEL_CONFIGS.flux;
                } else {
                    modelConfig = MODEL_CONFIGS.sd15; // Safe fallback
                }
            }
            
            console.log(`🔧 Auto-configuring for ${modelConfig.name}:`);
            console.log(`  Recommended steps: ${modelConfig.steps}`);
            console.log(`  Recommended CFG: ${modelConfig.cfg}`);
            console.log(`  Optimal size: ${modelConfig.optimal_size}`);
            console.log(`  Recommended sampler: ${modelConfig.sampler}`);
            
            // Store configuration for connected sampler nodes to use
            this.modelConfig = modelConfig;
            
            // Could emit event or call API to update connected samplers
            this.notifyConnectedSamplers(modelConfig);
            
        } catch (error) {
            console.error('Auto-configuration failed:', error);
        }
    }
    
    notifyConnectedSamplers(config) {
        // Find connected sampler nodes and update their properties
        if (this.outputs && this.outputs.length > 0) {
            for (let i = 0; i < this.outputs.length; i++) {
                const connections = this.outputs[i].links;
                if (connections) {
                    for (const linkId of connections) {
                        const link = this.graph.links[linkId];
                        if (link && link.target_node) {
                            const targetNode = this.graph.getNodeById(link.target_node);
                            if (targetNode && targetNode.componentType === 'sampler') {
                                // Update sampler properties
                                if (targetNode.setProperty) {
                                    targetNode.setProperty('steps', config.steps);
                                    targetNode.setProperty('cfg_scale', config.cfg);
                                    targetNode.setProperty('width', config.optimal_size[0]);
                                    targetNode.setProperty('height', config.optimal_size[1]);
                                    targetNode.setProperty('scheduler', config.sampler);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    openFileBrowser() {
        // Create file input for local model selection
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.safetensors,.ckpt,.pt,.pth';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                // For web, we can only get the file name, not full path
                this.setProperty("checkpoint_path", file.name);
                console.log('Selected file:', file.name);
                
                // In a real implementation, you might upload the file or 
                // provide instructions for placing it in the correct directory
            }
        };
        input.click();
    }
    
    openLoRAManager() {
        // TODO: Implement LoRA management dialog
        alert("LoRA Manager not yet implemented.\nFor now, enter LoRA names separated by commas in the text field.");
    }
    
    // Custom drawing for model-specific visual feedback
    onDrawForeground(ctx) {
        super.onDrawForeground(ctx);
        
        // Draw model type indicator
        if (this.properties.detected_model_type && this.properties.detected_model_type !== "Unknown") {
            ctx.save();
            ctx.font = "10px Arial";
            ctx.fillStyle = "#00FF00";
            ctx.textAlign = "right";
            ctx.fillText(this.properties.detected_model_type, this.size[0] - 8, this.size[1] - 8);
            ctx.restore();
        }
        
        // Draw auto-configure indicator
        if (this.properties.auto_configure) {
            ctx.save();
            ctx.fillStyle = "#FFD700";
            ctx.beginPath();
            ctx.arc(this.size[0] - 25, this.size[1] - 15, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }
    
    getTitle() {
        return "📦 Model";
    }
    
    // Override to handle model-specific serialization
    serialize() {
        const data = super.serialize();
        data.modelConfig = this.modelConfig;
        return data;
    }
    
    configure(data) {
        super.configure(data);
        if (data.modelConfig) {
            this.modelConfig = data.modelConfig;
        }
        this.updateWidgetVisibility();
    }
}

// Register the specialized Model node
if (typeof LiteGraph !== 'undefined') {
    LiteGraph.registerNodeType("SD Forms/Core/Model", ModelNode);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelNode;
} else {
    window.ModelNode = ModelNode;
}