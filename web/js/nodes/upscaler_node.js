/**
 * Upscaler Node - AI-powered image upscaling with multiple model presets
 * Supports Real-ESRGAN, ESRGAN, SwinIR, and LDSR models
 */

class UpscalerNode {
    constructor() {
        this.title = "Upscaler";
        this.desc = "AI-powered image upscaling";
        this.type = "sd_forms/upscaler";
        
        // Input ports
        this.addInput("images", "image");
        this.addInput("scale_override", "number");
        
        // Output ports
        this.addOutput("images", "image");
        this.addOutput("metadata", "any");
        
        // Model presets configuration
        this.modelPresets = {
            "realesrgan_x4plus": {
                name: "Real-ESRGAN x4+",
                description: "Best quality general purpose",
                scale: 4,
                type: "realesrgan",
                optimized_for: "general"
            },
            "realesrgan_x4plus_anime": {
                name: "Real-ESRGAN x4+ Anime",
                description: "Optimized for anime/artwork",
                scale: 4,
                type: "realesrgan",
                optimized_for: "anime"
            },
            "esrgan_x4": {
                name: "ESRGAN x4",
                description: "Original ESRGAN model",
                scale: 4,
                type: "esrgan",
                optimized_for: "general"
            },
            "swinir_x4": {
                name: "SwinIR x4",
                description: "Transformer-based upscaler",
                scale: 4,
                type: "swinir",
                optimized_for: "natural"
            },
            "ldsr_x4": {
                name: "LDSR x4",
                description: "Latent diffusion super resolution",
                scale: 4,
                type: "ldsr",
                optimized_for: "details"
            }
        };

        // Default properties
        this.properties = {
            // Model Settings
            model_type: "preset",
            preset_model: "realesrgan_x4plus",
            model_path: "",
            
            // Upscaling Settings
            scale_factor: 4,
            tile_size: 512,
            tile_overlap: 32,
            face_enhance: false,
            
            // Processing Settings
            batch_size: 1,
            precision: "fp16",
            denoise_strength: 0.0,
            
            // Output Settings
            output_format: "same",
            preserve_metadata: true
        };
        
        this.size = [240, 320];
        this.color = "#e67e22";
        this.bgcolor = "#8b4513";
        
        this.setupWidgets();
    }

    setupWidgets() {
        // Model Type Selection
        this.addWidget("combo", "Model Type", this.properties.model_type, (value) => {
            this.properties.model_type = value;
            this.onPropertyChanged("model_type", value);
            this.updateModelTypeOptions(value);
        }, {
            values: ["preset", "local", "auto"],
            property: "model_type"
        });

        // Preset Model Selection
        this.presetWidget = this.addWidget("combo", "Preset Model", this.properties.preset_model, (value) => {
            this.properties.preset_model = value;
            this.onPropertyChanged("preset_model", value);
            this.updatePresetSettings(value);
        }, {
            values: Object.keys(this.modelPresets),
            property: "preset_model"
        });

        // Custom Model Path
        this.modelPathWidget = this.addWidget("text", "Model Path", this.properties.model_path, (value) => {
            this.properties.model_path = value;
            this.onPropertyChanged("model_path", value);
        }, { 
            property: "model_path"
        });

        // Scale Factor
        this.scaleWidget = this.addWidget("slider", "Scale Factor", this.properties.scale_factor, (value) => {
            this.properties.scale_factor = value;
            this.onPropertyChanged("scale_factor", value);
        }, { 
            min: 2, 
            max: 8, 
            step: 1,
            property: "scale_factor"
        });

        // Tile Size for memory management
        this.addWidget("slider", "Tile Size", this.properties.tile_size, (value) => {
            this.properties.tile_size = Math.round(value);
            this.onPropertyChanged("tile_size", this.properties.tile_size);
        }, { 
            min: 128, 
            max: 1024, 
            step: 64,
            property: "tile_size"
        });

        // Tile Overlap
        this.addWidget("slider", "Tile Overlap", this.properties.tile_overlap, (value) => {
            this.properties.tile_overlap = Math.round(value);
            this.onPropertyChanged("tile_overlap", this.properties.tile_overlap);
        }, { 
            min: 0, 
            max: 128, 
            step: 8,
            property: "tile_overlap"
        });

        // Face Enhancement
        this.addWidget("toggle", "Face Enhance", this.properties.face_enhance, (value) => {
            this.properties.face_enhance = value;
            this.onPropertyChanged("face_enhance", value);
        }, { property: "face_enhance" });

        // Batch Size
        this.addWidget("slider", "Batch Size", this.properties.batch_size, (value) => {
            this.properties.batch_size = Math.round(value);
            this.onPropertyChanged("batch_size", this.properties.batch_size);
        }, { 
            min: 1, 
            max: 8, 
            step: 1,
            property: "batch_size"
        });

        // Precision
        this.addWidget("combo", "Precision", this.properties.precision, (value) => {
            this.properties.precision = value;
            this.onPropertyChanged("precision", value);
        }, {
            values: ["fp32", "fp16", "bf16"],
            property: "precision"
        });

        // Denoise Strength
        this.addWidget("slider", "Denoise", this.properties.denoise_strength, (value) => {
            this.properties.denoise_strength = value;
            this.onPropertyChanged("denoise_strength", value);
        }, { 
            min: 0.0, 
            max: 1.0, 
            step: 0.1,
            property: "denoise_strength"
        });

        // Output Format
        this.addWidget("combo", "Output Format", this.properties.output_format, (value) => {
            this.properties.output_format = value;
            this.onPropertyChanged("output_format", value);
        }, {
            values: ["same", "png", "jpeg", "webp"],
            property: "output_format"
        });

        // Preserve Metadata
        this.addWidget("toggle", "Preserve Metadata", this.properties.preserve_metadata, (value) => {
            this.properties.preserve_metadata = value;
            this.onPropertyChanged("preserve_metadata", value);
        }, { property: "preserve_metadata" });

        // Utility Buttons
        this.addWidget("button", "Browse Model", "", () => {
            if (window.modelBrowser) {
                window.modelBrowser.show(this);
            }
        });

        this.addWidget("button", "Memory Settings", "", () => {
            this.showMemoryOptimizationDialog();
        });

        // Update initial visibility
        this.updateModelTypeOptions(this.properties.model_type);
    }

    updateModelTypeOptions(modelType) {
        // Show/hide widgets based on model type
        const showPreset = modelType === "preset";
        const showCustom = modelType === "local";
        
        this.presetWidget.hidden = !showPreset;
        this.modelPathWidget.hidden = !showCustom;
        
        // Update scale factor constraints based on model type
        if (showPreset) {
            const preset = this.modelPresets[this.properties.preset_model];
            if (preset) {
                this.properties.scale_factor = preset.scale;
                this.scaleWidget.value = preset.scale;
                this.scaleWidget.options.min = preset.scale;
                this.scaleWidget.options.max = preset.scale;
            }
        } else {
            this.scaleWidget.options.min = 2;
            this.scaleWidget.options.max = 8;
        }

        this.setDirtyCanvas(true);
    }

    updatePresetSettings(presetKey) {
        const preset = this.modelPresets[presetKey];
        if (!preset) return;

        // Update scale factor to match preset
        this.properties.scale_factor = preset.scale;
        this.scaleWidget.value = preset.scale;

        // Update optimal settings for the preset
        switch (preset.type) {
            case "realesrgan":
                this.properties.tile_size = 512;
                this.properties.tile_overlap = 32;
                this.properties.precision = "fp16";
                break;
            case "esrgan":
                this.properties.tile_size = 512;
                this.properties.tile_overlap = 32;
                this.properties.precision = "fp32";
                break;
            case "swinir":
                this.properties.tile_size = 256;
                this.properties.tile_overlap = 16;
                this.properties.precision = "fp16";
                break;
            case "ldsr":
                this.properties.tile_size = 256;
                this.properties.tile_overlap = 16;
                this.properties.precision = "fp16";
                this.properties.batch_size = 1; // LDSR is memory intensive
                break;
        }

        // Update widget values
        this.updateWidgetValues();
    }

    updateWidgetValues() {
        // Update all widget values to match properties
        this.widgets.forEach(widget => {
            if (widget.property && this.properties.hasOwnProperty(widget.property)) {
                widget.value = this.properties[widget.property];
            }
        });
        this.setDirtyCanvas(true);
    }

    onPropertyChanged(name, value) {
        // Handle property-specific logic
        switch (name) {
            case "model_type":
                this.updateModelTypeOptions(value);
                break;
            case "preset_model":
                this.updatePresetSettings(value);
                break;
            case "scale_factor":
                this.validateScaleFactor(value);
                break;
            case "tile_size":
                this.validateTileSettings();
                break;
        }

        // Trigger graph update
        if (this.graph) {
            this.graph.setDirtyCanvas(true);
        }
    }

    validateScaleFactor(scale) {
        // Ensure scale factor is reasonable
        if (scale > 8) {
            console.warn("Scale factors above 8x may cause memory issues");
        }
        if (scale < 2) {
            this.properties.scale_factor = 2;
            this.scaleWidget.value = 2;
        }
    }

    validateTileSettings() {
        // Ensure tile overlap doesn't exceed tile size
        if (this.properties.tile_overlap >= this.properties.tile_size / 2) {
            this.properties.tile_overlap = Math.floor(this.properties.tile_size / 4);
            this.widgets.find(w => w.name === "Tile Overlap").value = this.properties.tile_overlap;
        }
    }

    showMemoryOptimizationDialog() {
        // Create a simple dialog for memory optimization presets
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 1000;
            color: white;
            min-width: 300px;
        `;

        dialog.innerHTML = `
            <h3>Memory Optimization</h3>
            <button onclick="this.parentElement.applyPreset('low')">Low Memory (512MB VRAM)</button><br><br>
            <button onclick="this.parentElement.applyPreset('medium')">Medium Memory (2GB VRAM)</button><br><br>
            <button onclick="this.parentElement.applyPreset('high')">High Memory (8GB+ VRAM)</button><br><br>
            <button onclick="this.parentElement.remove()">Cancel</button>
        `;

        dialog.applyPreset = (preset) => {
            switch (preset) {
                case 'low':
                    this.properties.tile_size = 256;
                    this.properties.tile_overlap = 16;
                    this.properties.batch_size = 1;
                    this.properties.precision = "fp16";
                    break;
                case 'medium':
                    this.properties.tile_size = 512;
                    this.properties.tile_overlap = 32;
                    this.properties.batch_size = 2;
                    this.properties.precision = "fp16";
                    break;
                case 'high':
                    this.properties.tile_size = 1024;
                    this.properties.tile_overlap = 64;
                    this.properties.batch_size = 4;
                    this.properties.precision = "fp16";
                    break;
            }
            this.updateWidgetValues();
            dialog.remove();
        };

        document.body.appendChild(dialog);
    }

    onExecute() {
        // Get input data
        const inputImages = this.getInputData(0);
        const scaleOverride = this.getInputData(1);

        if (!inputImages) {
            console.warn("Upscaler: No input images provided");
            return;
        }

        // Use scale override if provided
        const effectiveScale = scaleOverride || this.properties.scale_factor;

        // Prepare upscaling parameters
        const params = {
            ...this.properties,
            scale_factor: effectiveScale,
            input_images: inputImages,
            preset_config: this.modelPresets[this.properties.preset_model]
        };

        // Calculate estimated output resolution
        if (inputImages && inputImages.width && inputImages.height) {
            params.estimated_output = {
                width: inputImages.width * effectiveScale,
                height: inputImages.height * effectiveScale,
                memory_estimate: this.estimateMemoryUsage(inputImages, effectiveScale)
            };
        }

        // Set outputs (placeholder - actual processing happens on backend)
        this.setOutputData(0, null); // Upscaled images
        this.setOutputData(1, params); // Metadata
    }

    estimateMemoryUsage(inputImage, scale) {
        // Rough memory estimation for VRAM usage
        const inputPixels = (inputImage.width || 512) * (inputImage.height || 512);
        const outputPixels = inputPixels * scale * scale;
        const bytesPerPixel = this.properties.precision === "fp32" ? 12 : 6; // RGB * precision
        const estimatedMB = Math.round((outputPixels * bytesPerPixel) / (1024 * 1024));
        
        return {
            estimated_vram_mb: estimatedMB,
            recommended_tile_size: estimatedMB > 1000 ? 256 : 512
        };
    }

    onDrawBackground(ctx) {
        // Draw model type indicator
        ctx.fillStyle = "#e67e22";
        ctx.font = "10px Arial";
        ctx.textAlign = "right";
        
        let indicator = "";
        if (this.properties.model_type === "preset") {
            const preset = this.modelPresets[this.properties.preset_model];
            indicator = preset ? preset.name.split(" ")[0] : "PRESET";
        } else {
            indicator = this.properties.model_type.toUpperCase();
        }
        
        ctx.fillText(indicator, this.size[0] - 10, 20);

        // Draw scale factor
        ctx.fillStyle = "#fff";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`${this.properties.scale_factor}x`, 10, this.size[1] - 10);

        // Draw memory warning if needed
        if (this.properties.tile_size > 512 && this.properties.batch_size > 2) {
            ctx.fillStyle = "#ff6b6b";
            ctx.font = "8px Arial";
            ctx.textAlign = "center";
            ctx.fillText("HIGH MEMORY", this.size[0] / 2, this.size[1] - 25);
        }
    }

    getMenuOptions() {
        return [
            {
                content: "Reset to Defaults",
                callback: () => {
                    this.properties = {
                        model_type: "preset",
                        preset_model: "realesrgan_x4plus",
                        model_path: "",
                        scale_factor: 4,
                        tile_size: 512,
                        tile_overlap: 32,
                        face_enhance: false,
                        batch_size: 1,
                        precision: "fp16",
                        denoise_strength: 0.0,
                        output_format: "same",
                        preserve_metadata: true
                    };
                    this.setupWidgets();
                }
            },
            {
                content: "Memory Optimization",
                callback: () => {
                    this.showMemoryOptimizationDialog();
                }
            },
            {
                content: "Model Info",
                callback: () => {
                    if (this.properties.model_type === "preset") {
                        const preset = this.modelPresets[this.properties.preset_model];
                        alert(`Model: ${preset.name}\nDescription: ${preset.description}\nOptimized for: ${preset.optimized_for}\nScale: ${preset.scale}x`);
                    }
                }
            },
            {
                content: "Copy Settings",
                callback: () => {
                    navigator.clipboard.writeText(JSON.stringify(this.properties, null, 2));
                }
            }
        ];
    }

    serialize() {
        const data = LGraphNode.prototype.serialize.call(this);
        data.properties = this.properties;
        return data;
    }

    configure(data) {
        LGraphNode.prototype.configure.call(this, data);
        if (data.properties) {
            Object.assign(this.properties, data.properties);
        }
        this.setupWidgets();
    }

    onConnectionsChange() {
        // Update scale factor if override is connected
        const scaleOverride = this.getInputData(1);
        if (scaleOverride && scaleOverride !== this.properties.scale_factor) {
            // Visual indication that scale is overridden
            this.setDirtyCanvas(true);
        }
    }
}

// Register the node
LiteGraph.registerNodeType("sd_forms/upscaler", UpscalerNode);

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UpscalerNode;
}