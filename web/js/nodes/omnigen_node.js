/**
 * OmniGen Node - Multi-modal unified generation model
 * Supports text2img, editing, subject-driven, and multi-modal generation
 */

class OmniGenNode {
    constructor() {
        this.title = "OmniGen";
        this.desc = "Multi-modal unified generation model";
        this.type = "sd_forms/omnigen";
        
        // Input ports
        this.addInput("instruction", "string");
        this.addInput("image1", "image");
        this.addInput("image2", "image");
        this.addInput("image3", "image");
        this.addInput("mask", "image");
        this.addInput("subject_images", "image");
        
        // Output ports
        this.addOutput("images", "image");
        this.addOutput("edited_regions", "any");
        this.addOutput("metadata", "any");
        
        // Default properties
        this.properties = {
            // Model Settings
            model_variant: "omnigen-v1",
            model_path: "",
            task_mode: "auto",
            
            // Generation Settings
            instruction: "",
            num_inference_steps: 50,
            guidance_scale: 3.0,
            resolution: 1024,
            aspect_ratio: "1:1",
            seed: -1,
            batch_size: 1,
            
            // Subject Control
            subject_mode: "single",
            subject_strength: 0.8,
            subject_token_mode: "auto",
            
            // Editing Features
            edit_mode: "instruction",
            preserve_unmasked: true,
            edit_strength: 0.8,
            
            // Multi-Modal
            image_fusion_mode: "guided",
            cross_attention_scale: 1.0
        };
        
        this.size = [280, 400];
        this.color = "#4a90e2";
        this.bgcolor = "#1e3a5f";
        
        this.setupWidgets();
    }

    setupWidgets() {
        // Model Variant
        this.addWidget("combo", "Model", this.properties.model_variant, (value) => {
            this.properties.model_variant = value;
            this.onPropertyChanged("model_variant", value);
        }, {
            values: ["omnigen-v1", "omnigen-v1-turbo"],
            property: "model_variant"
        });

        // Model Path (with browse button)
        this.addWidget("text", "Model Path", this.properties.model_path, (value) => {
            this.properties.model_path = value;
            this.onPropertyChanged("model_path", value);
        }, { property: "model_path" });

        // Task Mode
        this.addWidget("combo", "Task Mode", this.properties.task_mode, (value) => {
            this.properties.task_mode = value;
            this.onPropertyChanged("task_mode", value);
            this.updateInputsForTaskMode(value);
        }, {
            values: ["auto", "text2img", "editing", "subject_driven", "multi_modal", "instruction"],
            property: "task_mode"
        });

        // Instruction (multiline text)
        this.addWidget("text", "Instruction", this.properties.instruction, (value) => {
            this.properties.instruction = value;
            this.onPropertyChanged("instruction", value);
        }, { 
            property: "instruction",
            multiline: true,
            height: 60
        });

        // Generation Settings
        this.addWidget("slider", "Steps", this.properties.num_inference_steps, (value) => {
            this.properties.num_inference_steps = Math.round(value);
            this.onPropertyChanged("num_inference_steps", this.properties.num_inference_steps);
        }, { 
            min: 1, 
            max: 200, 
            step: 1,
            property: "num_inference_steps"
        });

        this.addWidget("slider", "Guidance", this.properties.guidance_scale, (value) => {
            this.properties.guidance_scale = value;
            this.onPropertyChanged("guidance_scale", value);
        }, { 
            min: 1.0, 
            max: 20.0, 
            step: 0.1,
            property: "guidance_scale"
        });

        // Resolution
        this.addWidget("combo", "Resolution", this.properties.resolution.toString(), (value) => {
            this.properties.resolution = parseInt(value);
            this.onPropertyChanged("resolution", this.properties.resolution);
        }, {
            values: ["512", "768", "1024", "1280", "1536"],
            property: "resolution"
        });

        // Aspect Ratio
        this.addWidget("combo", "Aspect Ratio", this.properties.aspect_ratio, (value) => {
            this.properties.aspect_ratio = value;
            this.onPropertyChanged("aspect_ratio", value);
        }, {
            values: ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "custom"],
            property: "aspect_ratio"
        });

        // Seed
        this.addWidget("number", "Seed", this.properties.seed, (value) => {
            this.properties.seed = Math.round(value);
            this.onPropertyChanged("seed", this.properties.seed);
        }, { 
            property: "seed",
            precision: 0
        });

        // Batch Size
        this.addWidget("slider", "Batch Size", this.properties.batch_size, (value) => {
            this.properties.batch_size = Math.round(value);
            this.onPropertyChanged("batch_size", this.properties.batch_size);
        }, { 
            min: 1, 
            max: 4, 
            step: 1,
            property: "batch_size"
        });

        // Subject Settings (collapsible)
        this.addWidget("combo", "Subject Mode", this.properties.subject_mode, (value) => {
            this.properties.subject_mode = value;
            this.onPropertyChanged("subject_mode", value);
        }, {
            values: ["single", "multiple", "identity_preserve"],
            property: "subject_mode"
        });

        this.addWidget("slider", "Subject Strength", this.properties.subject_strength, (value) => {
            this.properties.subject_strength = value;
            this.onPropertyChanged("subject_strength", value);
        }, { 
            min: 0.0, 
            max: 1.0, 
            step: 0.1,
            property: "subject_strength"
        });

        // Editing Settings
        this.addWidget("combo", "Edit Mode", this.properties.edit_mode, (value) => {
            this.properties.edit_mode = value;
            this.onPropertyChanged("edit_mode", value);
        }, {
            values: ["instruction", "mask", "region", "global"],
            property: "edit_mode"
        });

        this.addWidget("toggle", "Preserve Unmasked", this.properties.preserve_unmasked, (value) => {
            this.properties.preserve_unmasked = value;
            this.onPropertyChanged("preserve_unmasked", value);
        }, { property: "preserve_unmasked" });

        this.addWidget("slider", "Edit Strength", this.properties.edit_strength, (value) => {
            this.properties.edit_strength = value;
            this.onPropertyChanged("edit_strength", value);
        }, { 
            min: 0.0, 
            max: 1.0, 
            step: 0.1,
            property: "edit_strength"
        });

        // Multi-Modal Settings
        this.addWidget("combo", "Fusion Mode", this.properties.image_fusion_mode, (value) => {
            this.properties.image_fusion_mode = value;
            this.onPropertyChanged("image_fusion_mode", value);
        }, {
            values: ["guided", "blend", "composite", "reference"],
            property: "image_fusion_mode"
        });

        this.addWidget("slider", "Cross Attention", this.properties.cross_attention_scale, (value) => {
            this.properties.cross_attention_scale = value;
            this.onPropertyChanged("cross_attention_scale", value);
        }, { 
            min: 0.0, 
            max: 2.0, 
            step: 0.1,
            property: "cross_attention_scale"
        });

        // Add utility buttons
        this.addWidget("button", "Random Seed", "", () => {
            this.properties.seed = Math.floor(Math.random() * 4294967295);
            this.widgets.find(w => w.name === "Seed").value = this.properties.seed;
            this.onPropertyChanged("seed", this.properties.seed);
        });

        this.addWidget("button", "Browse Model", "", () => {
            if (window.modelBrowser) {
                window.modelBrowser.show(this);
            }
        });
    }

    updateInputsForTaskMode(taskMode) {
        // Show/hide inputs based on task mode
        const inputsConfig = {
            "text2img": ["instruction"],
            "editing": ["instruction", "image1", "mask"],
            "subject_driven": ["instruction", "subject_images"],
            "multi_modal": ["instruction", "image1", "image2", "image3"],
            "instruction": ["instruction", "image1"],
            "auto": ["instruction", "image1", "image2", "image3", "mask", "subject_images"]
        };

        const activeInputs = inputsConfig[taskMode] || inputsConfig["auto"];
        
        // Update input visibility (if LiteGraph supports it)
        for (let i = 0; i < this.inputs.length; i++) {
            const input = this.inputs[i];
            if (input && input.name) {
                input.optional = !activeInputs.includes(input.name);
            }
        }

        this.setDirtyCanvas(true);
    }

    onPropertyChanged(name, value) {
        // Handle property-specific logic
        switch (name) {
            case "model_variant":
                this.updateModelConfiguration(value);
                break;
            case "task_mode":
                this.updateInputsForTaskMode(value);
                break;
            case "seed":
                if (value === -1) {
                    this.properties.seed = Math.floor(Math.random() * 4294967295);
                }
                break;
        }

        // Trigger graph update
        if (this.graph) {
            this.graph.setDirtyCanvas(true);
        }
    }

    updateModelConfiguration(variant) {
        // Update default settings based on model variant
        if (variant === "omnigen-v1-turbo") {
            this.properties.num_inference_steps = Math.min(this.properties.num_inference_steps, 10);
            this.properties.guidance_scale = Math.min(this.properties.guidance_scale, 5.0);
            
            // Update widget values
            const stepsWidget = this.widgets.find(w => w.name === "Steps");
            const guidanceWidget = this.widgets.find(w => w.name === "Guidance");
            
            if (stepsWidget) stepsWidget.value = this.properties.num_inference_steps;
            if (guidanceWidget) guidanceWidget.value = this.properties.guidance_scale;
        }
    }

    onExecute() {
        // Get input data
        const inputs = {
            instruction: this.getInputData(0) || this.properties.instruction,
            image1: this.getInputData(1),
            image2: this.getInputData(2),
            image3: this.getInputData(3),
            mask: this.getInputData(4),
            subject_images: this.getInputData(5)
        };

        // Process instruction for image tokens
        let processedInstruction = inputs.instruction;
        if (inputs.image1) processedInstruction = processedInstruction.replace(/<img1>/g, "[image1]");
        if (inputs.image2) processedInstruction = processedInstruction.replace(/<img2>/g, "[image2]");
        if (inputs.image3) processedInstruction = processedInstruction.replace(/<img3>/g, "[image3]");

        // Prepare generation parameters
        const params = {
            ...this.properties,
            instruction: processedInstruction,
            images: [inputs.image1, inputs.image2, inputs.image3].filter(Boolean),
            mask: inputs.mask,
            subject_images: inputs.subject_images
        };

        // Set outputs (placeholder - actual processing happens on backend)
        this.setOutputData(0, null); // Generated images
        this.setOutputData(1, null); // Edited regions
        this.setOutputData(2, params); // Metadata
    }

    onDrawBackground(ctx) {
        // Draw task mode indicator
        if (this.properties.task_mode !== "auto") {
            ctx.fillStyle = "#4a90e2";
            ctx.font = "12px Arial";
            ctx.textAlign = "right";
            ctx.fillText(this.properties.task_mode.toUpperCase(), this.size[0] - 10, 20);
        }

        // Draw model variant indicator
        if (this.properties.model_variant === "omnigen-v1-turbo") {
            ctx.fillStyle = "#ff6b6b";
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("TURBO", 10, this.size[1] - 10);
        }
    }

    getMenuOptions() {
        return [
            {
                content: "Reset to Defaults",
                callback: () => {
                    this.properties = {
                        model_variant: "omnigen-v1",
                        model_path: "",
                        task_mode: "auto",
                        instruction: "",
                        num_inference_steps: 50,
                        guidance_scale: 3.0,
                        resolution: 1024,
                        aspect_ratio: "1:1",
                        seed: -1,
                        batch_size: 1,
                        subject_mode: "single",
                        subject_strength: 0.8,
                        subject_token_mode: "auto",
                        edit_mode: "instruction",
                        preserve_unmasked: true,
                        edit_strength: 0.8,
                        image_fusion_mode: "guided",
                        cross_attention_scale: 1.0
                    };
                    this.setupWidgets();
                }
            },
            {
                content: "Copy Settings",
                callback: () => {
                    navigator.clipboard.writeText(JSON.stringify(this.properties, null, 2));
                }
            },
            {
                content: "Paste Settings",
                callback: () => {
                    navigator.clipboard.readText().then(text => {
                        try {
                            const settings = JSON.parse(text);
                            Object.assign(this.properties, settings);
                            this.setupWidgets();
                        } catch (e) {
                            console.error("Invalid settings format");
                        }
                    });
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
        // Auto-detect task mode based on connections
        if (this.properties.task_mode === "auto") {
            this.detectTaskMode();
        }
    }

    detectTaskMode() {
        const hasInstruction = this.getInputData(0) || this.properties.instruction;
        const hasImage1 = this.getInputData(1);
        const hasImage2 = this.getInputData(2);
        const hasImage3 = this.getInputData(3);
        const hasMask = this.getInputData(4);
        const hasSubjects = this.getInputData(5);

        let detectedMode = "text2img";

        if (hasSubjects) {
            detectedMode = "subject_driven";
        } else if (hasMask) {
            detectedMode = "editing";
        } else if (hasImage2 || hasImage3) {
            detectedMode = "multi_modal";
        } else if (hasImage1) {
            detectedMode = "instruction";
        }

        // Update task mode indicator without changing property
        this.detectedTaskMode = detectedMode;
        this.setDirtyCanvas(true);
    }
}

// Register the node
LiteGraph.registerNodeType("sd_forms/omnigen", OmniGenNode);

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OmniGenNode;
}