/**
 * SD Forms Node Definitions for LiteGraph
 * Auto-generates nodes from backend component definitions
 */

class SDFormsNodes {
    static components = [];
    static registeredTypes = new Set();

    // Base node class for all SD Forms components
    static createNodeClass(component) {
        function SDFormsNode() {
            this.title = component.display_name;
            this.size = [250, 100]; // Will auto-resize based on content
            this.component_type = component.component_type;
            this.component_data = component;
            
            // Add input ports
            if (component.input_ports) {
                component.input_ports.forEach(port => {
                    this.addInput(port.name, port.data_type);
                });
            }
            
            // Add output ports
            if (component.output_ports) {
                component.output_ports.forEach(port => {
                    this.addOutput(port.name, port.data_type);
                });
            }
            
            // Initialize properties
            this.properties = {};
            if (component.property_definitions) {
                component.property_definitions.forEach(prop => {
                    this.properties[prop.name] = prop.default_value;
                });
            }
            
            // Add widgets for properties
            this.createWidgets();
            
            // Auto-resize based on content
            this.autoResize();
        }

        SDFormsNode.title = component.display_name;
        SDFormsNode.desc = component.description || "";
        
        // Create widgets for component properties
        SDFormsNode.prototype.createWidgets = function() {
            if (!this.component_data.property_definitions) return;
            
            this.component_data.property_definitions.forEach(prop => {
                this.addWidget(prop);
            });
        };

        // Add a widget based on property definition
        SDFormsNode.prototype.addWidget = function(propDef) {
            const name = propDef.name;
            const value = this.properties[name];
            const metadata = propDef.metadata || {};
            
            let widget;
            
            switch (propDef.property_type) {
                case 'STRING':
                    if (metadata.multiline || name.includes('prompt')) {
                        widget = this.addWidget("text", propDef.display_name, value, (v) => {
                            this.properties[name] = v;
                        }, { multiline: true });
                    } else {
                        widget = this.addWidget("text", propDef.display_name, value, (v) => {
                            this.properties[name] = v;
                        });
                    }
                    break;
                    
                case 'INTEGER':
                    widget = this.addWidget("number", propDef.display_name, value, (v) => {
                        this.properties[name] = parseInt(v);
                    }, { 
                        min: metadata.min || 1, 
                        max: metadata.max || 1000,
                        step: 1,
                        precision: 0
                    });
                    break;
                    
                case 'FLOAT':
                    widget = this.addWidget("number", propDef.display_name, value, (v) => {
                        this.properties[name] = parseFloat(v);
                    }, { 
                        min: metadata.min || 0.0, 
                        max: metadata.max || 100.0,
                        step: 0.1,
                        precision: 2
                    });
                    break;
                    
                case 'BOOLEAN':
                    widget = this.addWidget("toggle", propDef.display_name, value, (v) => {
                        this.properties[name] = v;
                    });
                    break;
                    
                case 'ENUM':
                    if (metadata.options) {
                        widget = this.addWidget("combo", propDef.display_name, value, (v) => {
                            this.properties[name] = v;
                        }, { values: metadata.options });
                    }
                    break;
                    
                case 'FILE_PATH':
                    widget = this.addWidget("text", propDef.display_name, value, (v) => {
                        this.properties[name] = v;
                    });
                    break;
                    
                default:
                    widget = this.addWidget("text", propDef.display_name, value, (v) => {
                        this.properties[name] = v;
                    });
            }
            
            if (widget && propDef.description) {
                widget.tooltip = propDef.description;
            }
        };

        // Auto-resize node based on content
        SDFormsNode.prototype.autoResize = function() {
            let height = 40; // Base height for title
            
            // Add height for inputs/outputs
            const maxPorts = Math.max(
                (this.inputs ? this.inputs.length : 0),
                (this.outputs ? this.outputs.length : 0)
            );
            height += maxPorts * 20;
            
            // Add height for widgets
            if (this.widgets) {
                height += this.widgets.length * 25;
            }
            
            // Set minimum height
            height = Math.max(height, 80);
            
            this.size[1] = height;
        };

        // Override to handle execution
        SDFormsNode.prototype.onExecute = function() {
            // Execution is handled by the main app
        };

        return SDFormsNode;
    }

    // Register all component types as LiteGraph nodes
    static registerComponents(components) {
        this.components = components;
        
        components.forEach(component => {
            const nodeType = `sdforms/${component.component_type}`;
            
            if (!this.registeredTypes.has(nodeType)) {
                const NodeClass = this.createNodeClass(component);
                LiteGraph.registerNodeType(nodeType, NodeClass);
                this.registeredTypes.add(nodeType);
                console.log(`✅ Registered node type: ${nodeType}`);
            }
        });
    }

    // Get component by type
    static getComponent(componentType) {
        return this.components.find(c => c.component_type === componentType);
    }

    // Convert LiteGraph graph to SD Forms workflow format
    static convertToWorkflow(graph) {
        const workflow = {
            components: [],
            connections: []
        };

        // Convert nodes to components
        for (let i = 0; i < graph._nodes.length; i++) {
            const node = graph._nodes[i];
            
            const component = {
                id: node.id.toString(),
                component_type: node.component_type,
                properties: { ...node.properties },
                position: {
                    x: node.pos[0],
                    y: node.pos[1]
                }
            };
            
            workflow.components.push(component);
        }

        // Convert connections
        for (let i = 0; i < graph._nodes.length; i++) {
            const node = graph._nodes[i];
            
            if (node.outputs) {
                for (let j = 0; j < node.outputs.length; j++) {
                    const output = node.outputs[j];
                    
                    if (output.links) {
                        for (let k = 0; k < output.links.length; k++) {
                            const linkId = output.links[k];
                            const link = graph.links[linkId];
                            
                            if (link) {
                                const targetNode = graph._nodes_by_id[link.target_id];
                                const targetInput = targetNode.inputs[link.target_slot];
                                
                                const connection = {
                                    from_component: node.id.toString(),
                                    from_port: output.name,
                                    to_component: link.target_id.toString(),
                                    to_port: targetInput.name
                                };
                                
                                workflow.connections.push(connection);
                            }
                        }
                    }
                }
            }
        }

        return workflow;
    }

    // Create basic SDXL workflow for testing
    static createBasicWorkflow(graph) {
        graph.clear();

        // Create Model node
        const modelNode = LiteGraph.createNode("sdforms/model");
        if (modelNode) {
            modelNode.pos = [50, 100];
            modelNode.properties.model_path = "stabilityai/stable-diffusion-xl-base-1.0";
            modelNode.properties.prompt = "A beautiful landscape with mountains and a lake, highly detailed, photorealistic";
            modelNode.properties.negative_prompt = "blurry, low quality, distorted";
            graph.add(modelNode);
        }

        // Create Sampler node
        const samplerNode = LiteGraph.createNode("sdforms/sampler");
        if (samplerNode) {
            samplerNode.pos = [350, 100];
            samplerNode.properties.scheduler = "DPMSolverMultistep";
            samplerNode.properties.num_inference_steps = 25;
            samplerNode.properties.guidance_scale = 7.0;
            samplerNode.properties.width = 1024;
            samplerNode.properties.height = 1024;
            graph.add(samplerNode);
        }

        // Create Output node
        const outputNode = LiteGraph.createNode("sdforms/output");
        if (outputNode) {
            outputNode.pos = [650, 100];
            outputNode.properties.output_format = "PNG";
            outputNode.properties.save_path = "output/";
            outputNode.properties.filename_prefix = "sdxl_test";
            graph.add(outputNode);
        }

        // Connect the nodes
        if (modelNode && samplerNode && outputNode) {
            // Connect model to sampler
            if (modelNode.outputs.length > 0 && samplerNode.inputs.length > 0) {
                modelNode.connect(0, samplerNode, 0);
            }
            if (modelNode.outputs.length > 1 && samplerNode.inputs.length > 1) {
                modelNode.connect(1, samplerNode, 1);
            }
            
            // Connect sampler to output
            if (samplerNode.outputs.length > 0 && outputNode.inputs.length > 0) {
                samplerNode.connect(0, outputNode, 0);
            }
        }

        console.log("✅ Basic SDXL workflow created");
        return { modelNode, samplerNode, outputNode };
    }

    // Add nodes to context menu
    static setupContextMenu(graph) {
        const original_getMenuOptions = LiteGraph.ContextMenu;
        
        // Override the context menu to add our nodes
        graph.onNodeTypesList = function(position) {
            const options = [];
            
            // Add SD Forms components by category
            const categories = {
                'Core': ['model', 'sampler', 'output', 'vae'],
                'Models': ['sdxl_model', 'omnigen', 'lumina'],
                'Enhancement': ['upscaler', 'controlnet'],
                'Training': ['simpletuner']
            };
            
            Object.entries(categories).forEach(([category, types]) => {
                const categoryOptions = types
                    .filter(type => SDFormsNodes.registeredTypes.has(`sdforms/${type}`))
                    .map(type => {
                        const component = SDFormsNodes.getComponent(type);
                        return {
                            content: component ? component.display_name : type,
                            value: `sdforms/${type}`
                        };
                    });
                
                if (categoryOptions.length > 0) {
                    options.push({
                        content: category,
                        submenu: {
                            options: categoryOptions
                        }
                    });
                }
            });
            
            return options;
        };
    }
}