/**
 * Workflow Management System
 * Handles save/load functionality, templates, and Flutter workflow import
 */

class WorkflowManager {
    constructor(graph) {
        this.graph = graph;
        this.currentWorkflow = null;
        this.workflows = new Map();
        this.templates = new Map();
        
        this.init();
        this.loadTemplates();
    }

    init() {
        // Load saved workflows from localStorage
        this.loadSavedWorkflows();
        
        // Set up auto-save
        this.setupAutoSave();
    }

    /**
     * Save current workflow to JSON
     */
    saveWorkflow(name, description = '') {
        try {
            const workflowData = this.serializeWorkflow();
            
            const workflow = {
                id: this.generateId(),
                name: name,
                description: description,
                data: workflowData,
                created: new Date().toISOString(),
                modified: new Date().toISOString(),
                version: '1.0'
            };

            this.workflows.set(workflow.id, workflow);
            this.currentWorkflow = workflow;
            
            // Save to localStorage
            this.saveToStorage();
            
            console.log(`Workflow "${name}" saved successfully`);
            return workflow;
            
        } catch (error) {
            console.error('Error saving workflow:', error);
            throw new Error(`Failed to save workflow: ${error.message}`);
        }
    }

    /**
     * Load workflow from saved data
     */
    async loadWorkflow(workflowId) {
        try {
            const workflow = this.workflows.get(workflowId);
            if (!workflow) {
                throw new Error(`Workflow with ID ${workflowId} not found`);
            }

            // Clear current graph
            this.graph.clear();
            
            // Deserialize workflow data
            await this.deserializeWorkflow(workflow.data);
            
            this.currentWorkflow = workflow;
            
            console.log(`Workflow "${workflow.name}" loaded successfully`);
            return workflow;
            
        } catch (error) {
            console.error('Error loading workflow:', error);
            throw new Error(`Failed to load workflow: ${error.message}`);
        }
    }

    /**
     * Serialize current graph to JSON
     */
    serializeWorkflow() {
        const data = this.graph.serialize();
        
        // Add metadata
        const workflowData = {
            graph: data,
            metadata: {
                nodeCount: this.graph._nodes.length,
                timestamp: Date.now(),
                version: '1.0',
                creator: 'SD Forms LiteGraph'
            }
        };

        return workflowData;
    }

    /**
     * Deserialize JSON to graph
     */
    async deserializeWorkflow(workflowData) {
        try {
            // Validate workflow data
            if (!workflowData || !workflowData.graph) {
                throw new Error('Invalid workflow data format');
            }

            // Configure graph before loading
            this.graph.configure(workflowData.graph);
            
            // Post-process nodes if needed
            await this.postProcessNodes();
            
            // Trigger graph update
            this.graph.setDirtyCanvas(true, true);
            
        } catch (error) {
            console.error('Error deserializing workflow:', error);
            throw error;
        }
    }

    /**
     * Post-process nodes after loading
     */
    async postProcessNodes() {
        // Wait for nodes to be fully initialized
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Update node properties and connections
        for (const node of this.graph._nodes) {
            if (node.onLoaded) {
                node.onLoaded();
            }
            
            // Refresh node widgets
            if (node.widgets) {
                node.widgets.forEach(widget => {
                    if (widget.callback) {
                        widget.callback(widget.value, null, node, null, widget);
                    }
                });
            }
        }
    }

    /**
     * Export workflow to file
     */
    exportWorkflow(workflowId, format = 'json') {
        try {
            const workflow = this.workflows.get(workflowId);
            if (!workflow) {
                throw new Error('Workflow not found');
            }

            let data, filename, mimeType;

            switch (format.toLowerCase()) {
                case 'json':
                    data = JSON.stringify(workflow, null, 2);
                    filename = `${workflow.name}.json`;
                    mimeType = 'application/json';
                    break;
                    
                case 'workflow':
                    // Custom workflow format
                    data = JSON.stringify({
                        name: workflow.name,
                        description: workflow.description,
                        data: workflow.data,
                        exported: new Date().toISOString()
                    }, null, 2);
                    filename = `${workflow.name}.workflow`;
                    mimeType = 'application/json';
                    break;
                    
                default:
                    throw new Error(`Unsupported export format: ${format}`);
            }

            this.downloadFile(data, filename, mimeType);
            
        } catch (error) {
            console.error('Error exporting workflow:', error);
            throw error;
        }
    }

    /**
     * Import workflow from file
     */
    async importWorkflow(file) {
        try {
            const content = await this.readFile(file);
            const data = JSON.parse(content);
            
            // Detect file type and import accordingly
            if (this.isFlutterWorkflow(data)) {
                return await this.importFlutterWorkflow(data);
            } else if (this.isLiteGraphWorkflow(data)) {
                return await this.importLiteGraphWorkflow(data);
            } else {
                throw new Error('Unrecognized workflow format');
            }
            
        } catch (error) {
            console.error('Error importing workflow:', error);
            throw error;
        }
    }

    /**
     * Import Flutter workflow and convert to LiteGraph format
     */
    async importFlutterWorkflow(flutterData) {
        try {
            console.log('Importing Flutter workflow...');
            
            // Map Flutter components to LiteGraph nodes
            const convertedData = await this.convertFlutterToLiteGraph(flutterData);
            
            // Create new workflow
            const workflow = {
                id: this.generateId(),
                name: flutterData.name || 'Imported Flutter Workflow',
                description: flutterData.description || 'Imported from Flutter frontend',
                data: convertedData,
                created: new Date().toISOString(),
                modified: new Date().toISOString(),
                version: '1.0',
                imported: true,
                source: 'flutter'
            };

            this.workflows.set(workflow.id, workflow);
            this.saveToStorage();
            
            console.log('Flutter workflow imported successfully');
            return workflow;
            
        } catch (error) {
            console.error('Error importing Flutter workflow:', error);
            throw error;
        }
    }

    /**
     * Convert Flutter workflow format to LiteGraph format
     */
    async convertFlutterToLiteGraph(flutterData) {
        const nodeMapping = {
            'ModelComponent': 'sd_forms/model',
            'SamplerComponent': 'sd_forms/sampler', 
            'VAEComponent': 'sd_forms/vae',
            'OutputComponent': 'sd_forms/output',
            'LoRAComponent': 'sd_forms/lora',
            'UpscalerComponent': 'sd_forms/upscaler'
        };

        const nodes = [];
        const links = [];
        let nodeId = 1;
        let linkId = 1;

        // Convert components to nodes
        if (flutterData.components) {
            for (const component of flutterData.components) {
                const nodeType = nodeMapping[component.type] || component.type;
                
                const node = {
                    id: nodeId++,
                    type: nodeType,
                    pos: component.position || [100 * nodes.length, 100],
                    size: component.size || [200, 100],
                    flags: {},
                    order: component.order || nodes.length,
                    mode: 0,
                    inputs: this.convertFlutterInputs(component.inputs),
                    outputs: this.convertFlutterOutputs(component.outputs),
                    properties: component.properties || {},
                    widgets_values: this.extractWidgetValues(component.properties)
                };

                nodes.push(node);
            }
        }

        // Convert connections to links
        if (flutterData.connections) {
            for (const connection of flutterData.connections) {
                const link = {
                    id: linkId++,
                    origin_id: connection.sourceId,
                    origin_slot: connection.sourcePort || 0,
                    target_id: connection.targetId,
                    target_slot: connection.targetPort || 0,
                    type: connection.type || "number"
                };
                
                links.push(link);
            }
        }

        return {
            graph: {
                last_node_id: nodeId,
                last_link_id: linkId,
                nodes: nodes,
                links: links,
                groups: [],
                config: {},
                extra: {},
                version: 0.4
            },
            metadata: {
                converted: true,
                originalFormat: 'flutter',
                timestamp: Date.now()
            }
        };
    }

    /**
     * Convert Flutter inputs to LiteGraph format
     */
    convertFlutterInputs(inputs) {
        if (!inputs) return [];
        
        return inputs.map((input, index) => ({
            name: input.name || `input_${index}`,
            type: input.type || "number",
            link: null
        }));
    }

    /**
     * Convert Flutter outputs to LiteGraph format
     */
    convertFlutterOutputs(outputs) {
        if (!outputs) return [];
        
        return outputs.map((output, index) => ({
            name: output.name || `output_${index}`,
            type: output.type || "number",
            links: []
        }));
    }

    /**
     * Extract widget values from properties
     */
    extractWidgetValues(properties) {
        if (!properties) return [];
        
        const values = [];
        Object.entries(properties).forEach(([key, value]) => {
            values.push(value);
        });
        
        return values;
    }

    /**
     * Import LiteGraph workflow
     */
    async importLiteGraphWorkflow(data) {
        const workflow = {
            id: this.generateId(),
            name: data.name || 'Imported Workflow',
            description: data.description || 'Imported LiteGraph workflow',
            data: data.data || data,
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            version: '1.0',
            imported: true,
            source: 'litegraph'
        };

        this.workflows.set(workflow.id, workflow);
        this.saveToStorage();
        
        return workflow;
    }

    /**
     * Check if data is Flutter workflow format
     */
    isFlutterWorkflow(data) {
        return data.components && Array.isArray(data.components) &&
               data.connections && Array.isArray(data.connections);
    }

    /**
     * Check if data is LiteGraph workflow format
     */
    isLiteGraphWorkflow(data) {
        const graphData = data.data?.graph || data.graph || data;
        return graphData && 
               (graphData.nodes || graphData._nodes) && 
               (graphData.links || graphData._links);
    }

    /**
     * Load predefined workflow templates
     */
    loadTemplates() {
        const templates = [
            {
                id: 'basic_generation',
                name: 'Basic Image Generation',
                description: 'Simple text-to-image generation workflow',
                category: 'basic',
                data: this.createBasicGenerationTemplate()
            },
            {
                id: 'img2img',
                name: 'Image-to-Image',
                description: 'Transform existing images with AI',
                category: 'basic',
                data: this.createImg2ImgTemplate()
            },
            {
                id: 'lora_generation',
                name: 'LoRA Enhanced Generation',
                description: 'Generation with LoRA model enhancement',
                category: 'advanced',
                data: this.createLoRATemplate()
            },
            {
                id: 'upscaling_workflow',
                name: 'Upscaling Workflow',
                description: 'Generate and upscale images',
                category: 'advanced',
                data: this.createUpscalingTemplate()
            },
            {
                id: 'controlnet_workflow',
                name: 'ControlNet Workflow',
                description: 'Guided generation with ControlNet',
                category: 'advanced',
                data: this.createControlNetTemplate()
            }
        ];

        templates.forEach(template => {
            this.templates.set(template.id, template);
        });
    }

    /**
     * Create basic generation template
     */
    createBasicGenerationTemplate() {
        return {
            graph: {
                last_node_id: 4,
                last_link_id: 3,
                nodes: [
                    {
                        id: 1,
                        type: "sd_forms/model",
                        pos: [50, 100],
                        size: [200, 100],
                        properties: {
                            model_path: "",
                            prompt: "a beautiful landscape"
                        }
                    },
                    {
                        id: 2,
                        type: "sd_forms/sampler",
                        pos: [300, 100],
                        size: [200, 120],
                        properties: {
                            steps: 20,
                            cfg_scale: 7.0,
                            scheduler: "DPMSolverMultistepScheduler"
                        }
                    },
                    {
                        id: 3,
                        type: "sd_forms/output",
                        pos: [550, 100],
                        size: [150, 80],
                        properties: {
                            format: "png",
                            quality: 95
                        }
                    }
                ],
                links: [
                    { id: 1, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
                    { id: 2, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0 }
                ]
            }
        };
    }

    /**
     * Create image-to-image template
     */
    createImg2ImgTemplate() {
        return {
            graph: {
                last_node_id: 5,
                last_link_id: 4,
                nodes: [
                    {
                        id: 1,
                        type: "sd_forms/image_input",
                        pos: [50, 50],
                        size: [200, 100],
                        properties: {
                            image_path: ""
                        }
                    },
                    {
                        id: 2,
                        type: "sd_forms/model",
                        pos: [50, 200],
                        size: [200, 100],
                        properties: {
                            model_path: "",
                            prompt: "transform this image"
                        }
                    },
                    {
                        id: 3,
                        type: "sd_forms/sampler",
                        pos: [300, 150],
                        size: [200, 120],
                        properties: {
                            steps: 20,
                            cfg_scale: 7.0,
                            strength: 0.7
                        }
                    },
                    {
                        id: 4,
                        type: "sd_forms/output",
                        pos: [550, 150],
                        size: [150, 80]
                    }
                ],
                links: [
                    { id: 1, origin_id: 1, origin_slot: 0, target_id: 3, target_slot: 1 },
                    { id: 2, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0 },
                    { id: 3, origin_id: 3, origin_slot: 0, target_id: 4, target_slot: 0 }
                ]
            }
        };
    }

    /**
     * Create LoRA template
     */
    createLoRATemplate() {
        return {
            graph: {
                last_node_id: 5,
                last_link_id: 4,
                nodes: [
                    {
                        id: 1,
                        type: "sd_forms/model",
                        pos: [50, 100],
                        size: [200, 100],
                        properties: {
                            model_path: "",
                            prompt: "masterpiece, detailed"
                        }
                    },
                    {
                        id: 2,
                        type: "sd_forms/lora",
                        pos: [50, 250],
                        size: [200, 100],
                        properties: {
                            lora_path: "",
                            strength: 0.8
                        }
                    },
                    {
                        id: 3,
                        type: "sd_forms/sampler",
                        pos: [300, 150],
                        size: [200, 120],
                        properties: {
                            steps: 25,
                            cfg_scale: 8.0
                        }
                    },
                    {
                        id: 4,
                        type: "sd_forms/output",
                        pos: [550, 150],
                        size: [150, 80]
                    }
                ],
                links: [
                    { id: 1, origin_id: 1, origin_slot: 0, target_id: 3, target_slot: 0 },
                    { id: 2, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 1 },
                    { id: 3, origin_id: 3, origin_slot: 0, target_id: 4, target_slot: 0 }
                ]
            }
        };
    }

    /**
     * Create upscaling template
     */
    createUpscalingTemplate() {
        return {
            graph: {
                last_node_id: 5,
                last_link_id: 4,
                nodes: [
                    {
                        id: 1,
                        type: "sd_forms/model",
                        pos: [50, 100],
                        size: [200, 100]
                    },
                    {
                        id: 2,
                        type: "sd_forms/sampler",
                        pos: [300, 100],
                        size: [200, 120]
                    },
                    {
                        id: 3,
                        type: "sd_forms/upscaler",
                        pos: [550, 100],
                        size: [200, 120],
                        properties: {
                            scale_factor: 2.0,
                            model_name: "RealESRGAN_x2plus"
                        }
                    },
                    {
                        id: 4,
                        type: "sd_forms/output",
                        pos: [800, 100],
                        size: [150, 80]
                    }
                ],
                links: [
                    { id: 1, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
                    { id: 2, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0 },
                    { id: 3, origin_id: 3, origin_slot: 0, target_id: 4, target_slot: 0 }
                ]
            }
        };
    }

    /**
     * Create ControlNet template
     */
    createControlNetTemplate() {
        return {
            graph: {
                last_node_id: 6,
                last_link_id: 5,
                nodes: [
                    {
                        id: 1,
                        type: "sd_forms/image_input",
                        pos: [50, 50],
                        size: [200, 100]
                    },
                    {
                        id: 2,
                        type: "sd_forms/controlnet",
                        pos: [300, 50],
                        size: [200, 120],
                        properties: {
                            controlnet_type: "canny",
                            conditioning_scale: 1.0
                        }
                    },
                    {
                        id: 3,
                        type: "sd_forms/model",
                        pos: [50, 200],
                        size: [200, 100]
                    },
                    {
                        id: 4,
                        type: "sd_forms/sampler",
                        pos: [300, 200],
                        size: [200, 120]
                    },
                    {
                        id: 5,
                        type: "sd_forms/output",
                        pos: [550, 200],
                        size: [150, 80]
                    }
                ],
                links: [
                    { id: 1, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
                    { id: 2, origin_id: 2, origin_slot: 0, target_id: 4, target_slot: 1 },
                    { id: 3, origin_id: 3, origin_slot: 0, target_id: 4, target_slot: 0 },
                    { id: 4, origin_id: 4, origin_slot: 0, target_id: 5, target_slot: 0 }
                ]
            }
        };
    }

    /**
     * Create new workflow from template
     */
    createFromTemplate(templateId, name) {
        const template = this.templates.get(templateId);
        if (!template) {
            throw new Error(`Template ${templateId} not found`);
        }

        const workflow = {
            id: this.generateId(),
            name: name || template.name,
            description: template.description,
            data: JSON.parse(JSON.stringify(template.data)), // Deep clone
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            version: '1.0',
            template: templateId
        };

        this.workflows.set(workflow.id, workflow);
        this.saveToStorage();
        
        return workflow;
    }

    /**
     * Get all saved workflows
     */
    getWorkflows() {
        return Array.from(this.workflows.values());
    }

    /**
     * Get all templates grouped by category
     */
    getTemplates() {
        const templates = Array.from(this.templates.values());
        const grouped = {};
        
        templates.forEach(template => {
            const category = template.category || 'other';
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push(template);
        });
        
        return grouped;
    }

    /**
     * Delete workflow
     */
    deleteWorkflow(workflowId) {
        if (this.workflows.has(workflowId)) {
            this.workflows.delete(workflowId);
            this.saveToStorage();
            return true;
        }
        return false;
    }

    /**
     * Setup auto-save functionality
     */
    setupAutoSave() {
        let saveTimeout;
        
        // Auto-save on graph changes
        this.graph.onAfterStep = () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                if (this.currentWorkflow) {
                    this.autoSave();
                }
            }, 5000); // Auto-save after 5 seconds of inactivity
        };
    }

    /**
     * Auto-save current workflow
     */
    autoSave() {
        if (!this.currentWorkflow) return;
        
        try {
            const workflowData = this.serializeWorkflow();
            this.currentWorkflow.data = workflowData;
            this.currentWorkflow.modified = new Date().toISOString();
            
            this.saveToStorage();
            console.log('Workflow auto-saved');
            
        } catch (error) {
            console.error('Auto-save failed:', error);
        }
    }

    /**
     * Save workflows to localStorage
     */
    saveToStorage() {
        try {
            const data = {
                workflows: Array.from(this.workflows.entries()),
                timestamp: Date.now()
            };
            
            localStorage.setItem('sd_forms_workflows', JSON.stringify(data));
            
        } catch (error) {
            console.error('Error saving to storage:', error);
        }
    }

    /**
     * Load workflows from localStorage
     */
    loadSavedWorkflows() {
        try {
            const data = localStorage.getItem('sd_forms_workflows');
            if (data) {
                const parsed = JSON.parse(data);
                this.workflows = new Map(parsed.workflows || []);
            }
            
        } catch (error) {
            console.error('Error loading from storage:', error);
            this.workflows = new Map();
        }
    }

    /**
     * Utility functions
     */
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }
}

// Create global instance
window.workflowManager = null;

// Initialize when graph is available
function initializeWorkflowManager(graph) {
    window.workflowManager = new WorkflowManager(graph);
    return window.workflowManager;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WorkflowManager, initializeWorkflowManager };
}