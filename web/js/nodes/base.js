/**
 * Base LiteGraph Node for SD Forms Components
 * Wraps SD Forms components as LiteGraph nodes with automatic UI generation
 */

// Widget type mappings from SD Forms PropertyType to LiteGraph widgets
const PROPERTY_TYPE_MAPPINGS = {
    'string': 'text',
    'integer': 'number', 
    'float': 'number',
    'boolean': 'toggle',
    'file_path': 'text',
    'directory': 'text',
    'choice': 'combo',
    'slider': 'slider',
    'color': 'text',
    'text': 'textarea'
};

// Port type color mappings for visual consistency
const PORT_TYPE_COLORS = {
    'any': '#999999',
    'image': '#4CAF50',
    'text': '#2196F3', 
    'tensor': '#FF9800',
    'model': '#9C27B0',
    'pipeline': '#E91E63',
    'conditioning': '#00BCD4',
    'latent': '#795548',
    'vae': '#3F51B5',
    'clip': '#607D8B'
};

/**
 * Base class for SD Forms component nodes
 * Automatically maps component structure to LiteGraph node interface
 */
class SDFormsNode extends LGraphNode {
    constructor(componentInfo = null) {
        super();
        
        // Store component information
        this.componentInfo = componentInfo;
        this.componentInstance = null;
        this.propertyValues = {};
        
        if (componentInfo) {
            this.setupFromComponentInfo(componentInfo);
        }
        
        // Track property changes for API synchronization
        this.propertyChangeHandlers = new Map();
        
        // Node appearance
        this.shape = LiteGraph.BOX_SHAPE;
        this.color = "#222";
        this.bgcolor = "#000";
        this.boxcolor = "#666";
        
        // Processing state
        this.isProcessing = false;
        this.hasError = false;
        this.status = 'idle';
        
        // Bind methods
        this.onPropertyChanged = this.onPropertyChanged.bind(this);
        this.onExecute = this.onExecute.bind(this);
    }
    
    /**
     * Set up node from SD Forms ComponentInfo
     */
    setupFromComponentInfo(componentInfo) {
        // Set basic node properties
        this.title = componentInfo.display_name || componentInfo.type;
        this.size = [200, 100]; // Default size, will be adjusted based on content
        
        // Store component type and metadata
        this.componentType = componentInfo.type;
        this.componentId = componentInfo.id;
        this.category = componentInfo.category;
        this.icon = componentInfo.icon;
        
        // Set up input ports
        this.setupInputPorts(componentInfo.input_ports || []);
        
        // Set up output ports  
        this.setupOutputPorts(componentInfo.output_ports || []);
        
        // Set up property widgets
        this.setupPropertyWidgets(componentInfo.property_definitions || []);
        
        // Create component instance for data model
        this.createComponentInstance();
        
        // Adjust size based on content
        this.adjustNodeSize();
    }
    
    /**
     * Set up input ports from component port definitions
     */
    setupInputPorts(inputPorts) {
        for (const port of inputPorts) {
            const portType = this.getPortTypeId(port.type);
            const color = PORT_TYPE_COLORS[port.type] || PORT_TYPE_COLORS.any;
            
            this.addInput(port.name, portType, {
                color: color,
                optional: port.optional || false,
                multiple: port.multiple || false,
                ...port.metadata
            });
        }
    }
    
    /**
     * Set up output ports from component port definitions
     */
    setupOutputPorts(outputPorts) {
        for (const port of outputPorts) {
            const portType = this.getPortTypeId(port.type);
            const color = PORT_TYPE_COLORS[port.type] || PORT_TYPE_COLORS.any;
            
            this.addOutput(port.name, portType, {
                color: color,
                multiple: port.multiple || false,
                ...port.metadata
            });
        }
    }
    
    /**
     * Set up property widgets from component property definitions
     */
    setupPropertyWidgets(propertyDefs) {
        this.properties = {};
        
        // Group properties by category for better organization
        const categorizedProps = this.categorizeProperties(propertyDefs);
        
        for (const [category, props] of Object.entries(categorizedProps)) {
            // Add category separator if multiple categories
            if (Object.keys(categorizedProps).length > 1) {
                this.addWidget("text", `--- ${category} ---`, "", null, {
                    property: "__category_separator",
                    disabled: true
                });
            }
            
            for (const propDef of props) {
                this.createPropertyWidget(propDef);
            }
        }
    }
    
    /**
     * Group properties by category
     */
    categorizeProperties(propertyDefs) {
        const categorized = {};
        
        for (const propDef of propertyDefs) {
            const category = propDef.category || 'General';
            if (!categorized[category]) {
                categorized[category] = [];
            }
            categorized[category].push(propDef);
        }
        
        return categorized;
    }
    
    /**
     * Create a widget for a single property definition
     */
    createPropertyWidget(propDef) {
        const widgetType = this.getWidgetType(propDef);
        const defaultValue = propDef.default_value;
        
        // Store default in properties
        this.properties[propDef.name] = defaultValue;
        this.propertyValues[propDef.name] = defaultValue;
        
        // Create widget options
        const options = {
            property: propDef.name,
            min: propDef.metadata.min,
            max: propDef.metadata.max,
            step: propDef.metadata.step,
            precision: propDef.metadata.precision,
            multiline: propDef.type === 'text',
            ...propDef.metadata
        };
        
        // Handle choice/combo widgets
        if (propDef.type === 'choice' && propDef.metadata.choices) {
            options.values = propDef.metadata.choices;
        }
        
        // Create widget with change callback
        const widget = this.addWidget(
            widgetType,
            propDef.display_name,
            defaultValue,
            (value, widget, node) => this.onPropertyChanged(propDef.name, value, widget),
            options
        );
        
        // Store reference for property lookups
        widget.property_name = propDef.name;
        widget.property_definition = propDef;
        
        return widget;
    }
    
    /**
     * Get LiteGraph widget type from SD Forms property type
     */
    getWidgetType(propDef) {
        const baseType = PROPERTY_TYPE_MAPPINGS[propDef.type] || 'text';
        
        // Handle special editor types
        if (propDef.metadata.editor_type) {
            switch (propDef.metadata.editor_type) {
                case 'float_slider':
                case 'int_slider':
                    return 'slider';
                case 'model_picker':
                case 'file_picker':
                    return 'text'; // TODO: Could create custom file picker widget
                case 'prompt':
                    return 'textarea';
                case 'lora_collection':
                    return 'text'; // TODO: Could create custom LoRA widget
                default:
                    return baseType;
            }
        }
        
        return baseType;
    }
    
    /**
     * Get LiteGraph port type ID from SD Forms port type
     */
    getPortTypeId(portType) {
        // LiteGraph uses strings for port types
        return portType.toLowerCase();
    }
    
    /**
     * Create component instance for data storage
     */
    createComponentInstance() {
        if (!this.componentInfo) return;
        
        this.componentInstance = new ComponentInstance(
            this.componentId || this.generateUniqueId(),
            this.componentType,
            { ...this.propertyValues },
            new ComponentPosition(this.pos[0], this.pos[1]),
            new ComponentSize(this.size[0], this.size[1])
        );
    }
    
    /**
     * Handle property value changes
     */
    onPropertyChanged(propertyName, value, widget) {
        // Update internal storage
        this.propertyValues[propertyName] = value;
        this.properties[propertyName] = value;
        
        // Update component instance
        if (this.componentInstance) {
            this.componentInstance.properties[propertyName] = value;
        }
        
        // Call any registered change handlers
        if (this.propertyChangeHandlers.has(propertyName)) {
            const handler = this.propertyChangeHandlers.get(propertyName);
            handler(value, widget, this);
        }
        
        // Mark node as modified
        this.setDirtyCanvas(true, false);
        
        // Trigger auto-configuration if enabled
        this.handleAutoConfiguration(propertyName, value);
    }
    
    /**
     * Handle auto-configuration based on property changes
     */
    handleAutoConfiguration(propertyName, value) {
        // Example: Auto-configure sampler settings when model type changes
        if (propertyName === 'checkpoint_type' || propertyName === 'checkpoint_path') {
            if (this.properties.auto_configure) {
                // Could trigger API call to detect model type and configure accordingly
                this.detectAndConfigureModel();
            }
        }
    }
    
    /**
     * Detect model type and configure settings
     */
    async detectAndConfigureModel() {
        // This would call the backend API to detect model type
        // and auto-configure optimal settings
        try {
            // Example implementation - would need actual API integration
            console.log('Auto-configuring model settings...');
        } catch (error) {
            console.error('Failed to auto-configure model:', error);
        }
    }
    
    /**
     * Adjust node size based on content
     */
    adjustNodeSize() {
        const baseHeight = 80;
        const widgetHeight = 25;
        const portHeight = 20;
        
        const inputCount = this.inputs ? this.inputs.length : 0;
        const outputCount = this.outputs ? this.outputs.length : 0;
        const widgetCount = this.widgets ? this.widgets.length : 0;
        
        const portSpace = Math.max(inputCount, outputCount) * portHeight;
        const widgetSpace = widgetCount * widgetHeight;
        
        const newHeight = baseHeight + Math.max(portSpace, widgetSpace);
        const newWidth = Math.max(200, this.title.length * 8 + 40);
        
        this.size = [newWidth, newHeight];
    }
    
    /**
     * Update visual state based on processing status
     */
    updateVisualState() {
        if (this.hasError) {
            this.color = "#660000";
            this.boxcolor = "#CC0000";
        } else if (this.isProcessing) {
            this.color = "#006600";  
            this.boxcolor = "#00CC00";
        } else if (this.status === 'complete') {
            this.color = "#000066";
            this.boxcolor = "#0000CC";
        } else {
            this.color = "#222";
            this.boxcolor = "#666";
        }
        
        this.setDirtyCanvas(true, false);
    }
    
    /**
     * Set processing status
     */
    setStatus(status, message = '') {
        this.status = status;
        this.isProcessing = status === 'processing';
        this.hasError = status === 'error';
        
        if (message) {
            this.statusMessage = message;
        }
        
        this.updateVisualState();
    }
    
    /**
     * Execute node (called during workflow execution)
     */
    onExecute() {
        // This will be called by LiteGraph during execution
        // For SD Forms, actual execution happens via API calls
        
        // Collect input data
        const inputData = {};
        if (this.inputs) {
            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                inputData[input.name] = this.getInputData(i);
            }
        }
        
        // Store input data for API execution
        if (this.componentInstance) {
            // Update component instance with current input data
            for (const [portName, data] of Object.entries(inputData)) {
                if (data !== undefined) {
                    this.componentInstance.inputs = this.componentInstance.inputs || {};
                    this.componentInstance.inputs[portName] = data;
                }
            }
        }
        
        // Output current property values (for immediate connections)
        if (this.outputs) {
            for (let i = 0; i < this.outputs.length; i++) {
                const output = this.outputs[i];
                // For most SD Forms nodes, outputs are generated during backend execution
                // But some might have immediate passthrough values
                if (this.componentInstance && this.componentInstance.outputs) {
                    this.setOutputData(i, this.componentInstance.outputs[output.name]);
                }
            }
        }
    }
    
    /**
     * Get component instance for workflow serialization
     */
    getComponentInstance() {
        // Update position and size
        if (this.componentInstance) {
            this.componentInstance.position.x = this.pos[0];
            this.componentInstance.position.y = this.pos[1];
            this.componentInstance.size.width = this.size[0];
            this.componentInstance.size.height = this.size[1];
        }
        
        return this.componentInstance;
    }
    
    /**
     * Serialize node state
     */
    serialize() {
        const data = super.serialize();
        
        // Add SD Forms specific data
        data.componentType = this.componentType;
        data.componentId = this.componentId;
        data.propertyValues = { ...this.propertyValues };
        data.status = this.status;
        
        return data;
    }
    
    /**
     * Configure node from serialized data
     */
    configure(data) {
        super.configure(data);
        
        // Restore SD Forms specific data
        this.componentType = data.componentType;
        this.componentId = data.componentId;
        this.status = data.status || 'idle';
        
        if (data.propertyValues) {
            this.propertyValues = { ...data.propertyValues };
            
            // Update widget values
            if (this.widgets) {
                for (const widget of this.widgets) {
                    if (widget.property_name && data.propertyValues.hasOwnProperty(widget.property_name)) {
                        widget.value = data.propertyValues[widget.property_name];
                    }
                }
            }
            
            // Update component instance
            if (this.componentInstance) {
                this.componentInstance.properties = { ...this.propertyValues };
            }
        }
        
        this.updateVisualState();
    }
    
    /**
     * Custom drawing for status indicators
     */
    onDrawForeground(ctx) {
        if (this.flags.collapsed) return;
        
        // Draw status indicator
        if (this.isProcessing || this.hasError || this.status === 'complete') {
            ctx.save();
            ctx.fillStyle = this.isProcessing ? "#00FF00" : 
                           this.hasError ? "#FF0000" : "#0000FF";
            ctx.beginPath();
            ctx.arc(this.size[0] - 15, 15, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
        
        // Draw category icon if available
        if (this.icon && !this.flags.collapsed) {
            ctx.save();
            ctx.font = "16px Arial";
            ctx.textAlign = "left";
            ctx.fillStyle = "#FFF";
            ctx.fillText(this.icon, 8, 20);
            ctx.restore();
        }
    }
    
    /**
     * Generate unique component ID
     */
    generateUniqueId() {
        return 'comp_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Add property change handler
     */
    addPropertyChangeHandler(propertyName, handler) {
        this.propertyChangeHandlers.set(propertyName, handler);
    }
    
    /**
     * Remove property change handler
     */
    removePropertyChangeHandler(propertyName) {
        this.propertyChangeHandlers.delete(propertyName);
    }
    
    /**
     * Get property value
     */
    getProperty(propertyName) {
        return this.propertyValues[propertyName];
    }
    
    /**
     * Set property value programmatically
     */
    setProperty(propertyName, value) {
        // Find widget and update it
        if (this.widgets) {
            for (const widget of this.widgets) {
                if (widget.property_name === propertyName) {
                    widget.value = value;
                    this.onPropertyChanged(propertyName, value, widget);
                    break;
                }
            }
        }
    }
}

// Export for use in node factory
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SDFormsNode;
} else {
    window.SDFormsNode = SDFormsNode;
}