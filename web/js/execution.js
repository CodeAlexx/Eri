/**
 * SD Forms Execution Module with Error Handling
 * Converts LiteGraph format to SD Forms workflow format for backend execution
 * Includes comprehensive error handling, validation, and node highlighting
 */

class WorkflowExecutor {
    constructor() {
        this.api = null; // Will be injected by main app
        this.isExecuting = false;
        this.executionResults = new Map();
        this.abortController = null;
    }

    /**
     * Convert LiteGraph serialized data to SD Forms ExecuteRequest format
     * @param {Object} graphData - LiteGraph.serialize() output
     * @param {Object} settings - Additional execution settings
     * @returns {Object} ExecuteRequest compatible object
     */
    convertLiteGraphToWorkflow(graphData, settings = {}) {
        console.log('🔄 Converting LiteGraph data to SD Forms workflow format...');
        
        const workflow = this.extractWorkflowFromGraph(graphData);
        
        // Create ExecuteRequest structure matching backend/server.py
        const executeRequest = {
            workflow: {
                components: workflow.components,
                connections: workflow.connections
            },
            settings: settings
        };

        console.log('✅ Conversion complete:', {
            components: executeRequest.workflow.components.length,
            connections: executeRequest.workflow.connections.length
        });

        return executeRequest;
    }

    /**
     * Extract workflow data from LiteGraph serialized format
     * @param {Object} graphData - LiteGraph.serialize() output
     * @returns {Object} Workflow with components and connections
     */
    extractWorkflowFromGraph(graphData) {
        const components = [];
        const connections = [];

        // Convert nodes to WorkflowComponent format
        if (graphData.nodes) {
            graphData.nodes.forEach(node => {
                const component = this.convertNodeToComponent(node);
                if (component) {
                    components.push(component);
                }
            });
        }

        // Convert links to WorkflowConnection format
        if (graphData.links) {
            graphData.links.forEach(link => {
                const connection = this.convertLinkToConnection(link, graphData.nodes);
                if (connection) {
                    connections.push(connection);
                }
            });
        }

        return { components, connections };
    }

    /**
     * Convert LiteGraph node to SD Forms WorkflowComponent
     * @param {Object} node - LiteGraph node object
     * @returns {Object} WorkflowComponent matching backend/server.py structure
     */
    convertNodeToComponent(node) {
        if (!node.type || !node.type.startsWith('sdforms/')) {
            console.warn('Skipping non-SD Forms node:', node.type);
            return null;
        }

        // Extract component type from LiteGraph type (remove 'sdforms/' prefix)
        const componentType = node.type.replace('sdforms/', '');

        // Use node's componentId if available, otherwise generate one
        const componentId = node.componentId || this.generateComponentId();

        // Extract properties from node
        const properties = this.extractNodeProperties(node);

        // Extract position
        const position = {
            x: node.pos ? node.pos[0] : 0,
            y: node.pos ? node.pos[1] : 0
        };

        // Create component matching WorkflowComponent structure
        const component = {
            id: componentId,
            type: componentType,
            properties: properties,
            position: position
        };

        console.log(`📦 Converted node to component:`, {
            id: component.id,
            type: component.type,
            propertyCount: Object.keys(component.properties).length
        });

        return component;
    }

    /**
     * Extract all properties from a LiteGraph node
     * @param {Object} node - LiteGraph node object
     * @returns {Object} Properties object
     */
    extractNodeProperties(node) {
        const properties = {};

        // Extract from node.properties (main property storage)
        if (node.properties) {
            Object.assign(properties, node.properties);
        }

        // Extract from node.widgets if they exist and have values
        if (node.widgets) {
            node.widgets.forEach(widget => {
                if (widget.name && widget.value !== undefined) {
                    properties[widget.name] = widget.value;
                }
            });
        }

        // Extract any additional property-like fields
        const propertyFields = ['title', 'mode', 'flags'];
        propertyFields.forEach(field => {
            if (node[field] !== undefined && field !== 'title') {
                properties[field] = node[field];
            }
        });

        return properties;
    }

    /**
     * Convert LiteGraph link to SD Forms WorkflowConnection
     * @param {Array} link - LiteGraph link array [id, origin_id, origin_slot, target_id, target_slot, type]
     * @param {Array} nodes - Array of nodes to resolve port names
     * @returns {Object} WorkflowConnection matching backend/server.py structure
     */
    convertLinkToConnection(link, nodes) {
        if (!link || link.length < 5) {
            console.warn('Invalid link format:', link);
            return null;
        }

        const [linkId, originId, originSlot, targetId, targetSlot, linkType] = link;

        // Find origin and target nodes
        const originNode = nodes.find(n => n.id === originId);
        const targetNode = nodes.find(n => n.id === targetId);

        if (!originNode || !targetNode) {
            console.warn('Could not find nodes for link:', { originId, targetId });
            return null;
        }

        // Get port names from nodes
        const fromPort = this.getOutputPortName(originNode, originSlot);
        const toPort = this.getInputPortName(targetNode, targetSlot);

        if (!fromPort || !toPort) {
            console.warn('Could not resolve port names for link:', { originSlot, targetSlot });
            return null;
        }

        // Get component IDs
        const fromComponentId = originNode.componentId || this.generateComponentId();
        const toComponentId = targetNode.componentId || this.generateComponentId();

        // Create connection matching WorkflowConnection structure
        const connection = {
            id: linkId.toString(),
            from_component: fromComponentId,
            from_port: fromPort,
            to_component: toComponentId,
            to_port: toPort
        };

        console.log(`🔗 Converted link to connection:`, {
            from: `${connection.from_component}.${connection.from_port}`,
            to: `${connection.to_component}.${connection.to_port}`
        });

        return connection;
    }

    /**
     * Get output port name from node and slot index
     * @param {Object} node - LiteGraph node
     * @param {number} slotIndex - Output slot index
     * @returns {string} Port name
     */
    getOutputPortName(node, slotIndex) {
        if (node.outputs && node.outputs[slotIndex]) {
            return node.outputs[slotIndex].name;
        }

        // Fallback to generic names
        return `output_${slotIndex}`;
    }

    /**
     * Get input port name from node and slot index
     * @param {Object} node - LiteGraph node
     * @param {number} slotIndex - Input slot index
     * @returns {string} Port name
     */
    getInputPortName(node, slotIndex) {
        if (node.inputs && node.inputs[slotIndex]) {
            return node.inputs[slotIndex].name;
        }

        // Fallback to generic names
        return `input_${slotIndex}`;
    }

    /**
     * Generate a unique component ID
     * @returns {string} Unique ID
     */
    generateComponentId() {
        return 'comp_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Execute workflow using the API service
     * @param {LGraph} graph - LiteGraph instance
     * @param {Object} settings - Execution settings
     * @returns {Promise} Execution result
     */
    async executeWorkflow(graph, settings = {}) {
        if (!this.api) {
            throw new Error('API service not configured');
        }

        console.log('🚀 Starting workflow execution...');

        // Serialize the graph
        const graphData = graph.serialize();
        
        // Convert to SD Forms format
        const executeRequest = this.convertLiteGraphToWorkflow(graphData, settings);

        // Validate the workflow
        const validation = this.validateWorkflow(executeRequest.workflow);
        if (!validation.valid) {
            throw new Error(`Workflow validation failed: ${validation.errors.join(', ')}`);
        }

        // Execute via API
        try {
            console.log('📤 Sending execution request to backend...');
            const result = await this.api.executeWorkflow(executeRequest);
            console.log('✅ Workflow execution completed:', result);
            return result;
        } catch (error) {
            console.error('❌ Workflow execution failed:', error);
            throw error;
        }
    }

    /**
     * Validate workflow before execution
     * @param {Object} workflow - Workflow object with components and connections
     * @returns {Object} Validation result with valid flag and errors array
     */
    validateWorkflow(workflow) {
        const errors = [];

        // Check for components
        if (!workflow.components || workflow.components.length === 0) {
            errors.push('Workflow must contain at least one component');
        }

        // Validate components
        workflow.components.forEach((component, index) => {
            if (!component.id) {
                errors.push(`Component ${index} missing id`);
            }
            if (!component.type) {
                errors.push(`Component ${index} missing type`);
            }
            if (!component.properties) {
                errors.push(`Component ${index} missing properties`);
            }
            if (!component.position) {
                errors.push(`Component ${index} missing position`);
            }
        });

        // Validate connections
        if (workflow.connections) {
            workflow.connections.forEach((connection, index) => {
                if (!connection.id) {
                    errors.push(`Connection ${index} missing id`);
                }
                if (!connection.from_component) {
                    errors.push(`Connection ${index} missing from_component`);
                }
                if (!connection.from_port) {
                    errors.push(`Connection ${index} missing from_port`);
                }
                if (!connection.to_component) {
                    errors.push(`Connection ${index} missing to_component`);
                }
                if (!connection.to_port) {
                    errors.push(`Connection ${index} missing to_port`);
                }

                // Check if referenced components exist
                const fromComponentExists = workflow.components.some(c => c.id === connection.from_component);
                const toComponentExists = workflow.components.some(c => c.id === connection.to_component);
                
                if (!fromComponentExists) {
                    errors.push(`Connection ${index} references non-existent from_component: ${connection.from_component}`);
                }
                if (!toComponentExists) {
                    errors.push(`Connection ${index} references non-existent to_component: ${connection.to_component}`);
                }
            });
        }

        const result = {
            valid: errors.length === 0,
            errors: errors
        };

        if (!result.valid) {
            console.warn('❌ Workflow validation failed:', result.errors);
            
            // Use error system for better error reporting
            if (window.errorSystem) {
                window.errorSystem.showValidationError(
                    `Validation Failed (${result.errors.length} issues)`,
                    result.errors.map(e => `• ${e}`).join('\n')
                );
            }
        } else {
            console.log('✅ Workflow validation passed');
        }

        return result;
    }

    /**
     * Cancel currently running execution
     * @returns {Promise} Cancellation result
     */
    async cancelExecution() {
        if (!this.api) {
            throw new Error('API service not configured');
        }

        console.log('🛑 Cancelling workflow execution...');
        return await this.api.cancelExecution();
    }

    /**
     * Set the API service instance
     * @param {ApiService} apiService - API service instance
     */
    setApiService(apiService) {
        this.api = apiService;
    }

    /**
     * Create ExecuteRequest from current graph state
     * @param {LGraph} graph - LiteGraph instance
     * @param {Object} settings - Additional settings
     * @returns {ExecutionRequest} Ready-to-send execution request
     */
    createExecutionRequest(graph, settings = {}) {
        const graphData = graph.serialize();
        const executeRequest = this.convertLiteGraphToWorkflow(graphData, settings);
        
        // Wrap in ExecutionRequest class if available
        if (typeof ExecutionRequest !== 'undefined') {
            const workflow = new Workflow(
                executeRequest.workflow.components.map(c => new ComponentInstance(
                    c.id, c.type, c.properties, 
                    new ComponentPosition(c.position.x, c.position.y)
                )),
                executeRequest.workflow.connections.map(c => new Connection(
                    c.id, c.from_component, c.from_port, c.to_component, c.to_port
                ))
            );
            
            return new ExecutionRequest(workflow, executeRequest.settings);
        }
        
        return executeRequest;
    }

    /**
     * Debug method to log the conversion process
     * @param {LGraph} graph - LiteGraph instance
     */
    debugConversion(graph) {
        console.log('🔍 Debug: Converting LiteGraph to SD Forms format...');
        
        const graphData = graph.serialize();
        console.log('📊 Original LiteGraph data:', graphData);
        
        const executeRequest = this.convertLiteGraphToWorkflow(graphData);
        console.log('🔄 Converted SD Forms data:', executeRequest);
        
        const validation = this.validateWorkflow(executeRequest.workflow);
        console.log('✓ Validation result:', validation);
        
        return {
            original: graphData,
            converted: executeRequest,
            validation: validation
        };
    }
}

// Create singleton instance
const workflowExecutor = new WorkflowExecutor();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WorkflowExecutor,
        workflowExecutor
    };
} else {
    // Browser globals
    window.WorkflowExecutor = WorkflowExecutor;
    window.workflowExecutor = workflowExecutor;
}