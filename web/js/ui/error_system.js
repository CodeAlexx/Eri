/**
 * Comprehensive Error Handling System
 * Manages toast notifications, node error states, validation, and connection checking
 */

class ErrorSystem {
    constructor() {
        this.toastContainer = null;
        this.nodeErrors = new Map();
        this.validationErrors = new Map();
        this.connectionErrors = new Map();
        this.errorHistory = [];
        
        this.init();
    }

    init() {
        this.createToastContainer();
        this.setupGlobalErrorHandlers();
    }

    /**
     * Create toast notification container
     */
    createToastContainer() {
        this.toastContainer = document.createElement('div');
        this.toastContainer.id = 'toast-container';
        this.toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 400px;
            pointer-events: none;
        `;
        document.body.appendChild(this.toastContainer);
    }

    /**
     * Setup global error handlers
     */
    setupGlobalErrorHandlers() {
        // Catch uncaught JavaScript errors
        window.addEventListener('error', (event) => {
            this.showError('JavaScript Error', event.message, 'error');
            console.error('Uncaught error:', event.error);
        });

        // Catch unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.showError('Promise Rejection', event.reason.message || 'Unhandled promise rejection', 'error');
            console.error('Unhandled promise rejection:', event.reason);
        });

        // Catch fetch errors
        this.interceptFetch();
    }

    /**
     * Intercept fetch requests to handle API errors
     */
    interceptFetch() {
        const originalFetch = window.fetch;
        
        window.fetch = async (...args) => {
            try {
                const response = await originalFetch(...args);
                
                if (!response.ok) {
                    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                    
                    try {
                        const errorData = await response.clone().json();
                        if (errorData.error) {
                            errorMessage = errorData.error;
                        } else if (errorData.detail) {
                            errorMessage = errorData.detail;
                        }
                    } catch (e) {
                        // Response is not JSON, use status text
                    }
                    
                    this.showError('API Error', errorMessage, 'error');
                    throw new Error(errorMessage);
                }
                
                return response;
            } catch (error) {
                if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    this.showError('Network Error', 'Unable to connect to server', 'error');
                }
                throw error;
            }
        };
    }

    /**
     * Show toast notification
     */
    showToast(title, message, type = 'info', duration = 5000, actions = []) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const typeColors = {
            success: { bg: '#4caf50', icon: '✓' },
            error: { bg: '#f44336', icon: '✕' },
            warning: { bg: '#ff9800', icon: '⚠' },
            info: { bg: '#2196f3', icon: 'ℹ' },
            validation: { bg: '#9c27b0', icon: '⚡' }
        };
        
        const color = typeColors[type] || typeColors.info;
        
        toast.style.cssText = `
            background: ${color.bg};
            color: white;
            padding: 16px 20px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transform: translateX(100%);
            transition: transform 0.3s ease, opacity 0.3s ease;
            pointer-events: auto;
            cursor: pointer;
            max-width: 100%;
            word-wrap: break-word;
        `;

        const header = document.createElement('div');
        header.style.cssText = `
            display: flex;
            align-items: center;
            font-weight: bold;
            margin-bottom: 4px;
            font-size: 14px;
        `;

        const icon = document.createElement('span');
        icon.textContent = color.icon;
        icon.style.cssText = `
            margin-right: 8px;
            font-size: 16px;
        `;

        const titleSpan = document.createElement('span');
        titleSpan.textContent = title;

        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            margin-left: auto;
            padding: 0;
            width: 20px;
            height: 20px;
        `;

        header.appendChild(icon);
        header.appendChild(titleSpan);
        header.appendChild(closeBtn);

        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            font-size: 13px;
            line-height: 1.4;
            opacity: 0.9;
            margin-bottom: 8px;
        `;
        messageDiv.textContent = message;

        toast.appendChild(header);
        toast.appendChild(messageDiv);

        // Add action buttons
        if (actions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.style.cssText = `
                display: flex;
                gap: 8px;
                margin-top: 12px;
            `;

            actions.forEach(action => {
                const btn = document.createElement('button');
                btn.textContent = action.label;
                btn.style.cssText = `
                    background: rgba(255,255,255,0.2);
                    border: 1px solid rgba(255,255,255,0.3);
                    color: white;
                    padding: 4px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                `;
                btn.onclick = (e) => {
                    e.stopPropagation();
                    action.callback();
                    this.removeToast(toast);
                };
                actionsDiv.appendChild(btn);
            });

            toast.appendChild(actionsDiv);
        }

        // Close functionality
        const removeToast = () => this.removeToast(toast);
        closeBtn.onclick = (e) => {
            e.stopPropagation();
            removeToast();
        };
        toast.onclick = removeToast;

        // Add to container
        this.toastContainer.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(0)';
        });

        // Auto-remove
        if (duration > 0) {
            setTimeout(removeToast, duration);
        }

        return toast;
    }

    /**
     * Remove toast notification
     */
    removeToast(toast) {
        toast.style.transform = 'translateX(100%)';
        toast.style.opacity = '0';
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    /**
     * Show different types of notifications
     */
    showSuccess(title, message, duration = 3000) {
        return this.showToast(title, message, 'success', duration);
    }

    showError(title, message, type = 'error', duration = 8000, actions = []) {
        this.logError({ title, message, type, timestamp: new Date() });
        return this.showToast(title, message, type, duration, actions);
    }

    showWarning(title, message, duration = 6000) {
        return this.showToast(title, message, 'warning', duration);
    }

    showInfo(title, message, duration = 4000) {
        return this.showToast(title, message, 'info', duration);
    }

    showValidationError(title, message, actions = []) {
        return this.showToast(title, message, 'validation', 0, actions);
    }

    /**
     * Log errors for debugging
     */
    logError(error) {
        this.errorHistory.push(error);
        
        // Keep only last 100 errors
        if (this.errorHistory.length > 100) {
            this.errorHistory = this.errorHistory.slice(-100);
        }
        
        console.error(`[${error.type.toUpperCase()}] ${error.title}:`, error.message);
    }

    /**
     * Handle backend execution errors
     */
    handleBackendError(error, nodeId = null) {
        let title = 'Backend Error';
        let message = error.message || 'Unknown error occurred';
        let actions = [];

        // Parse different error types
        if (error.type) {
            switch (error.type) {
                case 'ModelLoadError':
                    title = 'Model Loading Failed';
                    message = `Failed to load model: ${error.message}`;
                    actions = [{
                        label: 'Check Models',
                        callback: () => window.modelBrowser?.show()
                    }];
                    break;
                    
                case 'ValidationError':
                    title = 'Validation Error';
                    message = error.message;
                    break;
                    
                case 'OutOfMemoryError':
                    title = 'Out of Memory';
                    message = 'Insufficient GPU memory. Try reducing batch size or image resolution.';
                    actions = [{
                        label: 'Memory Tips',
                        callback: () => this.showMemoryOptimizationTips()
                    }];
                    break;
                    
                case 'TimeoutError':
                    title = 'Generation Timeout';
                    message = 'Generation took too long and was cancelled.';
                    break;
                    
                case 'ConnectionError':
                    title = 'Connection Error';
                    message = 'Lost connection to backend server.';
                    actions = [{
                        label: 'Retry',
                        callback: () => this.testBackendConnection()
                    }];
                    break;
            }
        }

        // Highlight the failed node
        if (nodeId) {
            this.highlightNodeError(nodeId, error);
        }

        this.showError(title, message, 'error', 8000, actions);
    }

    /**
     * Show memory optimization tips
     */
    showMemoryOptimizationTips() {
        const tips = [
            "• Reduce batch size to 1",
            "• Lower image resolution (512x512 instead of 1024x1024)",
            "• Use fp16 precision instead of fp32",
            "• Enable model offloading",
            "• Close other GPU applications",
            "• Use smaller tile sizes for upscaling"
        ];

        this.showInfo(
            'Memory Optimization Tips',
            tips.join('\n'),
            10000
        );
    }

    /**
     * Test backend connection
     */
    async testBackendConnection() {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                this.showSuccess('Connection Restored', 'Backend is responding normally');
            }
        } catch (error) {
            this.showError('Connection Failed', 'Backend is still unreachable');
        }
    }

    /**
     * Highlight node with error
     */
    highlightNodeError(nodeId, error) {
        const node = this.findNodeById(nodeId);
        if (!node) return;

        // Store error info
        this.nodeErrors.set(nodeId, {
            error: error,
            timestamp: new Date(),
            highlighted: true
        });

        // Apply error styling
        node.bgcolor = "#660000";
        node.color = "#ff4444";
        
        // Add error badge
        node.errorBadge = {
            text: "ERROR",
            color: "#ff4444"
        };

        // Trigger redraw
        if (node.graph) {
            node.graph.setDirtyCanvas(true);
        }

        // Auto-clear after 30 seconds
        setTimeout(() => {
            this.clearNodeError(nodeId);
        }, 30000);
    }

    /**
     * Clear node error highlighting
     */
    clearNodeError(nodeId) {
        const node = this.findNodeById(nodeId);
        if (!node) return;

        this.nodeErrors.delete(nodeId);
        
        // Restore original colors
        node.bgcolor = node.constructor.prototype.bgcolor || "#353535";
        node.color = node.constructor.prototype.color || "#999";
        delete node.errorBadge;

        if (node.graph) {
            node.graph.setDirtyCanvas(true);
        }
    }

    /**
     * Find node by ID in graph
     */
    findNodeById(nodeId) {
        // Try the main app graph first
        if (window.sdFormsApp && window.sdFormsApp.graph && window.sdFormsApp.graph._nodes) {
            const node = window.sdFormsApp.graph._nodes.find(node => node.id === nodeId || node.componentId === nodeId);
            if (node) return node;
        }
        
        // Fallback to global graph reference
        if (window.graph && window.graph._nodes) {
            return window.graph._nodes.find(node => node.id === nodeId || node.componentId === nodeId);
        }
        
        return null;
    }

    /**
     * Validate workflow before execution
     */
    validateWorkflow(graph) {
        this.validationErrors.clear();
        const errors = [];

        if (!graph || !graph._nodes) {
            errors.push({
                type: 'workflow',
                message: 'No workflow to validate'
            });
            return errors;
        }

        // Check for nodes without required inputs
        graph._nodes.forEach(node => {
            if (!node.inputs) return;

            node.inputs.forEach((input, index) => {
                if (!input.optional && !input.link) {
                    errors.push({
                        type: 'missing_input',
                        nodeId: node.id,
                        nodeTitle: node.title,
                        inputName: input.name,
                        message: `Node "${node.title}" missing required input: ${input.name}`
                    });
                }
            });
        });

        // Check for isolated nodes (no outputs connected)
        graph._nodes.forEach(node => {
            if (!node.outputs) return;

            const hasConnectedOutputs = node.outputs.some(output => 
                output.links && output.links.length > 0
            );

            if (!hasConnectedOutputs && node.type !== 'sd_forms/output') {
                errors.push({
                    type: 'isolated_node',
                    nodeId: node.id,
                    nodeTitle: node.title,
                    message: `Node "${node.title}" has no connected outputs`
                });
            }
        });

        // Check for circular dependencies
        const circularDeps = this.detectCircularDependencies(graph);
        if (circularDeps.length > 0) {
            errors.push({
                type: 'circular_dependency',
                message: `Circular dependency detected: ${circularDeps.join(' -> ')}`
            });
        }

        // Store validation errors
        errors.forEach(error => {
            if (error.nodeId) {
                this.validationErrors.set(error.nodeId, error);
            }
        });

        return errors;
    }

    /**
     * Detect circular dependencies in graph
     */
    detectCircularDependencies(graph) {
        const visited = new Set();
        const recursionStack = new Set();
        const path = [];

        function dfs(nodeId) {
            if (recursionStack.has(nodeId)) {
                return path.slice(path.indexOf(nodeId));
            }

            if (visited.has(nodeId)) {
                return null;
            }

            visited.add(nodeId);
            recursionStack.add(nodeId);
            path.push(nodeId);

            const node = graph._nodes.find(n => n.id === nodeId);
            if (node && node.outputs) {
                for (const output of node.outputs) {
                    if (output.links) {
                        for (const linkId of output.links) {
                            const link = graph.links[linkId];
                            if (link) {
                                const cycle = dfs(link.target_id);
                                if (cycle) return cycle;
                            }
                        }
                    }
                }
            }

            recursionStack.delete(nodeId);
            path.pop();
            return null;
        }

        for (const node of graph._nodes) {
            if (!visited.has(node.id)) {
                const cycle = dfs(node.id);
                if (cycle) return cycle;
            }
        }

        return [];
    }

    /**
     * Validate connection between ports
     */
    validateConnection(sourceNode, sourceSlot, targetNode, targetSlot) {
        const sourceOutput = sourceNode.outputs[sourceSlot];
        const targetInput = targetNode.inputs[targetSlot];

        if (!sourceOutput || !targetInput) {
            return {
                valid: false,
                error: 'Invalid port connection'
            };
        }

        // Type compatibility check
        const compatible = this.areTypesCompatible(sourceOutput.type, targetInput.type);
        if (!compatible) {
            return {
                valid: false,
                error: `Type mismatch: ${sourceOutput.type} → ${targetInput.type}`
            };
        }

        // Check for self-connection
        if (sourceNode.id === targetNode.id) {
            return {
                valid: false,
                error: 'Cannot connect node to itself'
            };
        }

        // Check for duplicate connections
        if (targetInput.link !== null) {
            return {
                valid: false,
                error: 'Input already connected'
            };
        }

        return { valid: true };
    }

    /**
     * Check type compatibility
     */
    areTypesCompatible(sourceType, targetType) {
        // Exact match
        if (sourceType === targetType) return true;

        // Any type compatibility
        if (sourceType === 'any' || targetType === 'any') return true;
        if (sourceType === '*' || targetType === '*') return true;

        // Numeric compatibility
        const numericTypes = ['number', 'int', 'float'];
        if (numericTypes.includes(sourceType) && numericTypes.includes(targetType)) {
            return true;
        }

        // String compatibility
        const stringTypes = ['string', 'text'];
        if (stringTypes.includes(sourceType) && stringTypes.includes(targetType)) {
            return true;
        }

        // Image format compatibility
        const imageTypes = ['image', 'tensor', 'latent'];
        if (imageTypes.includes(sourceType) && imageTypes.includes(targetType)) {
            return true;
        }

        return false;
    }

    /**
     * Show validation errors summary
     */
    showValidationErrors(errors) {
        if (errors.length === 0) {
            this.showSuccess('Validation Passed', 'Workflow is ready for execution');
            return;
        }

        const errorSummary = errors.map(e => `• ${e.message}`).join('\n');
        
        const actions = [{
            label: 'Fix Issues',
            callback: () => this.highlightValidationErrors(errors)
        }];

        this.showValidationError(
            `Validation Failed (${errors.length} issues)`,
            errorSummary,
            actions
        );
    }

    /**
     * Highlight nodes with validation errors
     */
    highlightValidationErrors(errors) {
        errors.forEach(error => {
            if (error.nodeId) {
                this.highlightNodeValidationError(error.nodeId, error);
            }
        });
    }

    /**
     * Highlight single node validation error
     */
    highlightNodeValidationError(nodeId, error) {
        const node = this.findNodeById(nodeId);
        if (!node) return;

        // Apply validation error styling
        node.bgcolor = "#664400";
        node.color = "#ffaa00";
        
        node.validationBadge = {
            text: "VALIDATION",
            color: "#ffaa00"
        };

        if (node.graph) {
            node.graph.setDirtyCanvas(true);
        }
    }

    /**
     * Clear all validation errors
     */
    clearValidationErrors() {
        this.validationErrors.forEach((error, nodeId) => {
            const node = this.findNodeById(nodeId);
            if (node) {
                node.bgcolor = node.constructor.prototype.bgcolor || "#353535";
                node.color = node.constructor.prototype.color || "#999";
                delete node.validationBadge;
                
                if (node.graph) {
                    node.graph.setDirtyCanvas(true);
                }
            }
        });
        
        this.validationErrors.clear();
    }

    /**
     * Clear all node errors (both validation and execution errors)
     */
    clearAllNodeErrors() {
        // Clear validation errors
        this.clearValidationErrors();
        
        // Clear execution errors
        this.nodeErrors.forEach((error, nodeId) => {
            this.clearNodeError(nodeId);
        });
        
        // Clear connection errors
        this.connectionErrors.clear();
    }

    /**
     * Get error history for debugging
     */
    getErrorHistory() {
        return [...this.errorHistory];
    }

    /**
     * Clear error history
     */
    clearErrorHistory() {
        this.errorHistory = [];
    }

    /**
     * Export error log for debugging
     */
    exportErrorLog() {
        const log = {
            timestamp: new Date().toISOString(),
            errors: this.errorHistory,
            nodeErrors: Object.fromEntries(this.nodeErrors),
            validationErrors: Object.fromEntries(this.validationErrors)
        };

        const blob = new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `sd_forms_error_log_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }
}

// Create global instance
window.errorSystem = new ErrorSystem();

// Extend LiteGraph nodes with error handling
if (typeof LGraphNode !== 'undefined') {
    const originalOnDrawForeground = LGraphNode.prototype.onDrawForeground;
    
    LGraphNode.prototype.onDrawForeground = function(ctx) {
        // Call original method
        if (originalOnDrawForeground) {
            originalOnDrawForeground.call(this, ctx);
        }

        // Draw error badges
        if (this.errorBadge) {
            ctx.save();
            ctx.fillStyle = this.errorBadge.color;
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.errorBadge.text, this.size[0] / 2, -5);
            ctx.restore();
        }

        if (this.validationBadge) {
            ctx.save();
            ctx.fillStyle = this.validationBadge.color;
            ctx.font = "8px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.validationBadge.text, this.size[0] / 2, this.size[1] + 15);
            ctx.restore();
        }
    };

    // Override connection validation
    const originalOnConnectInput = LGraphNode.prototype.onConnectInput;
    
    LGraphNode.prototype.onConnectInput = function(targetSlot, sourceNode, sourceSlot) {
        const validation = window.errorSystem.validateConnection(
            sourceNode, sourceSlot, this, targetSlot
        );

        if (!validation.valid) {
            window.errorSystem.showWarning('Connection Invalid', validation.error);
            return false;
        }

        // Call original method
        if (originalOnConnectInput) {
            return originalOnConnectInput.call(this, targetSlot, sourceNode, sourceSlot);
        }

        return true;
    };
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ErrorSystem;
}