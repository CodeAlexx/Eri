/**
 * API Service for SD Forms - JavaScript version
 * Converted from Flutter ApiService to work with LiteGraph
 * Maintains exact API structure expected by FastAPI backend
 */

class ApiService {
    constructor() {
        this.baseUrl = 'http://localhost:8001';
        this.wsUrl = 'ws://localhost:8001/ws';
        this.websocket = null;
        this.wsCallbacks = new Set();
        this.eventListeners = new Map(); // Event type -> Set of callbacks
        this.isConnected = false;
        this.pingInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000; // 3 seconds
        this.currentExecution = null;
        this.componentStatuses = new Map(); // componentId -> status
        this.previewPanel = null;
        
        // Bind methods to maintain context
        this.onWebSocketMessage = this.onWebSocketMessage.bind(this);
        this.onWebSocketOpen = this.onWebSocketOpen.bind(this);
        this.onWebSocketClose = this.onWebSocketClose.bind(this);
        this.onWebSocketError = this.onWebSocketError.bind(this);
        
        // Initialize event listeners map
        this.initializeEventListeners();
        
        // Create preview panel
        this.createPreviewPanel();
        
        // Auto-connect WebSocket
        this.connectWebSocket().catch(console.error);
    }
    
    /**
     * Initialize event listeners map with default event types
     */
    initializeEventListeners() {
        const eventTypes = ['connection', 'progress', 'preview', 'result', 'component_status', 'error', 'execution_start', 'execution_complete'];
        eventTypes.forEach(type => {
            this.eventListeners.set(type, new Set());
        });
    }

    /**
     * Create floating preview panel for real-time image updates
     */
    createPreviewPanel() {
        // Check if panel already exists
        if (document.getElementById('floating-preview-panel')) {
            this.previewPanel = document.getElementById('floating-preview-panel');
            return;
        }
        
        const panel = document.createElement('div');
        panel.id = 'floating-preview-panel';
        panel.className = 'floating-preview-panel hidden';
        panel.innerHTML = `
            <div class="preview-header">
                <h3>Live Preview</h3>
                <div class="preview-controls">
                    <button id="preview-pin" class="btn btn-small" title="Pin/Unpin panel">📌</button>
                    <button id="preview-close" class="btn btn-small" title="Close preview">✖️</button>
                </div>
            </div>
            <div class="preview-content">
                <div id="preview-image-container" class="preview-image-container">
                    <div class="preview-placeholder">
                        <div class="placeholder-icon">🖼️</div>
                        <div class="placeholder-text">Live preview will appear here</div>
                    </div>
                </div>
                <div class="preview-info">
                    <div class="preview-step">Step: <span id="preview-step">-</span></div>
                    <div class="preview-progress">Progress: <span id="preview-progress">0%</span></div>
                </div>
            </div>
        `;
        
        // Add styles if not already present
        if (!document.getElementById('preview-panel-styles')) {
            const styles = document.createElement('style');
            styles.id = 'preview-panel-styles';
            styles.textContent = `
                .floating-preview-panel {
                    position: fixed;
                    top: 100px;
                    right: 20px;
                    width: 320px;
                    max-height: 500px;
                    background: #2a2a2a;
                    border: 1px solid #444;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    z-index: 1000;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    color: #fff;
                    display: flex;
                    flex-direction: column;
                    resize: both;
                    overflow: hidden;
                }
                .floating-preview-panel.hidden { display: none; }
                .floating-preview-panel.pinned { opacity: 0.9; }
                .preview-header {
                    padding: 12px 16px;
                    border-bottom: 1px solid #444;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: move;
                    background: linear-gradient(135deg, #3a3a3a, #2a2a2a);
                }
                .preview-header h3 {
                    margin: 0;
                    font-size: 14px;
                    font-weight: 600;
                }
                .preview-controls {
                    display: flex;
                    gap: 4px;
                }
                .preview-controls .btn {
                    padding: 4px 8px;
                    font-size: 12px;
                    background: #444;
                    border: 1px solid #555;
                    color: #fff;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .preview-controls .btn:hover {
                    background: #555;
                }
                .preview-content {
                    flex: 1;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                }
                .preview-image-container {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 200px;
                    background: #1a1a1a;
                    position: relative;
                }
                .preview-placeholder {
                    text-align: center;
                    color: #666;
                }
                .placeholder-icon {
                    font-size: 48px;
                    margin-bottom: 8px;
                }
                .placeholder-text {
                    font-size: 12px;
                }
                .preview-image {
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                    border-radius: 4px;
                }
                .preview-info {
                    padding: 12px 16px;
                    border-top: 1px solid #444;
                    background: #1a1a1a;
                    display: flex;
                    justify-content: space-between;
                    font-size: 12px;
                }
                .component-status-indicator {
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    border: 2px solid #fff;
                    box-shadow: 0 0 4px rgba(0,0,0,0.5);
                }
                .status-idle { background-color: #6c757d; }
                .status-processing { background-color: #ffc107; animation: pulse 1s infinite; }
                .status-complete { background-color: #28a745; }
                .status-error { background-color: #dc3545; }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(panel);
        this.previewPanel = panel;
        
        // Make panel draggable
        this.makePreviewPanelDraggable();
        
        // Setup event listeners
        this.setupPreviewPanelEvents();
    }
    
    /**
     * Make preview panel draggable
     */
    makePreviewPanelDraggable() {
        const header = this.previewPanel.querySelector('.preview-header');
        let isDragging = false;
        let currentX;
        let currentY;
        let initialX;
        let initialY;
        let xOffset = 0;
        let yOffset = 0;
        
        header.addEventListener('mousedown', (e) => {
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;
            
            if (e.target === header || header.contains(e.target)) {
                isDragging = true;
            }
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                e.preventDefault();
                currentX = e.clientX - initialX;
                currentY = e.clientY - initialY;
                
                xOffset = currentX;
                yOffset = currentY;
                
                this.previewPanel.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;
            }
        });
        
        document.addEventListener('mouseup', () => {
            initialX = currentX;
            initialY = currentY;
            isDragging = false;
        });
    }
    
    /**
     * Setup preview panel event listeners
     */
    setupPreviewPanelEvents() {
        const closeBtn = this.previewPanel.querySelector('#preview-close');
        const pinBtn = this.previewPanel.querySelector('#preview-pin');
        
        closeBtn.addEventListener('click', () => {
            this.hidePreviewPanel();
        });
        
        pinBtn.addEventListener('click', () => {
            this.previewPanel.classList.toggle('pinned');
            pinBtn.textContent = this.previewPanel.classList.contains('pinned') ? '📍' : '📌';
        });
    }
    
    /**
     * Show preview panel
     */
    showPreviewPanel() {
        if (this.previewPanel) {
            this.previewPanel.classList.remove('hidden');
        }
    }
    
    /**
     * Hide preview panel
     */
    hidePreviewPanel() {
        if (this.previewPanel) {
            this.previewPanel.classList.add('hidden');
        }
    }
    
    /**
     * Update preview image in floating panel
     */
    updatePreviewImage(imageBase64, step, progress) {
        if (!this.previewPanel) return;
        
        const container = this.previewPanel.querySelector('#preview-image-container');
        const stepSpan = this.previewPanel.querySelector('#preview-step');
        const progressSpan = this.previewPanel.querySelector('#preview-progress');
        
        // Update image
        let img = container.querySelector('.preview-image');
        if (!img) {
            img = document.createElement('img');
            img.className = 'preview-image';
            container.innerHTML = '';
            container.appendChild(img);
        }
        
        img.src = `data:image/png;base64,${imageBase64}`;
        img.alt = `Preview Step ${step}`;
        
        // Update info
        if (stepSpan) stepSpan.textContent = step || '-';
        if (progressSpan) progressSpan.textContent = `${Math.round(progress || 0)}%`;
        
        // Show panel if hidden
        this.showPreviewPanel();
    }

    /**
     * REST API Methods
     */

    /**
     * Health check - matches Flutter healthCheck()
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                },
                timeout: 5000 // 5 second timeout
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Backend health check passed:', data);
                this.emit('connection', { status: 'healthy', data });
                return true;
            } else {
                console.warn('⚠️ Backend health check failed:', response.status);
                this.emit('connection', { status: 'unhealthy', status_code: response.status });
                return false;
            }
        } catch (error) {
            console.error('❌ Backend health check error:', error);
            this.emit('connection', { status: 'error', error: error.message });
            return false;
        }
    }
    
    /**
     * Periodic health check
     */
    startHealthCheck(intervalMs = 30000) {
        // Check immediately
        this.healthCheck();
        
        // Then check periodically
        setInterval(() => {
            this.healthCheck();
        }, intervalMs);
    }

    /**
     * Get available components - matches Flutter getComponents()
     * Returns array of ComponentInfo objects
     */
    async getComponents() {
        try {
            const response = await fetch(`${this.baseUrl}/api/components`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to load components: ${response.status}`);
            }

            const jsonList = await response.json();
            return jsonList.map(json => ComponentInfo.fromJson(json));
        } catch (error) {
            throw new Error(`Error fetching components: ${error.message}`);
        }
    }

    /**
     * Get available models - matches Flutter getModels()
     */
    async getModels() {
        try {
            const response = await fetch(`${this.baseUrl}/api/models`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to load models: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            throw new Error(`Error fetching models: ${error.message}`);
        }
    }

    /**
     * Execute workflow - matches Flutter executeWorkflow()
     * @param {ExecutionRequest} request - ExecutionRequest object with workflow and settings
     * @returns {Promise<ExecutionResult>} ExecutionResult object
     */
    async executeWorkflow(request) {
        try {
            const response = await fetch(`${this.baseUrl}/api/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request.toJson())
            });

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({}));
                throw new Error(`Execution failed: ${errorBody.detail || 'Unknown error'}`);
            }

            const result = await response.json();
            return ExecutionResult.fromJson(result);
        } catch (error) {
            throw new Error(`Error executing workflow: ${error.message}`);
        }
    }

    // Validate workflow
    async validateWorkflow(workflow) {
        try {
            const response = await fetch(`${this.baseUrl}/api/validate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    components: workflow.components,
                    connections: workflow.connections
                })
            });

            if (!response.ok) {
                throw new Error(`Validation failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error validating workflow:', error);
            throw error;
        }
    }

    // Cancel execution
    async cancelExecution() {
        try {
            const response = await fetch(`${this.baseUrl}/api/cancel`, {
                method: 'POST',
            });

            if (!response.ok) {
                throw new Error(`Cancel failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error canceling execution:', error);
            throw error;
        }
    }

    // Save workflow
    async saveWorkflow(workflow, name = null) {
        try {
            const response = await fetch(`${this.baseUrl}/api/workflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: name || `Workflow_${Date.now()}`,
                    workflow: workflow
                })
            });

            if (!response.ok) {
                throw new Error(`Save failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error saving workflow:', error);
            throw error;
        }
    }

    // Load workflow
    async loadWorkflow(workflowId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/workflows/${workflowId}`);
            
            if (!response.ok) {
                throw new Error(`Load failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error loading workflow:', error);
            throw error;
        }
    }

    /**
     * WebSocket Methods - matches Flutter API structure
     */

    /**
     * Connect to WebSocket for real-time updates - matches Flutter connectWebSocket()
     */
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(this.wsUrl);
                
                this.websocket.onopen = (event) => {
                    this.onWebSocketOpen(event);
                    resolve();
                };
                
                this.websocket.onmessage = this.onWebSocketMessage;
                this.websocket.onclose = this.onWebSocketClose;
                this.websocket.onerror = (event) => {
                    this.onWebSocketError(event);
                    reject(new Error('WebSocket connection failed'));
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Disconnect WebSocket - matches Flutter disconnectWebSocket()
     */
    disconnectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
        this.wsCallbacks.clear();
        
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    /**
     * Send ping to keep WebSocket alive - matches Flutter sendPing()
     */
    sendPing() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'ping' }));
        }
    }

    onWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('📨 WebSocket message received:', data);
            
            // Handle different message types based on backend/server.py ConnectionManager
            this.handleWebSocketMessage(data);
            
            // Also call legacy callbacks for backward compatibility
            this.wsCallbacks.forEach(callback => callback(data));
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    /**
     * Handle WebSocket messages based on type - matches backend/server.py message formats
     * @param {Object} data - Message data from backend
     */
    handleWebSocketMessage(data) {
        const { type } = data;
        
        switch (type) {
            case 'progress':
                // Format: {"type": "progress", "message": "string"}
                this.handleProgressMessage(data);
                break;
                
            case 'preview':
                // Format: {"type": "preview", "image": "base64", "progress": int, "step": int}
                this.handlePreviewMessage(data);
                break;
                
            case 'result':
                // Format: {"type": "result", "images": ["base64", ...]}
                this.handleResultMessage(data);
                break;
                
            case 'component_status':
                // Custom format: {"type": "component_status", "component_id": "string", "status": "idle|processing|complete|error"}
                this.handleComponentStatusMessage(data);
                break;
                
            case 'execution_start':
                // Custom format: {"type": "execution_start", "execution_id": "string"}
                this.handleExecutionStart(data);
                break;
                
            case 'execution_complete':
                // Custom format: {"type": "execution_complete", "execution_id": "string", "success": boolean}
                this.handleExecutionComplete(data);
                break;
                
            case 'pong':
                // Heartbeat response
                console.log('🏓 Pong received');
                break;
                
            case 'error':
                // Format: {"type": "error", "message": "string", "details": "string"}
                this.handleErrorMessage(data);
                break;
                
            default:
                console.warn('Unknown WebSocket message type:', type);
        }
        
        // Emit event to registered listeners
        this.emit(type, data);
    }
    
    /**
     * Handle progress update messages
     */
    handleProgressMessage(data) {
        console.log('📈 Progress update:', data.message);
        // Update any global progress indicators
        this.emit('progress', { message: data.message, percentage: data.percentage || 0 });
    }
    
    /**
     * Handle preview image messages
     */
    handlePreviewMessage(data) {
        console.log('🖼️ Preview image received - Step:', data.step, 'Progress:', data.progress);
        
        // Update floating preview panel
        this.updatePreviewImage(data.image, data.step, data.progress);
        
        // Emit for other components to handle
        this.emit('preview', data);
    }
    
    /**
     * Handle final result messages
     */
    handleResultMessage(data) {
        console.log('🎉 Execution results received:', data.images.length, 'images');
        
        // Hide preview panel when final results arrive
        this.hidePreviewPanel();
        
        // Reset component statuses
        this.resetComponentStatuses();
        
        // Emit results
        this.emit('result', data);
    }
    
    /**
     * Handle component status updates
     */
    handleComponentStatusMessage(data) {
        const { component_id, status } = data;
        console.log(`🔧 Component ${component_id} status:`, status);
        
        // Update internal status tracking
        this.componentStatuses.set(component_id, status);
        
        // Update visual indicators for the component
        this.updateComponentVisualStatus(component_id, status);
        
        // Emit for other components to handle
        this.emit('component_status', data);
    }
    
    /**
     * Handle execution start
     */
    handleExecutionStart(data) {
        console.log('🚀 Execution started:', data.execution_id);
        this.currentExecution = data.execution_id;
        
        // Reset component statuses
        this.resetComponentStatuses();
        
        // Show preview panel
        this.showPreviewPanel();
        
        this.emit('execution_start', data);
    }
    
    /**
     * Handle execution complete
     */
    handleExecutionComplete(data) {
        console.log('✅ Execution completed:', data.execution_id, 'Success:', data.success);
        this.currentExecution = null;
        
        this.emit('execution_complete', data);
    }
    
    /**
     * Handle error messages
     */
    handleErrorMessage(data) {
        console.error('❌ WebSocket error:', data.message, data.details);
        this.emit('error', data);
    }
    
    /**
     * Update visual status indicators for components
     */
    updateComponentVisualStatus(componentId, status) {
        // Find all visual representations of this component
        const indicators = document.querySelectorAll(`[data-component-id="${componentId}"] .component-status-indicator`);
        
        indicators.forEach(indicator => {
            // Remove all status classes
            indicator.classList.remove('status-idle', 'status-processing', 'status-complete', 'status-error');
            
            // Add new status class
            indicator.classList.add(`status-${status}`);
        });
        
        // Also update LiteGraph nodes if they exist
        this.updateLiteGraphNodeStatus(componentId, status);
    }
    
    /**
     * Update LiteGraph node appearance based on status
     */
    updateLiteGraphNodeStatus(componentId, status) {
        // This will be called by the main app to update node colors
        // We emit an event that the main app can listen to
        this.emit('node_status_update', { componentId, status });
    }
    
    /**
     * Add status indicator to a component element
     */
    addStatusIndicatorToComponent(componentElement, componentId) {
        if (!componentElement.querySelector('.component-status-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'component-status-indicator status-idle';
            componentElement.style.position = 'relative';
            componentElement.appendChild(indicator);
            
            // Store component ID for future status updates
            componentElement.setAttribute('data-component-id', componentId);
        }
    }
    
    /**
     * Reset all component statuses to idle
     */
    resetComponentStatuses() {
        this.componentStatuses.clear();
        
        // Reset all visual indicators
        const indicators = document.querySelectorAll('.component-status-indicator');
        indicators.forEach(indicator => {
            indicator.classList.remove('status-processing', 'status-complete', 'status-error');
            indicator.classList.add('status-idle');
        });
    }
    
    /**
     * Get current status of a component
     */
    getComponentStatus(componentId) {
        return this.componentStatuses.get(componentId) || 'idle';
    }
    
    /**
     * Event emitter methods for real-time updates
     */
    on(eventType, callback) {
        if (!this.eventListeners.has(eventType)) {
            this.eventListeners.set(eventType, new Set());
        }
        this.eventListeners.get(eventType).add(callback);
    }
    
    off(eventType, callback) {
        if (this.eventListeners.has(eventType)) {
            this.eventListeners.get(eventType).delete(callback);
        }
    }
    
    emit(eventType, data) {
        if (this.eventListeners.has(eventType)) {
            this.eventListeners.get(eventType).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${eventType}:`, error);
                }
            });
        }
    }

    onWebSocketOpen(event) {
        this.isConnected = true;
        this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        console.log('🔗 WebSocket connected successfully');
        
        // Emit connection event
        this.emit('connection', { status: 'connected' });
        
        // Send ping to keep connection alive
        this.sendPing();
        
        // Set up ping interval
        this.pingInterval = setInterval(() => {
            this.sendPing();
        }, 30000); // Every 30 seconds
    }

    onWebSocketClose(event) {
        this.isConnected = false;
        console.log('🔌 WebSocket disconnected - Code:', event.code, 'Reason:', event.reason);
        
        // Emit connection event
        this.emit('connection', { status: 'disconnected', code: event.code, reason: event.reason });
        
        // Clear ping interval
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
        
        // Attempt to reconnect with exponential backoff
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
            console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                if (!this.isConnected) {
                    this.reconnectAttempts++;
                    this.connectWebSocket().catch(error => {
                        console.error('❌ Reconnection failed:', error);
                    });
                }
            }, delay);
        } else {
            console.error('❌ Max reconnection attempts reached. Please refresh the page.');
            this.emit('connection', { status: 'failed', message: 'Max reconnection attempts reached' });
        }
    }

    onWebSocketError(event) {
        console.error('❌ WebSocket error:', event);
        this.emit('connection', { status: 'error', error: event });
    }

    addWebSocketCallback(callback) {
        this.wsCallbacks.add(callback);
    }

    removeWebSocketCallback(callback) {
        this.wsCallbacks.delete(callback);
    }

    /**
     * Utility Methods - matches Flutter API
     */

    /**
     * Convert base64 string to Uint8Array - matches Flutter base64ToBytes()
     */
    base64ToBytes(base64String) {
        const binaryString = atob(base64String);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes;
    }

    /**
     * Convert Uint8Array to base64 string - matches Flutter bytesToBase64()
     */
    bytesToBase64(bytes) {
        const binaryString = Array.from(bytes, byte => String.fromCharCode(byte)).join('');
        return btoa(binaryString);
    }
    
    /**
     * Download image from base64 data
     * @param {string} base64Data - Base64 encoded image data
     * @param {string} filename - Filename for download
     */
    downloadImage(base64Data, filename = 'generated_image.png') {
        try {
            // Create download link
            const link = document.createElement('a');
            link.href = `data:image/png;base64,${base64Data}`;
            link.download = filename;
            
            // Trigger download
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log('💾 Image downloaded:', filename);
        } catch (error) {
            console.error('❌ Failed to download image:', error);
        }
    }
    
    /**
     * Manual WebSocket reconnection
     */
    forceReconnect() {
        console.log('🔄 Forcing WebSocket reconnection...');
        this.disconnectWebSocket();
        this.reconnectAttempts = 0; // Reset attempts
        setTimeout(() => {
            this.connectWebSocket().catch(console.error);
        }, 1000);
    }
    
    /**
     * Get WebSocket connection status
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.maxReconnectAttempts,
            currentExecution: this.currentExecution,
            componentStatusCount: this.componentStatuses.size
        };
    }
    
    /**
     * Debug method - get all component statuses
     */
    getComponentStatuses() {
        return Object.fromEntries(this.componentStatuses);
    }
}

/**
 * Workflow and Component Classes
 * These match the Dart model classes for API compatibility
 */

class Port {
    constructor(name, type, direction, optional = false) {
        this.name = name;
        this.type = type;
        this.direction = direction;
        this.optional = optional;
    }

    get isInput() {
        return this.direction === 'INPUT';
    }

    get isOutput() {
        return this.direction === 'OUTPUT';
    }

    static fromJson(json) {
        return new Port(
            json.name,
            json.type,
            json.direction,
            json.optional || false
        );
    }

    toJson() {
        return {
            name: this.name,
            type: this.type,
            direction: this.direction,
            optional: this.optional
        };
    }
}

class PropertyDefinition {
    constructor(name, displayName, type, defaultValue = null, category = '', metadata = {}) {
        this.name = name;
        this.display_name = displayName;
        this.type = type;
        this.default_value = defaultValue;
        this.category = category;
        this.metadata = metadata;
    }

    static fromJson(json) {
        return new PropertyDefinition(
            json.name,
            json.display_name,
            json.type,
            json.default_value,
            json.category || '',
            json.metadata || {}
        );
    }

    toJson() {
        return {
            name: this.name,
            display_name: this.display_name,
            type: this.type,
            default_value: this.default_value,
            category: this.category,
            metadata: this.metadata
        };
    }
}

class ComponentInfo {
    constructor(id, type, displayName, category, icon, inputPorts = [], outputPorts = [], propertyDefinitions = []) {
        this.id = id;
        this.type = type;
        this.display_name = displayName;
        this.category = category;
        this.icon = icon;
        this.input_ports = inputPorts;
        this.output_ports = outputPorts;
        this.property_definitions = propertyDefinitions;
    }

    static fromJson(json) {
        return new ComponentInfo(
            json.id,
            json.type,
            json.display_name,
            json.category,
            json.icon,
            (json.input_ports || []).map(port => Port.fromJson(port)),
            (json.output_ports || []).map(port => Port.fromJson(port)),
            (json.property_definitions || []).map(prop => PropertyDefinition.fromJson(prop))
        );
    }

    toJson() {
        return {
            id: this.id,
            type: this.type,
            display_name: this.display_name,
            category: this.category,
            icon: this.icon,
            input_ports: this.input_ports.map(port => port.toJson()),
            output_ports: this.output_ports.map(port => port.toJson()),
            property_definitions: this.property_definitions.map(prop => prop.toJson())
        };
    }
}

class ComponentPosition {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    static fromJson(json) {
        return new ComponentPosition(json.x, json.y);
    }

    toJson() {
        return { x: this.x, y: this.y };
    }

    copyWith(x, y) {
        return new ComponentPosition(
            x !== undefined ? x : this.x,
            y !== undefined ? y : this.y
        );
    }
}

class ComponentSize {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }

    static fromJson(json) {
        return new ComponentSize(json.width, json.height);
    }

    toJson() {
        return { width: this.width, height: this.height };
    }

    copyWith(width, height) {
        return new ComponentSize(
            width !== undefined ? width : this.width,
            height !== undefined ? height : this.height
        );
    }
}

class ComponentInstance {
    constructor(id, type, properties = {}, position = null, size = null) {
        this.id = id;
        this.type = type;
        this.properties = properties;
        this.position = position || new ComponentPosition(0, 0);
        this.size = size || new ComponentSize(120, 80);
    }

    static fromJson(json) {
        return new ComponentInstance(
            json.id,
            json.type,
            json.properties || {},
            json.position ? ComponentPosition.fromJson(json.position) : null,
            json.size ? ComponentSize.fromJson(json.size) : null
        );
    }

    toJson() {
        return {
            id: this.id,
            type: this.type,
            properties: this.properties,
            position: this.position.toJson(),
            size: this.size.toJson()
        };
    }

    copyWith(id, type, properties, position, size) {
        return new ComponentInstance(
            id !== undefined ? id : this.id,
            type !== undefined ? type : this.type,
            properties !== undefined ? properties : { ...this.properties },
            position !== undefined ? position : this.position,
            size !== undefined ? size : this.size
        );
    }
}

class Connection {
    constructor(id, fromComponent, fromPort, toComponent, toPort) {
        this.id = id;
        this.from_component = fromComponent;
        this.from_port = fromPort;
        this.to_component = toComponent;
        this.to_port = toPort;
    }

    static fromJson(json) {
        return new Connection(
            json.id,
            json.from_component,
            json.from_port,
            json.to_component,
            json.to_port
        );
    }

    toJson() {
        return {
            id: this.id,
            from_component: this.from_component,
            from_port: this.from_port,
            to_component: this.to_component,
            to_port: this.to_port
        };
    }
}

class Workflow {
    constructor(components = [], connections = []) {
        this.components = components;
        this.connections = connections;
    }

    static fromJson(json) {
        return new Workflow(
            (json.components || []).map(comp => ComponentInstance.fromJson(comp)),
            (json.connections || []).map(conn => Connection.fromJson(conn))
        );
    }

    toJson() {
        return {
            components: this.components.map(comp => comp.toJson()),
            connections: this.connections.map(conn => conn.toJson())
        };
    }

    copyWith(components, connections) {
        return new Workflow(
            components !== undefined ? components : [...this.components],
            connections !== undefined ? connections : [...this.connections]
        );
    }

    getComponent(id) {
        return this.components.find(c => c.id === id) || null;
    }

    getConnectionsFrom(componentId) {
        return this.connections.filter(conn => conn.from_component === componentId);
    }

    getConnectionsTo(componentId) {
        return this.connections.filter(conn => conn.to_component === componentId);
    }

    canConnect(fromComponent, fromPort, toComponent, toPort) {
        // Check if connection already exists
        const connectionExists = this.connections.some(conn =>
            conn.from_component === fromComponent &&
            conn.from_port === fromPort &&
            conn.to_component === toComponent &&
            conn.to_port === toPort
        );
        
        if (connectionExists) return false;

        // Check if target port is already connected (most ports only accept one connection)
        const targetPortConnected = this.connections.some(conn =>
            conn.to_component === toComponent && conn.to_port === toPort
        );
        
        return !targetPortConnected;
    }
}

class ExecutionRequest {
    constructor(workflow, settings = {}) {
        this.workflow = workflow;
        this.settings = settings;
    }

    static fromJson(json) {
        return new ExecutionRequest(
            Workflow.fromJson(json.workflow),
            json.settings || {}
        );
    }

    toJson() {
        return {
            workflow: this.workflow.toJson(),
            settings: this.settings
        };
    }
}

class ExecutionResult {
    constructor(success, message, images = [], executionTime = '') {
        this.success = success;
        this.message = message;
        this.images = images;
        this.execution_time = executionTime;
    }

    static fromJson(json) {
        return new ExecutionResult(
            json.success,
            json.message,
            json.images || [],
            json.execution_time || ''
        );
    }

    toJson() {
        return {
            success: this.success,
            message: this.message,
            images: this.images,
            execution_time: this.execution_time
        };
    }
}

// Create singleton instance
const apiService = new ApiService();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ApiService,
        apiService,
        Port,
        PropertyDefinition,
        ComponentInfo,
        ComponentPosition,
        ComponentSize,
        ComponentInstance,
        Connection,
        Workflow,
        ExecutionRequest,
        ExecutionResult
    };
} else {
    // Browser globals
    window.ApiService = ApiService;
    window.apiService = apiService;
    window.Port = Port;
    window.PropertyDefinition = PropertyDefinition;
    window.ComponentInfo = ComponentInfo;
    window.ComponentPosition = ComponentPosition;
    window.ComponentSize = ComponentSize;
    window.ComponentInstance = ComponentInstance;
    window.Connection = Connection;
    window.Workflow = Workflow;
    window.ExecutionRequest = ExecutionRequest;
    window.ExecutionResult = ExecutionResult;
}