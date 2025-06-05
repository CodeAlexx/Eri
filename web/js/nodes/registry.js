/**
 * SD Forms Node Registry
 * Automatically fetches components from backend and registers them as LiteGraph nodes
 */

/**
 * Registry for SD Forms components as LiteGraph nodes
 */
class SDFormsNodeRegistry {
    constructor() {
        this.registeredNodes = new Map();
        this.componentInfoCache = new Map();
        this.isInitialized = false;
        this.initPromise = null;
        
        // Node class cache to avoid recreating classes
        this.nodeClassCache = new Map();
        
        // Category organization
        this.categories = new Set();
        
        // Registration callbacks
        this.onNodeRegistered = null;
        this.onRegistrationComplete = null;
    }
    
    /**
     * Initialize the registry by fetching components from backend
     */
    async initialize() {
        if (this.initPromise) {
            return this.initPromise;
        }
        
        this.initPromise = this._doInitialize();
        return this.initPromise;
    }
    
    async _doInitialize() {
        try {
            console.log('🔄 Initializing SD Forms Node Registry...');
            
            // Check if API service is available
            if (typeof apiService === 'undefined') {
                throw new Error('ApiService not available. Make sure api.js is loaded first.');
            }
            
            // Fetch available components from backend
            const components = await this.fetchComponents();
            
            if (!components || components.length === 0) {
                console.warn('⚠️ No components received from backend');
                return;
            }
            
            console.log(`📦 Found ${components.length} components to register`);
            
            // Register each component as a LiteGraph node
            for (const componentInfo of components) {
                await this.registerComponentAsNode(componentInfo);
            }
            
            this.isInitialized = true;
            console.log(`✅ Registry initialized with ${this.registeredNodes.size} nodes in ${this.categories.size} categories`);
            
            // Call completion callback
            if (this.onRegistrationComplete) {
                this.onRegistrationComplete(this.registeredNodes.size, Array.from(this.categories));
            }
            
        } catch (error) {
            console.error('❌ Failed to initialize SD Forms Node Registry:', error);
            throw error;
        }
    }
    
    /**
     * Fetch components from backend API
     */
    async fetchComponents() {
        try {
            // First try to get components from API service
            const components = await apiService.getComponents();
            
            // Validate component structure
            const validComponents = components.filter(comp => this.validateComponentInfo(comp));
            
            if (validComponents.length !== components.length) {
                console.warn(`⚠️ ${components.length - validComponents.length} components failed validation`);
            }
            
            // Cache component info for later use
            for (const comp of validComponents) {
                this.componentInfoCache.set(comp.type, comp);
            }
            
            return validComponents;
            
        } catch (error) {
            console.error('Failed to fetch components from backend:', error);
            
            // Return mock data for development if backend is unavailable
            if (error.message.includes('fetch')) {
                console.log('📝 Backend unavailable, using mock component data');
                return this.getMockComponents();
            }
            
            throw error;
        }
    }
    
    /**
     * Validate component info structure
     */
    validateComponentInfo(componentInfo) {
        if (!componentInfo) return false;
        
        const required = ['type', 'display_name'];
        for (const field of required) {
            if (!componentInfo[field]) {
                console.warn(`Component missing required field '${field}':`, componentInfo);
                return false;
            }
        }
        
        // Ensure arrays exist
        componentInfo.input_ports = componentInfo.input_ports || [];
        componentInfo.output_ports = componentInfo.output_ports || [];
        componentInfo.property_definitions = componentInfo.property_definitions || [];
        
        return true;
    }
    
    /**
     * Register a single component as a LiteGraph node
     */
    async registerComponentAsNode(componentInfo) {
        try {
            const nodeTypeName = this.generateNodeTypeName(componentInfo);
            
            // Check if already registered
            if (this.registeredNodes.has(nodeTypeName)) {
                console.log(`⏭️ Node ${nodeTypeName} already registered, skipping`);
                return;
            }
            
            // Create dynamic node class
            const NodeClass = this.createNodeClass(componentInfo);
            
            // Register with LiteGraph
            LiteGraph.registerNodeType(nodeTypeName, NodeClass);
            
            // Store in our registry
            this.registeredNodes.set(nodeTypeName, {
                nodeClass: NodeClass,
                componentInfo: componentInfo,
                category: componentInfo.category || 'General'
            });
            
            // Track categories
            const category = componentInfo.category || 'General';
            this.categories.add(category);
            
            console.log(`✅ Registered node: ${nodeTypeName} (${category})`);
            
            // Call registration callback
            if (this.onNodeRegistered) {
                this.onNodeRegistered(nodeTypeName, componentInfo);
            }
            
        } catch (error) {
            console.error(`❌ Failed to register component ${componentInfo.type}:`, error);
        }
    }
    
    /**
     * Generate LiteGraph node type name from component info
     */
    generateNodeTypeName(componentInfo) {
        const category = componentInfo.category || 'General';
        const displayName = componentInfo.display_name || componentInfo.type;
        return `SD Forms/${category}/${displayName}`;
    }
    
    /**
     * Create dynamic node class for a component
     */
    createNodeClass(componentInfo) {
        const cacheKey = componentInfo.type;
        
        // Return cached class if available
        if (this.nodeClassCache.has(cacheKey)) {
            return this.nodeClassCache.get(cacheKey);
        }
        
        // Create new dynamic class
        class DynamicSDFormsNode extends SDFormsNode {
            constructor() {
                // Initialize with component info
                super(componentInfo);
                
                // Set additional metadata
                this.desc = this.generateNodeDescription(componentInfo);
            }
            
            // Static properties for LiteGraph
            static title = componentInfo.display_name || componentInfo.type;
            static desc = this.generateNodeDescription(componentInfo);
        }
        
        // Set static properties
        DynamicSDFormsNode.title = componentInfo.display_name || componentInfo.type;
        DynamicSDFormsNode.desc = this.generateNodeDescription(componentInfo);
        
        // Cache the class
        this.nodeClassCache.set(cacheKey, DynamicSDFormsNode);
        
        return DynamicSDFormsNode;
    }
    
    /**
     * Generate node description from component info
     */
    generateNodeDescription(componentInfo) {
        const parts = [];
        
        if (componentInfo.input_ports && componentInfo.input_ports.length > 0) {
            const inputTypes = componentInfo.input_ports.map(p => p.type).join(', ');
            parts.push(`Inputs: ${inputTypes}`);
        }
        
        if (componentInfo.output_ports && componentInfo.output_ports.length > 0) {
            const outputTypes = componentInfo.output_ports.map(p => p.type).join(', ');
            parts.push(`Outputs: ${outputTypes}`);
        }
        
        if (componentInfo.property_definitions && componentInfo.property_definitions.length > 0) {
            parts.push(`${componentInfo.property_definitions.length} properties`);
        }
        
        return parts.join(' | ');
    }
    
    /**
     * Get registered node info
     */
    getRegisteredNode(nodeTypeName) {
        return this.registeredNodes.get(nodeTypeName);
    }
    
    /**
     * Get all registered nodes
     */
    getAllRegisteredNodes() {
        return Array.from(this.registeredNodes.entries()).map(([typeName, info]) => ({
            typeName,
            ...info
        }));
    }
    
    /**
     * Get nodes by category
     */
    getNodesByCategory(category) {
        return Array.from(this.registeredNodes.entries())
            .filter(([typeName, info]) => info.category === category)
            .map(([typeName, info]) => ({ typeName, ...info }));
    }
    
    /**
     * Get all categories
     */
    getCategories() {
        return Array.from(this.categories);
    }
    
    /**
     * Get component info by type
     */
    getComponentInfo(componentType) {
        return this.componentInfoCache.get(componentType);
    }
    
    /**
     * Refresh registry (re-fetch from backend)
     */
    async refresh() {
        console.log('🔄 Refreshing SD Forms Node Registry...');
        
        // Clear caches
        this.registeredNodes.clear();
        this.componentInfoCache.clear();
        this.nodeClassCache.clear();
        this.categories.clear();
        
        // Reset state
        this.isInitialized = false;
        this.initPromise = null;
        
        // Re-initialize
        return this.initialize();
    }
    
    /**
     * Check if a specific component type is registered
     */
    isComponentRegistered(componentType) {
        return Array.from(this.registeredNodes.values())
            .some(info => info.componentInfo.type === componentType);
    }
    
    /**
     * Search nodes by name or type
     */
    searchNodes(query) {
        const lowerQuery = query.toLowerCase();
        return Array.from(this.registeredNodes.entries())
            .filter(([typeName, info]) => {
                return typeName.toLowerCase().includes(lowerQuery) ||
                       info.componentInfo.type.toLowerCase().includes(lowerQuery) ||
                       info.componentInfo.display_name.toLowerCase().includes(lowerQuery);
            })
            .map(([typeName, info]) => ({ typeName, ...info }));
    }
    
    /**
     * Get mock components for development
     */
    getMockComponents() {
        return [
            {
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
                property_definitions: [
                    {
                        name: 'checkpoint_type',
                        display_name: 'Model Source',
                        type: 'choice',
                        default_value: 'preset',
                        category: 'Model',
                        metadata: { choices: ['preset', 'local', 'huggingface'] }
                    },
                    {
                        name: 'prompt',
                        display_name: 'Prompt',
                        type: 'text',
                        default_value: 'a beautiful landscape',
                        category: 'Prompts',
                        metadata: { editor_type: 'prompt' }
                    }
                ]
            },
            {
                id: 'sampler_component',
                type: 'sampler',
                display_name: 'Sampler',
                category: 'Core',
                icon: '🎲',
                input_ports: [
                    { name: 'pipeline', type: 'pipeline', direction: 'INPUT' },
                    { name: 'conditioning', type: 'conditioning', direction: 'INPUT' }
                ],
                output_ports: [
                    { name: 'image', type: 'image', direction: 'OUTPUT' }
                ],
                property_definitions: [
                    {
                        name: 'steps',
                        display_name: 'Steps',
                        type: 'integer',
                        default_value: 20,
                        category: 'Sampling',
                        metadata: { min: 1, max: 150 }
                    },
                    {
                        name: 'cfg_scale',
                        display_name: 'CFG Scale',
                        type: 'float',
                        default_value: 7.5,
                        category: 'Sampling',
                        metadata: { min: 1.0, max: 20.0, step: 0.1 }
                    }
                ]
            },
            {
                id: 'output_component',
                type: 'output',
                display_name: 'Output',
                category: 'Core',
                icon: '💾',
                input_ports: [
                    { name: 'image', type: 'image', direction: 'INPUT' }
                ],
                output_ports: [],
                property_definitions: [
                    {
                        name: 'filename',
                        display_name: 'Filename',
                        type: 'string',
                        default_value: 'output',
                        category: 'Save',
                        metadata: {}
                    },
                    {
                        name: 'format',
                        display_name: 'Format',
                        type: 'choice',
                        default_value: 'png',
                        category: 'Save',
                        metadata: { choices: ['png', 'jpg', 'webp'] }
                    }
                ]
            }
        ];
    }
    
    /**
     * Create node instance by type name
     */
    createNodeInstance(nodeTypeName) {
        const registeredNode = this.registeredNodes.get(nodeTypeName);
        if (!registeredNode) {
            throw new Error(`Node type '${nodeTypeName}' not found in registry`);
        }
        
        return new registeredNode.nodeClass();
    }
    
    /**
     * Export registry statistics
     */
    getStatistics() {
        const stats = {
            totalNodes: this.registeredNodes.size,
            categories: Array.from(this.categories),
            categoryBreakdown: {}
        };
        
        // Count nodes per category
        for (const category of this.categories) {
            stats.categoryBreakdown[category] = this.getNodesByCategory(category).length;
        }
        
        return stats;
    }
}

/**
 * Auto-initialization utility
 */
class SDFormsAutoRegistry {
    /**
     * Auto-initialize registry when DOM is ready and API is available
     */
    static async autoInitialize(options = {}) {
        const {
            retryAttempts = 3,
            retryDelay = 2000,
            onProgress = null,
            onComplete = null,
            onError = null
        } = options;
        
        let attempt = 0;
        
        while (attempt < retryAttempts) {
            try {
                if (onProgress) {
                    onProgress(`Initializing registry (attempt ${attempt + 1}/${retryAttempts})...`);
                }
                
                // Wait for dependencies
                await this.waitForDependencies();
                
                // Create and initialize registry
                const registry = new SDFormsNodeRegistry();
                
                // Set up callbacks
                if (onProgress) {
                    registry.onNodeRegistered = (typeName, componentInfo) => {
                        onProgress(`Registered: ${typeName}`);
                    };
                }
                
                await registry.initialize();
                
                // Store global reference
                window.sdFormsNodeRegistry = registry;
                
                if (onComplete) {
                    onComplete(registry);
                }
                
                return registry;
                
            } catch (error) {
                attempt++;
                console.error(`Registry initialization attempt ${attempt} failed:`, error);
                
                if (attempt >= retryAttempts) {
                    if (onError) {
                        onError(error);
                    }
                    throw error;
                }
                
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    }
    
    /**
     * Wait for required dependencies to be available
     */
    static async waitForDependencies() {
        const maxWait = 30000; // 30 seconds
        const checkInterval = 100; // 100ms
        let waited = 0;
        
        while (waited < maxWait) {
            if (typeof LiteGraph !== 'undefined' && 
                typeof SDFormsNode !== 'undefined' && 
                typeof apiService !== 'undefined') {
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, checkInterval));
            waited += checkInterval;
        }
        
        throw new Error('Required dependencies not available after 30 seconds');
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SDFormsNodeRegistry, SDFormsAutoRegistry };
} else {
    window.SDFormsNodeRegistry = SDFormsNodeRegistry;
    window.SDFormsAutoRegistry = SDFormsAutoRegistry;
}

// Auto-initialize if in browser and dependencies are ready
if (typeof window !== 'undefined') {
    // Wait for DOM and try to auto-initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            // Give a small delay for other scripts to load
            setTimeout(() => {
                SDFormsAutoRegistry.autoInitialize({
                    onProgress: (message) => console.log('🔄', message),
                    onComplete: (registry) => {
                        console.log('✅ SD Forms Node Registry auto-initialized successfully');
                        console.log('📊 Registry statistics:', registry.getStatistics());
                    },
                    onError: (error) => {
                        console.error('❌ Failed to auto-initialize registry:', error);
                    }
                }).catch(console.error);
            }, 500);
        });
    }
}