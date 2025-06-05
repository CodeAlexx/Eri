/**
 * Property Editor System for SD Forms Web Interface
 * Handles dynamic property UI generation based on PropertyType
 * Supports metadata, dependencies, and ComfyUI-style interface
 */

class PropertyEditorManager {
    constructor() {
        this.editors = new Map(); // propertyKey -> editor instance
        this.dependencies = new Map(); // propertyKey -> dependent properties
        this.currentNode = null;
        this.updateCallbacks = new Set();
        
        // Register built-in property editors
        this.registerBuiltInEditors();
        
        // Add CSS styles
        this.addPropertyEditorStyles();
    }

    /**
     * Register all built-in property editors
     */
    registerBuiltInEditors() {
        this.registerEditor('STRING', StringPropertyEditor);
        this.registerEditor('TEXT', TextPropertyEditor);
        this.registerEditor('INTEGER', IntegerPropertyEditor);
        this.registerEditor('FLOAT', FloatPropertyEditor);
        this.registerEditor('BOOLEAN', BooleanPropertyEditor);
        this.registerEditor('CHOICE', ChoicePropertyEditor);
        this.registerEditor('SLIDER', SliderPropertyEditor);
        this.registerEditor('FILE_PATH', FilePathPropertyEditor);
        this.registerEditor('DIRECTORY', DirectoryPropertyEditor);
        this.registerEditor('COLOR', ColorPropertyEditor);
    }

    /**
     * Register a property editor class for a specific property type
     */
    registerEditor(propertyType, editorClass) {
        if (!this.propertyEditors) {
            this.propertyEditors = new Map();
        }
        this.propertyEditors.set(propertyType, editorClass);
    }

    /**
     * Create property editors for a node
     */
    createPropertyEditors(node, container) {
        if (!node || !node.componentInfo) return;
        
        this.currentNode = node;
        this.clearEditors();
        
        const { property_definitions } = node.componentInfo;
        if (!property_definitions || property_definitions.length === 0) {
            container.innerHTML = '<div class="no-properties">No properties available</div>';
            return;
        }

        // Group properties by category
        const categories = this.groupPropertiesByCategory(property_definitions);
        
        // Create UI for each category
        container.innerHTML = '';
        Object.entries(categories).forEach(([categoryName, properties]) => {
            const categorySection = this.createCategorySection(categoryName, properties, node);
            container.appendChild(categorySection);
        });

        // Initialize dependencies
        this.initializeDependencies(property_definitions);
        this.updateDependentProperties();
    }

    /**
     * Group properties by category
     */
    groupPropertiesByCategory(propertyDefinitions) {
        const categories = {};
        
        propertyDefinitions.forEach(prop => {
            const category = prop.category || 'General';
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push(prop);
        });

        return categories;
    }

    /**
     * Create a category section with properties
     */
    createCategorySection(categoryName, properties, node) {
        const section = document.createElement('div');
        section.className = 'property-category';
        
        const header = document.createElement('div');
        header.className = 'property-category-header';
        header.innerHTML = `
            <span class="category-name">${categoryName}</span>
            <span class="category-toggle">▼</span>
        `;
        
        const content = document.createElement('div');
        content.className = 'property-category-content';
        
        // Create property editors
        properties.forEach(propertyDef => {
            const propertyElement = this.createPropertyEditor(propertyDef, node);
            if (propertyElement) {
                content.appendChild(propertyElement);
            }
        });

        // Toggle functionality
        header.addEventListener('click', () => {
            const isCollapsed = content.classList.toggle('collapsed');
            header.querySelector('.category-toggle').textContent = isCollapsed ? '▶' : '▼';
        });

        section.appendChild(header);
        section.appendChild(content);
        
        return section;
    }

    /**
     * Create a property editor for a specific property definition
     */
    createPropertyEditor(propertyDef, node) {
        const editorClass = this.propertyEditors.get(propertyDef.type);
        if (!editorClass) {
            console.warn(`No editor found for property type: ${propertyDef.type}`);
            return null;
        }

        const currentValue = node.properties[propertyDef.name] ?? propertyDef.default_value;
        const editor = new editorClass(propertyDef, currentValue, node);
        
        // Store editor reference
        this.editors.set(propertyDef.name, editor);
        
        // Setup change callback
        editor.onChange = (value) => {
            this.handlePropertyChange(propertyDef.name, value, node);
        };

        return editor.createElement();
    }

    /**
     * Handle property value changes
     */
    handlePropertyChange(propertyName, value, node) {
        // Update node property
        node.setPropertyValue(propertyName, value);
        
        // Update dependent properties
        this.updateDependentProperties();
        
        // Notify listeners
        this.updateCallbacks.forEach(callback => {
            try {
                callback(propertyName, value, node);
            } catch (error) {
                console.error('Error in property update callback:', error);
            }
        });

        console.log(`🔧 Property ${propertyName} updated to:`, value);
    }

    /**
     * Initialize property dependencies
     */
    initializeDependencies(propertyDefinitions) {
        this.dependencies.clear();
        
        propertyDefinitions.forEach(prop => {
            if (prop.depends_on && prop.depends_on.length > 0) {
                prop.depends_on.forEach(dependencyProp => {
                    if (!this.dependencies.has(dependencyProp)) {
                        this.dependencies.set(dependencyProp, new Set());
                    }
                    this.dependencies.get(dependencyProp).add(prop.name);
                });
            }
        });
    }

    /**
     * Update visibility of dependent properties
     */
    updateDependentProperties() {
        if (!this.currentNode) return;

        this.dependencies.forEach((dependentProps, parentProp) => {
            const parentValue = this.currentNode.properties[parentProp];
            
            dependentProps.forEach(depProp => {
                const editor = this.editors.get(depProp);
                if (editor) {
                    const shouldShow = this.evaluatePropertyDependency(parentProp, parentValue, depProp);
                    editor.setVisible(shouldShow);
                }
            });
        });
    }

    /**
     * Evaluate if a dependent property should be visible
     */
    evaluatePropertyDependency(parentProp, parentValue, depProp) {
        // Get the dependent property definition
        const depPropDef = this.currentNode.componentInfo.property_definitions.find(p => p.name === depProp);
        if (!depPropDef || !depPropDef.metadata) return true;

        const dependencyRule = depPropDef.metadata.dependency_rule;
        if (!dependencyRule) return true;

        // Simple dependency evaluation - can be extended
        switch (dependencyRule.type) {
            case 'equals':
                return parentValue === dependencyRule.value;
            case 'not_equals':
                return parentValue !== dependencyRule.value;
            case 'in':
                return dependencyRule.values.includes(parentValue);
            case 'not_in':
                return !dependencyRule.values.includes(parentValue);
            case 'greater_than':
                return parentValue > dependencyRule.value;
            case 'less_than':
                return parentValue < dependencyRule.value;
            default:
                return true;
        }
    }

    /**
     * Clear all editors
     */
    clearEditors() {
        this.editors.clear();
        this.dependencies.clear();
    }

    /**
     * Add update callback
     */
    onPropertyUpdate(callback) {
        this.updateCallbacks.add(callback);
    }

    /**
     * Remove update callback
     */
    offPropertyUpdate(callback) {
        this.updateCallbacks.delete(callback);
    }

    /**
     * Get all property values from current node
     */
    getAllPropertyValues() {
        if (!this.currentNode) return {};
        return { ...this.currentNode.properties };
    }

    /**
     * Set multiple property values at once
     */
    setPropertyValues(values) {
        if (!this.currentNode) return;
        
        Object.entries(values).forEach(([name, value]) => {
            this.currentNode.setPropertyValue(name, value);
            const editor = this.editors.get(name);
            if (editor) {
                editor.setValue(value);
            }
        });
        
        this.updateDependentProperties();
    }

    /**
     * Get property editor instance
     */
    getPropertyEditor(propertyName) {
        return this.editors.get(propertyName);
    }

    /**
     * Refresh all property editors
     */
    refreshPropertyEditors() {
        if (!this.currentNode) return;
        
        this.editors.forEach((editor, propertyName) => {
            const currentValue = this.currentNode.properties[propertyName];
            editor.setValue(currentValue);
        });
        
        this.updateDependentProperties();
    }

    /**
     * Add custom validation to a property
     */
    addPropertyValidator(propertyName, validator) {
        const editor = this.editors.get(propertyName);
        if (editor) {
            editor.validator = validator;
        }
    }

    /**
     * Example dependency configurations for common use cases
     */
    static getExampleDependencyConfigs() {
        return {
            // Show advanced options only when "use_advanced" is true
            conditionalVisibility: {
                property_definitions: [
                    {
                        name: "use_advanced",
                        display_name: "Use Advanced Settings",
                        type: "BOOLEAN",
                        default_value: false,
                        category: "General"
                    },
                    {
                        name: "advanced_option1",
                        display_name: "Advanced Option 1",
                        type: "FLOAT",
                        default_value: 1.0,
                        category: "Advanced",
                        depends_on: ["use_advanced"],
                        metadata: {
                            dependency_rule: {
                                type: "equals",
                                value: true
                            }
                        }
                    }
                ]
            },
            
            // Show different options based on selected mode
            modeBasedVisibility: {
                property_definitions: [
                    {
                        name: "mode",
                        display_name: "Processing Mode",
                        type: "CHOICE",
                        default_value: "simple",
                        category: "General",
                        metadata: {
                            choices: ["simple", "advanced", "expert"]
                        }
                    },
                    {
                        name: "simple_setting",
                        display_name: "Simple Setting",
                        type: "INTEGER",
                        default_value: 10,
                        category: "Settings",
                        depends_on: ["mode"],
                        metadata: {
                            dependency_rule: {
                                type: "equals",
                                value: "simple"
                            }
                        }
                    },
                    {
                        name: "expert_setting",
                        display_name: "Expert Setting",
                        type: "FLOAT",
                        default_value: 0.5,
                        category: "Settings",
                        depends_on: ["mode"],
                        metadata: {
                            dependency_rule: {
                                type: "in",
                                values: ["advanced", "expert"]
                            }
                        }
                    }
                ]
            }
        };
    }

    /**
     * Add CSS styles for property editors
     */
    addPropertyEditorStyles() {
        if (document.getElementById('property-editor-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'property-editor-styles';
        styles.textContent = `
            /* Property Editor Styles - ComfyUI inspired */
            .property-category {
                margin-bottom: 8px;
                background: #2a2a2a;
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #3a3a3a;
            }

            .property-category-header {
                padding: 8px 12px;
                background: linear-gradient(135deg, #3a3a3a, #2a2a2a);
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #3a3a3a;
                user-select: none;
                font-size: 12px;
                font-weight: 600;
                color: #e0e0e0;
            }

            .property-category-header:hover {
                background: linear-gradient(135deg, #4a4a4a, #3a3a3a);
            }

            .category-name {
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .category-toggle {
                transition: transform 0.2s ease;
                font-size: 10px;
            }

            .property-category-content {
                padding: 4px;
                max-height: 1000px;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }

            .property-category-content.collapsed {
                max-height: 0;
                padding: 0 4px;
            }

            .property-item {
                display: flex;
                flex-direction: column;
                margin-bottom: 6px;
                padding: 6px 8px;
                background: #1e1e1e;
                border-radius: 4px;
                border: 1px solid #333;
                transition: all 0.2s ease;
            }

            .property-item:hover {
                border-color: #444;
                background: #252525;
            }

            .property-item.hidden {
                display: none;
            }

            .property-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
                font-size: 11px;
                color: #b0b0b0;
                font-weight: 500;
            }

            .property-label-text {
                flex: 1;
            }

            .property-value-display {
                font-size: 10px;
                color: #666;
                background: #333;
                padding: 2px 6px;
                border-radius: 3px;
                min-width: 40px;
                text-align: center;
            }

            .property-editor {
                position: relative;
            }

            /* Input Controls */
            .property-input, .property-select, .property-textarea {
                width: 100%;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 6px 8px;
                color: #e0e0e0;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
                transition: all 0.2s ease;
            }

            .property-input:focus, .property-select:focus, .property-textarea:focus {
                outline: none;
                border-color: #0084ff;
                box-shadow: 0 0 0 2px rgba(0, 132, 255, 0.2);
            }

            .property-input:hover, .property-select:hover, .property-textarea:hover {
                border-color: #555;
            }

            .property-textarea {
                resize: vertical;
                min-height: 60px;
                font-family: 'Consolas', 'Monaco', monospace;
            }

            /* Slider Control */
            .slider-container {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .property-slider {
                flex: 1;
                height: 4px;
                background: #333;
                border-radius: 2px;
                outline: none;
                cursor: pointer;
                appearance: none;
            }

            .property-slider::-webkit-slider-thumb {
                appearance: none;
                width: 14px;
                height: 14px;
                background: #0084ff;
                border-radius: 50%;
                cursor: pointer;
                border: 2px solid #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }

            .property-slider::-webkit-slider-thumb:hover {
                background: #0066cc;
                transform: scale(1.1);
            }

            .slider-input {
                width: 60px;
                text-align: center;
            }

            /* Checkbox Control */
            .checkbox-container {
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 4px 0;
            }

            .property-checkbox {
                width: 16px;
                height: 16px;
                accent-color: #0084ff;
                cursor: pointer;
            }

            .checkbox-label {
                font-size: 11px;
                color: #b0b0b0;
                cursor: pointer;
                user-select: none;
            }

            /* File Input Control */
            .file-editor-container {
                display: flex;
                gap: 4px;
            }

            .file-editor-container .property-input {
                flex: 1;
            }

            .file-browse-btn {
                padding: 6px 10px;
                background: #0084ff;
                border: none;
                border-radius: 3px;
                color: white;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.2s ease;
            }

            .file-browse-btn:hover {
                background: #0066cc;
            }

            /* Color Picker */
            .color-editor-container {
                display: flex;
                gap: 6px;
                align-items: center;
            }

            .color-preview {
                width: 24px;
                height: 24px;
                border-radius: 3px;
                border: 1px solid #444;
                cursor: pointer;
            }

            .property-color-input {
                opacity: 0;
                position: absolute;
                width: 24px;
                height: 24px;
                cursor: pointer;
            }

            /* Range indicators */
            .range-indicator {
                display: flex;
                justify-content: space-between;
                font-size: 9px;
                color: #666;
                margin-top: 2px;
            }

            /* No properties message */
            .no-properties {
                text-align: center;
                color: #666;
                padding: 20px;
                font-size: 12px;
                font-style: italic;
            }

            /* Validation states */
            .property-input.invalid, .property-select.invalid {
                border-color: #dc3545;
                box-shadow: 0 0 0 2px rgba(220, 53, 69, 0.2);
            }

            .property-input.valid, .property-select.valid {
                border-color: #28a745;
            }

            /* Animations */
            @keyframes propertyUpdate {
                0% { background-color: #0084ff; }
                100% { background-color: transparent; }
            }

            .property-item.updated {
                animation: propertyUpdate 0.3s ease;
            }
        `;
        document.head.appendChild(styles);
    }
}

/**
 * Base Property Editor Class
 */
class BasePropertyEditor {
    constructor(propertyDef, value, node) {
        this.propertyDef = propertyDef;
        this.value = value;
        this.node = node;
        this.element = null;
        this.onChange = null;
        this.validator = propertyDef.validator;
    }

    createElement() {
        this.element = document.createElement('div');
        this.element.className = 'property-item';
        this.element.innerHTML = this.generateHTML();
        this.setupEventListeners();
        return this.element;
    }

    generateHTML() {
        return `
            <div class="property-label">
                <span class="property-label-text">${this.propertyDef.display_name}</span>
                <span class="property-value-display" data-value-display>${this.formatDisplayValue(this.value)}</span>
            </div>
            <div class="property-editor">
                ${this.generateEditorHTML()}
            </div>
        `;
    }

    generateEditorHTML() {
        // Override in subclasses
        return `<input type="text" class="property-input" value="${this.value || ''}" />`;
    }

    setupEventListeners() {
        // Override in subclasses
    }

    setValue(value) {
        this.value = value;
        this.updateValueDisplay();
        this.updateEditor();
    }

    updateValueDisplay() {
        const display = this.element.querySelector('[data-value-display]');
        if (display) {
            display.textContent = this.formatDisplayValue(this.value);
        }
    }

    updateEditor() {
        // Override in subclasses to update the editor control
    }

    formatDisplayValue(value) {
        if (value === null || value === undefined) return '-';
        if (typeof value === 'number') return value.toFixed(2).replace(/\.?0+$/, '');
        if (typeof value === 'string' && value.length > 20) return value.substring(0, 17) + '...';
        return String(value);
    }

    validate(value) {
        if (this.validator) {
            return this.validator(value);
        }
        return true;
    }

    setVisible(visible) {
        if (this.element) {
            this.element.classList.toggle('hidden', !visible);
        }
    }

    triggerChange(value) {
        if (this.validate(value)) {
            this.setValue(value);
            if (this.onChange) {
                this.onChange(value);
            }
            this.element.classList.remove('invalid');
            this.element.classList.add('valid', 'updated');
            setTimeout(() => this.element.classList.remove('updated'), 300);
        } else {
            this.element.classList.add('invalid');
        }
    }
}

/**
 * String Property Editor
 */
class StringPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const placeholder = this.propertyDef.metadata?.placeholder || '';
        return `<input type="text" class="property-input" value="${this.value || ''}" placeholder="${placeholder}" />`;
    }

    setupEventListeners() {
        const input = this.element.querySelector('.property-input');
        input.addEventListener('input', (e) => {
            this.triggerChange(e.target.value);
        });
    }

    updateEditor() {
        const input = this.element.querySelector('.property-input');
        if (input) input.value = this.value || '';
    }
}

/**
 * Text Property Editor (multiline)
 */
class TextPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const placeholder = this.propertyDef.metadata?.placeholder || '';
        const rows = this.propertyDef.metadata?.rows || 3;
        return `<textarea class="property-textarea property-input" rows="${rows}" placeholder="${placeholder}">${this.value || ''}</textarea>`;
    }

    setupEventListeners() {
        const textarea = this.element.querySelector('.property-textarea');
        textarea.addEventListener('input', (e) => {
            this.triggerChange(e.target.value);
        });
    }

    updateEditor() {
        const textarea = this.element.querySelector('.property-textarea');
        if (textarea) textarea.value = this.value || '';
    }
}

/**
 * Integer Property Editor
 */
class IntegerPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const min = this.propertyDef.metadata?.min ?? '';
        const max = this.propertyDef.metadata?.max ?? '';
        const step = this.propertyDef.metadata?.step || 1;
        
        return `
            <input type="number" class="property-input" 
                   value="${this.value || 0}" 
                   min="${min}" max="${max}" step="${step}" />
            ${min !== '' && max !== '' ? `<div class="range-indicator"><span>${min}</span><span>${max}</span></div>` : ''}
        `;
    }

    setupEventListeners() {
        const input = this.element.querySelector('.property-input');
        input.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            if (!isNaN(value)) {
                this.triggerChange(value);
            }
        });
    }

    updateEditor() {
        const input = this.element.querySelector('.property-input');
        if (input) input.value = this.value || 0;
    }

    formatDisplayValue(value) {
        return value !== null && value !== undefined ? String(Math.round(value)) : '-';
    }
}

/**
 * Float Property Editor
 */
class FloatPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const min = this.propertyDef.metadata?.min ?? '';
        const max = this.propertyDef.metadata?.max ?? '';
        const step = this.propertyDef.metadata?.step || 0.01;
        
        return `
            <input type="number" class="property-input" 
                   value="${this.value || 0}" 
                   min="${min}" max="${max}" step="${step}" />
            ${min !== '' && max !== '' ? `<div class="range-indicator"><span>${min}</span><span>${max}</span></div>` : ''}
        `;
    }

    setupEventListeners() {
        const input = this.element.querySelector('.property-input');
        input.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (!isNaN(value)) {
                this.triggerChange(value);
            }
        });
    }

    updateEditor() {
        const input = this.element.querySelector('.property-input');
        if (input) input.value = this.value || 0;
    }
}

/**
 * Boolean Property Editor
 */
class BooleanPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const checked = this.value ? 'checked' : '';
        return `
            <div class="checkbox-container">
                <input type="checkbox" class="property-checkbox" ${checked} />
                <span class="checkbox-label">${this.propertyDef.display_name}</span>
            </div>
        `;
    }

    setupEventListeners() {
        const checkbox = this.element.querySelector('.property-checkbox');
        checkbox.addEventListener('change', (e) => {
            this.triggerChange(e.target.checked);
        });
    }

    updateEditor() {
        const checkbox = this.element.querySelector('.property-checkbox');
        if (checkbox) checkbox.checked = !!this.value;
    }

    formatDisplayValue(value) {
        return value ? 'true' : 'false';
    }
}

/**
 * Choice Property Editor (dropdown)
 */
class ChoicePropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const choices = this.propertyDef.metadata?.choices || [];
        const options = choices.map(choice => 
            `<option value="${choice}" ${choice === this.value ? 'selected' : ''}>${choice}</option>`
        ).join('');
        
        return `<select class="property-select">${options}</select>`;
    }

    setupEventListeners() {
        const select = this.element.querySelector('.property-select');
        select.addEventListener('change', (e) => {
            this.triggerChange(e.target.value);
        });
    }

    updateEditor() {
        const select = this.element.querySelector('.property-select');
        if (select) select.value = this.value || '';
    }
}

/**
 * Slider Property Editor
 */
class SliderPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const min = this.propertyDef.metadata?.min || 0;
        const max = this.propertyDef.metadata?.max || 100;
        const step = this.propertyDef.metadata?.step || 1;
        
        return `
            <div class="slider-container">
                <input type="range" class="property-slider" 
                       min="${min}" max="${max}" step="${step}" value="${this.value || min}" />
                <input type="number" class="property-input slider-input" 
                       min="${min}" max="${max}" step="${step}" value="${this.value || min}" />
            </div>
            <div class="range-indicator">
                <span>${min}</span>
                <span>${max}</span>
            </div>
        `;
    }

    setupEventListeners() {
        const slider = this.element.querySelector('.property-slider');
        const input = this.element.querySelector('.slider-input');
        
        const updateValue = (value) => {
            const numValue = parseFloat(value);
            if (!isNaN(numValue)) {
                slider.value = numValue;
                input.value = numValue;
                this.triggerChange(numValue);
            }
        };

        slider.addEventListener('input', (e) => updateValue(e.target.value));
        input.addEventListener('input', (e) => updateValue(e.target.value));
    }

    updateEditor() {
        const slider = this.element.querySelector('.property-slider');
        const input = this.element.querySelector('.slider-input');
        if (slider) slider.value = this.value || 0;
        if (input) input.value = this.value || 0;
    }
}

/**
 * File Path Property Editor
 */
class FilePathPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const accept = this.propertyDef.metadata?.accept || '';
        return `
            <div class="file-editor-container">
                <input type="text" class="property-input" value="${this.value || ''}" placeholder="Enter file path..." />
                <button type="button" class="file-browse-btn" title="Browse files">📁</button>
                <input type="file" class="file-input" style="display: none;" accept="${accept}" />
            </div>
        `;
    }

    setupEventListeners() {
        const textInput = this.element.querySelector('.property-input');
        const browseBtn = this.element.querySelector('.file-browse-btn');
        const fileInput = this.element.querySelector('.file-input');

        textInput.addEventListener('input', (e) => {
            this.triggerChange(e.target.value);
        });

        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                // For web, we might need to handle file differently
                // For now, just use the file name
                this.triggerChange(file.name);
                textInput.value = file.name;
            }
        });
    }

    updateEditor() {
        const input = this.element.querySelector('.property-input');
        if (input) input.value = this.value || '';
    }

    formatDisplayValue(value) {
        if (!value) return '-';
        const parts = value.split('/');
        return parts[parts.length - 1] || value;
    }
}

/**
 * Directory Property Editor
 */
class DirectoryPropertyEditor extends FilePathPropertyEditor {
    generateEditorHTML() {
        return `
            <div class="file-editor-container">
                <input type="text" class="property-input" value="${this.value || ''}" placeholder="Enter directory path..." />
                <button type="button" class="file-browse-btn" title="Browse directories">📂</button>
            </div>
        `;
    }

    setupEventListeners() {
        const textInput = this.element.querySelector('.property-input');
        const browseBtn = this.element.querySelector('.file-browse-btn');

        textInput.addEventListener('input', (e) => {
            this.triggerChange(e.target.value);
        });

        browseBtn.addEventListener('click', () => {
            // For web, directory picking is limited
            // Could implement a custom directory browser or use webkitdirectory
            alert('Directory browsing not implemented in web version');
        });
    }
}

/**
 * Color Property Editor
 */
class ColorPropertyEditor extends BasePropertyEditor {
    generateEditorHTML() {
        const color = this.value || '#000000';
        return `
            <div class="color-editor-container">
                <div class="color-preview" style="background-color: ${color};">
                    <input type="color" class="property-color-input" value="${color}" />
                </div>
                <input type="text" class="property-input" value="${color}" placeholder="#000000" />
            </div>
        `;
    }

    setupEventListeners() {
        const colorInput = this.element.querySelector('.property-color-input');
        const textInput = this.element.querySelector('.property-input');
        const preview = this.element.querySelector('.color-preview');

        const updateColor = (color) => {
            if (this.isValidColor(color)) {
                colorInput.value = color;
                textInput.value = color;
                preview.style.backgroundColor = color;
                this.triggerChange(color);
            }
        };

        colorInput.addEventListener('input', (e) => updateColor(e.target.value));
        textInput.addEventListener('input', (e) => updateColor(e.target.value));
    }

    updateEditor() {
        const colorInput = this.element.querySelector('.property-color-input');
        const textInput = this.element.querySelector('.property-input');
        const preview = this.element.querySelector('.color-preview');
        
        const color = this.value || '#000000';
        if (colorInput) colorInput.value = color;
        if (textInput) textInput.value = color;
        if (preview) preview.style.backgroundColor = color;
    }

    isValidColor(color) {
        return /^#[0-9A-F]{6}$/i.test(color);
    }

    formatDisplayValue(value) {
        return value || '#000000';
    }
}

// Create singleton instance
const propertyEditorManager = new PropertyEditorManager();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PropertyEditorManager,
        BasePropertyEditor,
        propertyEditorManager
    };
} else {
    // Browser globals
    window.PropertyEditorManager = PropertyEditorManager;
    window.BasePropertyEditor = BasePropertyEditor;
    window.propertyEditorManager = propertyEditorManager;
}