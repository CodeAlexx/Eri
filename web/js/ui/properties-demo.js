/**
 * Property Editor Demo - Shows various property types and dependencies
 */

// Example component with various property types for testing
const demoComponentInfo = {
    type: "demo_component",
    display_name: "Demo Component",
    category: "Demo",
    property_definitions: [
        // Basic types
        {
            name: "text_input",
            display_name: "Text Input",
            type: "STRING",
            default_value: "Hello World",
            category: "Basic",
            metadata: {
                placeholder: "Enter some text..."
            }
        },
        {
            name: "multiline_text",
            display_name: "Multiline Text",
            type: "TEXT",
            default_value: "Line 1\nLine 2\nLine 3",
            category: "Basic",
            metadata: {
                rows: 4,
                placeholder: "Enter multiline text..."
            }
        },
        {
            name: "integer_value",
            display_name: "Integer Value",
            type: "INTEGER",
            default_value: 42,
            category: "Numbers",
            metadata: {
                min: 0,
                max: 100,
                step: 1
            }
        },
        {
            name: "float_value",
            display_name: "Float Value",
            type: "FLOAT",
            default_value: 3.14,
            category: "Numbers",
            metadata: {
                min: 0.0,
                max: 10.0,
                step: 0.01
            }
        },
        {
            name: "boolean_flag",
            display_name: "Boolean Flag",
            type: "BOOLEAN",
            default_value: true,
            category: "Basic"
        },
        {
            name: "choice_option",
            display_name: "Choice Option",
            type: "CHOICE",
            default_value: "option2",
            category: "Basic",
            metadata: {
                choices: ["option1", "option2", "option3", "option4"]
            }
        },
        {
            name: "slider_value",
            display_name: "Slider Value",
            type: "SLIDER",
            default_value: 50,
            category: "Advanced",
            metadata: {
                min: 0,
                max: 100,
                step: 5
            }
        },
        {
            name: "file_path",
            display_name: "File Path",
            type: "FILE_PATH",
            default_value: "",
            category: "Advanced",
            metadata: {
                accept: ".png,.jpg,.jpeg"
            }
        },
        {
            name: "color_picker",
            display_name: "Color Picker",
            type: "COLOR",
            default_value: "#ff6b35",
            category: "Advanced"
        },
        
        // Dependency examples
        {
            name: "enable_advanced",
            display_name: "Enable Advanced Options",
            type: "BOOLEAN",
            default_value: false,
            category: "Dependencies"
        },
        {
            name: "advanced_setting1",
            display_name: "Advanced Setting 1",
            type: "FLOAT",
            default_value: 1.5,
            category: "Dependencies",
            depends_on: ["enable_advanced"],
            metadata: {
                dependency_rule: {
                    type: "equals",
                    value: true
                },
                min: 0.1,
                max: 5.0,
                step: 0.1
            }
        },
        {
            name: "mode_selector",
            display_name: "Processing Mode",
            type: "CHOICE",
            default_value: "simple",
            category: "Dependencies",
            metadata: {
                choices: ["simple", "advanced", "expert"]
            }
        },
        {
            name: "simple_only_option",
            display_name: "Simple Mode Option",
            type: "INTEGER",
            default_value: 10,
            category: "Dependencies",
            depends_on: ["mode_selector"],
            metadata: {
                dependency_rule: {
                    type: "equals",
                    value: "simple"
                },
                min: 1,
                max: 20
            }
        },
        {
            name: "advanced_expert_option",
            display_name: "Advanced/Expert Option",
            type: "STRING",
            default_value: "Complex setting",
            category: "Dependencies",
            depends_on: ["mode_selector"],
            metadata: {
                dependency_rule: {
                    type: "in",
                    values: ["advanced", "expert"]
                },
                placeholder: "Enter advanced configuration..."
            }
        },
        {
            name: "expert_only_slider",
            display_name: "Expert Only Slider",
            type: "SLIDER",
            default_value: 75,
            category: "Dependencies",
            depends_on: ["mode_selector"],
            metadata: {
                dependency_rule: {
                    type: "equals",
                    value: "expert"
                },
                min: 0,
                max: 100,
                step: 1
            }
        }
    ]
};

/**
 * Create a demo node for testing property editors
 */
function createDemoNode() {
    const demoNode = {
        componentInfo: demoComponentInfo,
        componentId: "demo_" + Math.random().toString(36).substr(2, 9),
        properties: {},
        
        setPropertyValue: function(name, value) {
            this.properties[name] = value;
            console.log(`Demo node property ${name} set to:`, value);
        },
        
        getPropertyValue: function(name) {
            return this.properties[name];
        }
    };
    
    // Initialize with default values
    demoComponentInfo.property_definitions.forEach(prop => {
        demoNode.properties[prop.name] = prop.default_value;
    });
    
    return demoNode;
}

/**
 * Initialize property editor demo
 */
function initPropertyEditorDemo() {
    console.log('🎮 Initializing Property Editor Demo...');
    
    // Create demo container if it doesn't exist
    let demoContainer = document.getElementById('property-demo-container');
    if (!demoContainer) {
        demoContainer = document.createElement('div');
        demoContainer.id = 'property-demo-container';
        demoContainer.style.cssText = `
            position: fixed;
            top: 50px;
            left: 50px;
            width: 400px;
            max-height: 80vh;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 8px;
            overflow-y: auto;
            z-index: 2000;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        `;
        
        // Add header
        const header = document.createElement('div');
        header.innerHTML = `
            <div style="padding: 12px 16px; background: #2a2a2a; border-bottom: 1px solid #444; display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #e0e0e0; font-size: 14px;">Property Editor Demo</h3>
                <button onclick="closeDemoContainer()" style="background: #dc3545; border: none; color: white; padding: 4px 8px; border-radius: 3px; cursor: pointer;">✖</button>
            </div>
        `;
        
        // Add content area
        const content = document.createElement('div');
        content.id = 'demo-properties-container';
        content.style.padding = '8px';
        
        demoContainer.appendChild(header);
        demoContainer.appendChild(content);
        document.body.appendChild(demoContainer);
    }
    
    // Create demo node and populate properties
    const demoNode = createDemoNode();
    const container = document.getElementById('demo-properties-container');
    
    // Create property editors
    propertyEditorManager.createPropertyEditors(demoNode, container);
    
    // Add property update listener
    propertyEditorManager.onPropertyUpdate((propertyName, value, node) => {
        console.log(`✅ Property updated: ${propertyName} = ${JSON.stringify(value)}`);
    });
    
    console.log('✅ Property Editor Demo initialized!');
    console.log('📋 Demo node properties:', demoNode.properties);
}

/**
 * Close demo container
 */
function closeDemoContainer() {
    const container = document.getElementById('property-demo-container');
    if (container) {
        container.remove();
    }
}

/**
 * Export demo functions to global scope
 */
if (typeof window !== 'undefined') {
    window.initPropertyEditorDemo = initPropertyEditorDemo;
    window.closeDemoContainer = closeDemoContainer;
    window.createDemoNode = createDemoNode;
    window.demoComponentInfo = demoComponentInfo;
}

// Auto-initialize demo if this script is loaded directly
if (typeof document !== 'undefined' && document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Add demo button to page if it doesn't exist
        if (!document.getElementById('demo-button')) {
            const demoButton = document.createElement('button');
            demoButton.id = 'demo-button';
            demoButton.textContent = '🎮 Property Demo';
            demoButton.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                background: #0084ff;
                border: none;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                z-index: 3000;
                font-size: 12px;
            `;
            demoButton.addEventListener('click', initPropertyEditorDemo);
            document.body.appendChild(demoButton);
        }
    });
}