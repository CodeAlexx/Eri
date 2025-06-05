/**
 * Keyboard Shortcuts System for SD Forms
 * ComfyUI-compatible shortcuts with additional functionality
 */

class KeyboardShortcuts {
    constructor(app) {
        this.app = app;
        this.shortcuts = new Map();
        this.isEnabled = true;
        this.pressedKeys = new Set();
        this.lastKeyTime = 0;
        this.doubleClickDelay = 300; // ms
        
        this.registerDefaultShortcuts();
        this.setupEventListeners();
    }

    /**
     * Register all default ComfyUI-compatible shortcuts
     */
    registerDefaultShortcuts() {
        // Node operations
        this.register('Space', 'Open node search', () => {
            const mousePos = this.getMousePosition();
            this.app.nodeSearch.show(mousePos);
        });

        this.register('Ctrl+Space', 'Open node search (alt)', () => {
            const mousePos = this.getMousePosition();
            this.app.nodeSearch.show(mousePos);
        });

        // Selection and editing
        this.register('Delete', 'Delete selected nodes', () => {
            this.deleteSelectedNodes();
        });

        this.register('Backspace', 'Delete selected nodes (alt)', () => {
            this.deleteSelectedNodes();
        });

        this.register('Ctrl+A', 'Select all nodes', (e) => {
            e.preventDefault();
            this.selectAllNodes();
        });

        this.register('Ctrl+D', 'Duplicate selected nodes', (e) => {
            e.preventDefault();
            this.duplicateSelectedNodes();
        });

        this.register('Ctrl+C', 'Copy selected nodes', (e) => {
            e.preventDefault();
            this.copySelectedNodes();
        });

        this.register('Ctrl+V', 'Paste nodes', (e) => {
            e.preventDefault();
            this.pasteNodes();
        });

        this.register('Ctrl+X', 'Cut selected nodes', (e) => {
            e.preventDefault();
            this.cutSelectedNodes();
        });

        // Workflow operations
        this.register('Ctrl+S', 'Save workflow', (e) => {
            e.preventDefault();
            this.app.saveWorkflow();
        });

        this.register('Ctrl+O', 'Load workflow', (e) => {
            e.preventDefault();
            this.app.loadWorkflow();
        });

        this.register('Ctrl+N', 'New workflow', (e) => {
            e.preventDefault();
            this.newWorkflow();
        });

        this.register('Ctrl+Z', 'Undo', (e) => {
            e.preventDefault();
            this.undo();
        });

        this.register('Ctrl+Y', 'Redo', (e) => {
            e.preventDefault();
            this.redo();
        });

        this.register('Ctrl+Shift+Z', 'Redo (alt)', (e) => {
            e.preventDefault();
            this.redo();
        });

        // Execution
        this.register('Ctrl+Enter', 'Execute workflow', (e) => {
            e.preventDefault();
            this.app.executeWorkflow();
        });

        this.register('F5', 'Execute workflow (alt)', (e) => {
            e.preventDefault();
            this.app.executeWorkflow();
        });

        this.register('Ctrl+Shift+Enter', 'Validate workflow', (e) => {
            e.preventDefault();
            this.app.validateWorkflow();
        });

        this.register('Escape', 'Cancel operation', () => {
            this.cancelOperation();
        });

        // View operations
        this.register('Ctrl+0', 'Fit canvas to view', (e) => {
            e.preventDefault();
            this.fitCanvasToView();
        });

        this.register('Home', 'Fit canvas to view (alt)', () => {
            this.fitCanvasToView();
        });

        this.register('Ctrl+=', 'Zoom in', (e) => {
            e.preventDefault();
            this.zoomIn();
        });

        this.register('Ctrl+-', 'Zoom out', (e) => {
            e.preventDefault();
            this.zoomOut();
        });

        // Node arrangement
        this.register('Ctrl+Shift+A', 'Auto-arrange nodes', (e) => {
            e.preventDefault();
            this.autoArrangeNodes();
        });

        this.register('Ctrl+G', 'Group selected nodes', (e) => {
            e.preventDefault();
            this.groupSelectedNodes();
        });

        this.register('Ctrl+Shift+G', 'Ungroup selected', (e) => {
            e.preventDefault();
            this.ungroupSelected();
        });

        // Connection operations
        this.register('Ctrl+L', 'Connect nearest nodes', (e) => {
            e.preventDefault();
            this.connectNearestNodes();
        });

        this.register('Ctrl+Shift+L', 'Disconnect selected nodes', (e) => {
            e.preventDefault();
            this.disconnectSelectedNodes();
        });

        // Property editing
        this.register('F2', 'Edit node title', () => {
            this.editNodeTitle();
        });

        this.register('Enter', 'Confirm edit', () => {
            this.confirmEdit();
        });

        // Search and navigation
        this.register('Ctrl+F', 'Find nodes', (e) => {
            e.preventDefault();
            this.app.nodeSearch.show();
        });

        this.register('F3', 'Find next', () => {
            this.findNext();
        });

        this.register('Shift+F3', 'Find previous', () => {
            this.findPrevious();
        });

        // Help
        this.register('F1', 'Show help', () => {
            this.showHelp();
        });

        this.register('Ctrl+/', 'Show shortcuts', (e) => {
            e.preventDefault();
            this.showShortcutsHelp();
        });

        // Canvas controls
        this.register('Ctrl+Shift+R', 'Reset canvas position', (e) => {
            e.preventDefault();
            this.resetCanvasPosition();
        });

        this.register('G', 'Toggle grid', () => {
            this.toggleGrid();
        });

        this.register('Ctrl+Shift+G', 'Toggle grid snap', (e) => {
            e.preventDefault();
            this.toggleGridSnap();
        });

        // Debug operations
        this.register('Ctrl+Shift+D', 'Toggle debug mode', (e) => {
            e.preventDefault();
            this.toggleDebugMode();
        });

        this.register('Ctrl+Shift+C', 'Clear console', (e) => {
            e.preventDefault();
            console.clear();
        });
    }

    /**
     * Register a keyboard shortcut
     * @param {string} key - Key combination (e.g., 'Ctrl+S', 'Space')
     * @param {string} description - Description of the action
     * @param {Function} action - Function to execute
     */
    register(key, description, action) {
        const normalizedKey = this.normalizeKey(key);
        this.shortcuts.set(normalizedKey, { description, action });
    }

    /**
     * Normalize key string for consistent matching
     * @param {string} key - Key combination
     * @returns {string} Normalized key string
     */
    normalizeKey(key) {
        return key.toLowerCase()
            .replace(/\s+/g, '')
            .replace('command', 'ctrl') // Mac compatibility
            .replace('cmd', 'ctrl');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (!this.isEnabled) return;
            if (this.shouldIgnoreEvent(e)) return;
            
            this.handleKeyDown(e);
        });

        document.addEventListener('keyup', (e) => {
            this.pressedKeys.delete(e.code);
        });

        // Handle window blur to clear pressed keys
        window.addEventListener('blur', () => {
            this.pressedKeys.clear();
        });
    }

    /**
     * Check if event should be ignored (e.g., typing in input fields)
     * @param {KeyboardEvent} e - Keyboard event
     * @returns {boolean} True if event should be ignored
     */
    shouldIgnoreEvent(e) {
        const target = e.target;
        const tagName = target.tagName.toLowerCase();
        
        // Ignore events in input fields, textareas, and contenteditable elements
        if (['input', 'textarea', 'select'].includes(tagName)) return true;
        if (target.contentEditable === 'true') return true;
        if (target.classList.contains('property-input')) return true;
        
        // Ignore if search overlay is open (except for allowed keys)
        if (this.app.nodeSearch && this.app.nodeSearch.isVisible) {
            const allowedKeys = ['Escape'];
            return !allowedKeys.includes(e.key);
        }
        
        return false;
    }

    /**
     * Handle keydown events
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeyDown(e) {
        this.pressedKeys.add(e.code);
        
        const keyCombo = this.getKeyCombo(e);
        const shortcut = this.shortcuts.get(keyCombo);
        
        if (shortcut) {
            e.preventDefault();
            e.stopPropagation();
            
            try {
                shortcut.action(e);
                this.showShortcutFeedback(keyCombo, shortcut.description);
            } catch (error) {
                console.error('Error executing shortcut:', error);
                window.errorSystem.showError('Shortcut Error', `Failed to execute ${shortcut.description}`);
            }
        }
    }

    /**
     * Get key combination string from event
     * @param {KeyboardEvent} e - Keyboard event
     * @returns {string} Key combination string
     */
    getKeyCombo(e) {
        const parts = [];
        
        if (e.ctrlKey || e.metaKey) parts.push('ctrl');
        if (e.shiftKey) parts.push('shift');
        if (e.altKey) parts.push('alt');
        
        let key = e.key.toLowerCase();
        
        // Normalize special keys
        if (key === ' ') key = 'space';
        if (key === 'escape') key = 'escape';
        if (key === 'enter') key = 'enter';
        if (key === 'backspace') key = 'backspace';
        if (key === 'delete') key = 'delete';
        if (key === 'home') key = 'home';
        if (key === 'f1') key = 'f1';
        if (key === 'f2') key = 'f2';
        if (key === 'f3') key = 'f3';
        if (key === 'f5') key = 'f5';
        
        parts.push(key);
        return parts.join('+');
    }

    /**
     * Show visual feedback for executed shortcut
     * @param {string} keyCombo - Key combination
     * @param {string} description - Action description
     */
    showShortcutFeedback(keyCombo, description) {
        // Create temporary toast for shortcut feedback
        const feedback = document.createElement('div');
        feedback.className = 'shortcut-feedback';
        feedback.innerHTML = `
            <div class="feedback-key">${keyCombo.toUpperCase()}</div>
            <div class="feedback-desc">${description}</div>
        `;
        
        // Add styles if not already present
        if (!document.getElementById('shortcut-feedback-styles')) {
            const styles = document.createElement('style');
            styles.id = 'shortcut-feedback-styles';
            styles.textContent = `
                .shortcut-feedback {
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 12px;
                    z-index: 10001;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    backdrop-filter: blur(10px);
                    animation: shortcutFeedbackShow 0.3s ease;
                }
                
                .feedback-key {
                    background: rgba(255, 255, 255, 0.2);
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-weight: bold;
                }
                
                @keyframes shortcutFeedbackShow {
                    from { opacity: 0; transform: translateX(-50%) translateY(10px); }
                    to { opacity: 1; transform: translateX(-50%) translateY(0); }
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(feedback);
        
        // Remove after delay
        setTimeout(() => {
            feedback.style.animation = 'shortcutFeedbackShow 0.3s ease reverse';
            setTimeout(() => feedback.remove(), 300);
        }, 1500);
    }

    /**
     * Get current mouse position relative to canvas
     * @returns {Object} Mouse position {x, y}
     */
    getMousePosition() {
        // Get position from app's graph canvas
        if (this.app.graphCanvas) {
            const rect = this.app.graphCanvas.canvas.getBoundingClientRect();
            return {
                x: rect.width / 2,
                y: rect.height / 2
            };
        }
        return { x: 200, y: 200 };
    }

    // Shortcut action implementations
    deleteSelectedNodes() {
        if (this.app.graph && this.app.graph.selected_nodes) {
            const selectedNodes = Object.values(this.app.graph.selected_nodes);
            selectedNodes.forEach(node => {
                if (node && this.app.graph) {
                    this.app.graph.remove(node);
                }
            });
            this.app.updateWorkflowStatus();
        }
    }

    selectAllNodes() {
        if (this.app.graph) {
            this.app.graph.selectAll();
        }
    }

    duplicateSelectedNodes() {
        if (this.app.selectedComponent) {
            this.app.duplicateNode(this.app.selectedComponent);
        }
    }

    copySelectedNodes() {
        // Store selected nodes in clipboard data
        if (this.app.graph && this.app.graph.selected_nodes) {
            const selectedNodes = Object.values(this.app.graph.selected_nodes);
            this.clipboard = selectedNodes.map(node => ({
                type: node.type,
                properties: { ...node.properties },
                pos: [...node.pos]
            }));
            window.errorSystem.showInfo('Copied', `${selectedNodes.length} node(s) copied`);
        }
    }

    pasteNodes() {
        if (this.clipboard && this.clipboard.length > 0) {
            const mousePos = this.getMousePosition();
            this.clipboard.forEach((nodeData, index) => {
                const offset = index * 30; // Offset for multiple nodes
                const pos = {
                    x: mousePos.x + offset,
                    y: mousePos.y + offset
                };
                
                if (nodeData.type.startsWith('sdforms/')) {
                    const type = nodeData.type.replace('sdforms/', '');
                    this.app.addComponentToCanvas(type, pos);
                }
            });
            window.errorSystem.showInfo('Pasted', `${this.clipboard.length} node(s) pasted`);
        }
    }

    cutSelectedNodes() {
        this.copySelectedNodes();
        this.deleteSelectedNodes();
    }

    newWorkflow() {
        if (confirm('Create new workflow? This will clear the current workflow.')) {
            this.app.clearWorkflow();
        }
    }

    undo() {
        // TODO: Implement undo functionality
        window.errorSystem.showInfo('Undo', 'Undo functionality not yet implemented');
    }

    redo() {
        // TODO: Implement redo functionality
        window.errorSystem.showInfo('Redo', 'Redo functionality not yet implemented');
    }

    cancelOperation() {
        if (this.app.nodeSearch && this.app.nodeSearch.isVisible) {
            this.app.nodeSearch.hide();
        } else if (this.app.isExecuting) {
            this.app.cancelExecution();
        } else {
            // Clear selection
            if (this.app.graph) {
                this.app.graph.clearSelection();
            }
        }
    }

    fitCanvasToView() {
        if (this.app.graphCanvas) {
            this.app.graphCanvas.centerOnNode();
        }
    }

    zoomIn() {
        if (this.app.graphCanvas) {
            this.app.graphCanvas.setZoom(this.app.graphCanvas.scale * 1.2);
        }
    }

    zoomOut() {
        if (this.app.graphCanvas) {
            this.app.graphCanvas.setZoom(this.app.graphCanvas.scale * 0.8);
        }
    }

    autoArrangeNodes() {
        window.errorSystem.showInfo('Auto-arrange', 'Auto-arrange functionality not yet implemented');
    }

    groupSelectedNodes() {
        window.errorSystem.showInfo('Group', 'Node grouping functionality not yet implemented');
    }

    ungroupSelected() {
        window.errorSystem.showInfo('Ungroup', 'Node ungrouping functionality not yet implemented');
    }

    connectNearestNodes() {
        window.errorSystem.showInfo('Connect', 'Auto-connect functionality not yet implemented');
    }

    disconnectSelectedNodes() {
        if (this.app.graph && this.app.graph.selected_nodes) {
            const selectedNodes = Object.values(this.app.graph.selected_nodes);
            selectedNodes.forEach(node => {
                if (node.inputs) {
                    node.inputs.forEach(input => {
                        if (input.link) {
                            this.app.graph.removeLink(input.link);
                        }
                    });
                }
            });
        }
    }

    editNodeTitle() {
        if (this.app.selectedComponent) {
            // TODO: Implement inline title editing
            window.errorSystem.showInfo('Edit Title', 'Title editing not yet implemented');
        }
    }

    confirmEdit() {
        // TODO: Implement edit confirmation
    }

    findNext() {
        // TODO: Implement find next
        window.errorSystem.showInfo('Find Next', 'Find functionality not yet implemented');
    }

    findPrevious() {
        // TODO: Implement find previous
        window.errorSystem.showInfo('Find Previous', 'Find functionality not yet implemented');
    }

    showHelp() {
        window.open('https://docs.anthropic.com/en/docs/claude-code', '_blank');
    }

    showShortcutsHelp() {
        this.createShortcutsModal();
    }

    resetCanvasPosition() {
        if (this.app.graphCanvas) {
            this.app.graphCanvas.setZoom(1.0);
            this.app.graphCanvas.offset = [0, 0];
            this.app.graphCanvas.setDirty(true);
        }
    }

    toggleGrid() {
        const gridCheckbox = document.getElementById('show-grid');
        if (gridCheckbox) {
            gridCheckbox.checked = !gridCheckbox.checked;
            gridCheckbox.dispatchEvent(new Event('change'));
        }
    }

    toggleGridSnap() {
        const snapCheckbox = document.getElementById('snap-to-grid');
        if (snapCheckbox) {
            snapCheckbox.checked = !snapCheckbox.checked;
            snapCheckbox.dispatchEvent(new Event('change'));
        }
    }

    toggleDebugMode() {
        // TODO: Implement debug mode
        window.errorSystem.showInfo('Debug Mode', 'Debug mode toggle not yet implemented');
    }

    /**
     * Create and show shortcuts help modal
     */
    createShortcutsModal() {
        // Remove existing modal if present
        const existing = document.getElementById('shortcuts-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'shortcuts-modal';
        modal.className = 'shortcuts-modal';
        
        const shortcuts = Array.from(this.shortcuts.entries())
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, { description }]) => ({ key, description }));

        const categories = {
            'Basic': shortcuts.filter(s => ['space', 'escape', 'delete', 'backspace'].some(k => s.key.includes(k))),
            'Selection': shortcuts.filter(s => s.key.includes('ctrl+a') || s.key.includes('ctrl+d') || s.description.toLowerCase().includes('select')),
            'Copy/Paste': shortcuts.filter(s => ['ctrl+c', 'ctrl+v', 'ctrl+x'].some(k => s.key.includes(k))),
            'Workflow': shortcuts.filter(s => ['ctrl+s', 'ctrl+o', 'ctrl+n', 'ctrl+z'].some(k => s.key.includes(k))),
            'Execution': shortcuts.filter(s => s.description.toLowerCase().includes('execute') || s.description.toLowerCase().includes('validate')),
            'View': shortcuts.filter(s => s.description.toLowerCase().includes('zoom') || s.description.toLowerCase().includes('canvas') || s.description.toLowerCase().includes('grid')),
            'Other': shortcuts.filter(s => !Object.values(categories).flat().includes(s))
        };

        let categoriesHtml = '';
        for (const [category, items] of Object.entries(categories)) {
            if (items.length === 0) continue;
            
            categoriesHtml += `
                <div class="shortcut-category">
                    <h3>${category}</h3>
                    <div class="shortcut-items">
                        ${items.map(({ key, description }) => `
                            <div class="shortcut-item">
                                <kbd class="shortcut-key">${key.toUpperCase().replace(/\+/g, ' + ')}</kbd>
                                <span class="shortcut-desc">${description}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        modal.innerHTML = `
            <div class="shortcuts-content">
                <div class="shortcuts-header">
                    <h2>Keyboard Shortcuts</h2>
                    <button class="shortcuts-close">✕</button>
                </div>
                <div class="shortcuts-body">
                    ${categoriesHtml}
                </div>
            </div>
        `;

        // Add styles
        if (!document.getElementById('shortcuts-modal-styles')) {
            const styles = document.createElement('style');
            styles.id = 'shortcuts-modal-styles';
            styles.textContent = `
                .shortcuts-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(4px);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .shortcuts-content {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-primary);
                    border-radius: 12px;
                    width: 90vw;
                    max-width: 800px;
                    max-height: 90vh;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }

                .shortcuts-header {
                    padding: 20px;
                    border-bottom: 1px solid var(--border-primary);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: var(--bg-tertiary);
                }

                .shortcuts-header h2 {
                    margin: 0;
                    font-size: 18px;
                }

                .shortcuts-close {
                    background: none;
                    border: none;
                    color: var(--text-muted);
                    cursor: pointer;
                    padding: 4px;
                    font-size: 16px;
                    border-radius: 4px;
                    transition: all 0.2s ease;
                }

                .shortcuts-close:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                .shortcuts-body {
                    padding: 20px;
                    overflow-y: auto;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 24px;
                }

                .shortcut-category h3 {
                    margin: 0 0 12px 0;
                    font-size: 14px;
                    font-weight: 600;
                    color: var(--accent-primary);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .shortcut-items {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .shortcut-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid var(--border-primary);
                }

                .shortcut-item:last-child {
                    border-bottom: none;
                }

                .shortcut-key {
                    background: var(--bg-tertiary);
                    border: 1px solid var(--border-secondary);
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                    min-width: 100px;
                    text-align: center;
                    color: var(--text-primary);
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                }

                .shortcut-desc {
                    flex: 1;
                    margin-left: 12px;
                    font-size: 13px;
                    color: var(--text-secondary);
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(modal);

        // Add event listeners
        modal.querySelector('.shortcuts-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        // Close on Escape
        const closeOnEscape = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', closeOnEscape);
            }
        };
        document.addEventListener('keydown', closeOnEscape);
    }

    /**
     * Enable or disable keyboard shortcuts
     * @param {boolean} enabled - Whether shortcuts should be enabled
     */
    setEnabled(enabled) {
        this.isEnabled = enabled;
    }

    /**
     * Get all registered shortcuts
     * @returns {Map} Map of shortcuts
     */
    getShortcuts() {
        return new Map(this.shortcuts);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KeyboardShortcuts;
}