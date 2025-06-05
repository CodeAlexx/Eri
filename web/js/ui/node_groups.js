/**
 * Node Groups and Subflows System for SD Forms
 * Allows organizing nodes into groups and creating reusable subflows
 */

class NodeGroupManager {
    constructor(app) {
        this.app = app;
        this.groups = new Map();
        this.groupCounter = 0;
        this.selectedGroups = new Set();
        
        this.setupGroupStyles();
        this.registerGroupShortcuts();
    }

    /**
     * Create a new group from selected nodes
     * @param {string} title - Group title
     * @param {string} color - Group color
     * @returns {string} Group ID
     */
    createGroup(title = null, color = null) {
        const selectedNodes = this.getSelectedNodes();
        if (selectedNodes.length === 0) {
            window.errorSystem.showWarning('No Selection', 'Please select nodes to group');
            return null;
        }

        const groupId = `group_${++this.groupCounter}`;
        const groupTitle = title || `Group ${this.groupCounter}`;
        const groupColor = color || this.getRandomGroupColor();

        // Calculate group bounds
        const bounds = this.calculateNodesBounds(selectedNodes);
        
        // Create group object
        const group = {
            id: groupId,
            title: groupTitle,
            color: groupColor,
            nodes: selectedNodes.map(node => node.id),
            bounds: bounds,
            collapsed: false,
            created: Date.now(),
            metadata: {
                description: '',
                tags: [],
                version: '1.0'
            }
        };

        this.groups.set(groupId, group);

        // Add visual group representation
        this.createGroupVisual(group);

        // Update node properties to include group membership
        selectedNodes.forEach(node => {
            node.groupId = groupId;
            node.setDirtyCanvas(true);
        });

        window.errorSystem.showSuccess('Group Created', `Created group "${groupTitle}" with ${selectedNodes.length} nodes`);
        return groupId;
    }

    /**
     * Create visual representation of group
     * @param {Object} group - Group object
     */
    createGroupVisual(group) {
        const groupElement = document.createElement('div');
        groupElement.id = `group-visual-${group.id}`;
        groupElement.className = 'node-group-visual';
        groupElement.style.cssText = `
            position: absolute;
            border: 2px solid ${group.color};
            border-radius: 8px;
            background: ${group.color}20;
            pointer-events: none;
            z-index: -1;
            transition: all 0.3s ease;
        `;

        // Add group header
        const header = document.createElement('div');
        header.className = 'group-header';
        header.style.cssText = `
            position: absolute;
            top: -25px;
            left: 0;
            background: ${group.color};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            pointer-events: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        `;
        header.textContent = group.title;

        // Add collapse/expand button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'group-toggle';
        toggleBtn.style.cssText = `
            background: none;
            border: none;
            color: white;
            margin-left: 8px;
            cursor: pointer;
            font-size: 10px;
        `;
        toggleBtn.textContent = group.collapsed ? '▶' : '▼';
        header.appendChild(toggleBtn);

        groupElement.appendChild(header);

        // Add to canvas
        const canvasContainer = document.querySelector('.canvas-container');
        if (canvasContainer) {
            canvasContainer.appendChild(groupElement);
        }

        // Setup event listeners
        this.setupGroupEventListeners(group, groupElement, header, toggleBtn);
        
        // Update visual position and size
        this.updateGroupVisual(group);
    }

    /**
     * Setup event listeners for group visual elements
     * @param {Object} group - Group object
     * @param {HTMLElement} groupElement - Group visual element
     * @param {HTMLElement} header - Group header element
     * @param {HTMLElement} toggleBtn - Toggle button element
     */
    setupGroupEventListeners(group, groupElement, header, toggleBtn) {
        // Double-click to edit title
        header.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.editGroupTitle(group);
        });

        // Toggle collapse/expand
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleGroupCollapse(group.id);
        });

        // Context menu
        header.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.showGroupContextMenu(group, e.clientX, e.clientY);
        });

        // Selection
        header.addEventListener('click', (e) => {
            e.stopPropagation();
            if (e.ctrlKey || e.metaKey) {
                this.toggleGroupSelection(group.id);
            } else {
                this.selectGroup(group.id);
            }
        });
    }

    /**
     * Update group visual position and size
     * @param {Object} group - Group object
     */
    updateGroupVisual(group) {
        const groupElement = document.getElementById(`group-visual-${group.id}`);
        if (!groupElement) return;

        const bounds = this.calculateGroupBounds(group);
        if (!bounds) return;

        const canvas = this.app.graphCanvas;
        if (!canvas) return;

        // Convert graph coordinates to screen coordinates
        const padding = 20;
        const screenBounds = {
            x: (bounds.x - padding) * canvas.scale + canvas.offset[0],
            y: (bounds.y - padding) * canvas.scale + canvas.offset[1],
            width: (bounds.width + padding * 2) * canvas.scale,
            height: (bounds.height + padding * 2) * canvas.scale
        };

        groupElement.style.left = `${screenBounds.x}px`;
        groupElement.style.top = `${screenBounds.y}px`;
        groupElement.style.width = `${screenBounds.width}px`;
        groupElement.style.height = `${screenBounds.height}px`;

        // Update opacity based on collapse state
        if (group.collapsed) {
            groupElement.style.opacity = '0.3';
            // Hide nodes in collapsed groups
            this.setGroupNodesVisibility(group.id, false);
        } else {
            groupElement.style.opacity = '1';
            this.setGroupNodesVisibility(group.id, true);
        }
    }

    /**
     * Calculate bounds for a group based on its nodes
     * @param {Object} group - Group object
     * @returns {Object} Bounds {x, y, width, height}
     */
    calculateGroupBounds(group) {
        const nodes = this.getGroupNodes(group.id);
        if (nodes.length === 0) return null;
        
        return this.calculateNodesBounds(nodes);
    }

    /**
     * Calculate bounds for a set of nodes
     * @param {Array} nodes - Array of nodes
     * @returns {Object} Bounds {x, y, width, height}
     */
    calculateNodesBounds(nodes) {
        if (nodes.length === 0) return null;

        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        nodes.forEach(node => {
            const x = node.pos[0];
            const y = node.pos[1];
            const width = node.size[0];
            const height = node.size[1];

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x + width);
            maxY = Math.max(maxY, y + height);
        });

        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }

    /**
     * Toggle group collapse state
     * @param {string} groupId - Group ID
     */
    toggleGroupCollapse(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        group.collapsed = !group.collapsed;
        this.updateGroupVisual(group);

        const toggleBtn = document.querySelector(`#group-visual-${groupId} .group-toggle`);
        if (toggleBtn) {
            toggleBtn.textContent = group.collapsed ? '▶' : '▼';
        }

        window.errorSystem.showInfo('Group ' + (group.collapsed ? 'Collapsed' : 'Expanded'), group.title);
    }

    /**
     * Set visibility of nodes in a group
     * @param {string} groupId - Group ID
     * @param {boolean} visible - Whether nodes should be visible
     */
    setGroupNodesVisibility(groupId, visible) {
        const nodes = this.getGroupNodes(groupId);
        nodes.forEach(node => {
            if (visible) {
                node.flags = node.flags || {};
                delete node.flags.hidden;
            } else {
                node.flags = node.flags || {};
                node.flags.hidden = true;
            }
            node.setDirtyCanvas(true);
        });
    }

    /**
     * Get nodes belonging to a group
     * @param {string} groupId - Group ID
     * @returns {Array} Array of nodes
     */
    getGroupNodes(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return [];

        return this.app.graph._nodes.filter(node => 
            group.nodes.includes(node.id)
        );
    }

    /**
     * Get currently selected nodes
     * @returns {Array} Array of selected nodes
     */
    getSelectedNodes() {
        if (!this.app.graph || !this.app.graph.selected_nodes) return [];
        return Object.values(this.app.graph.selected_nodes);
    }

    /**
     * Select all nodes in a group
     * @param {string} groupId - Group ID
     */
    selectGroup(groupId) {
        // Clear current selection
        if (this.app.graph) {
            this.app.graph.clearSelection();
        }

        // Select all nodes in group
        const nodes = this.getGroupNodes(groupId);
        nodes.forEach(node => {
            if (this.app.graph) {
                this.app.graph.selectNode(node, true);
            }
        });

        this.selectedGroups.clear();
        this.selectedGroups.add(groupId);
        this.updateGroupSelection();
    }

    /**
     * Toggle group selection
     * @param {string} groupId - Group ID
     */
    toggleGroupSelection(groupId) {
        if (this.selectedGroups.has(groupId)) {
            this.selectedGroups.delete(groupId);
            // Deselect group nodes
            const nodes = this.getGroupNodes(groupId);
            nodes.forEach(node => {
                if (this.app.graph) {
                    this.app.graph.selectNode(node, false);
                }
            });
        } else {
            this.selectedGroups.add(groupId);
            // Select group nodes
            const nodes = this.getGroupNodes(groupId);
            nodes.forEach(node => {
                if (this.app.graph) {
                    this.app.graph.selectNode(node, true);
                }
            });
        }
        this.updateGroupSelection();
    }

    /**
     * Update visual selection state of groups
     */
    updateGroupSelection() {
        this.groups.forEach((group, groupId) => {
            const groupElement = document.getElementById(`group-visual-${groupId}`);
            if (groupElement) {
                if (this.selectedGroups.has(groupId)) {
                    groupElement.classList.add('selected');
                } else {
                    groupElement.classList.remove('selected');
                }
            }
        });
    }

    /**
     * Edit group title
     * @param {Object} group - Group object
     */
    editGroupTitle(group) {
        const newTitle = prompt('Enter group title:', group.title);
        if (newTitle && newTitle.trim() !== group.title) {
            group.title = newTitle.trim();
            
            const header = document.querySelector(`#group-visual-${group.id} .group-header`);
            if (header) {
                header.childNodes[0].textContent = group.title;
            }
            
            window.errorSystem.showSuccess('Group Renamed', `Group renamed to "${group.title}"`);
        }
    }

    /**
     * Show group context menu
     * @param {Object} group - Group object
     * @param {number} x - Mouse X position
     * @param {number} y - Mouse Y position
     */
    showGroupContextMenu(group, x, y) {
        // Remove existing context menu
        const existing = document.getElementById('group-context-menu');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.id = 'group-context-menu';
        menu.className = 'context-menu group-context-menu';
        menu.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            z-index: 10001;
        `;

        const menuItems = [
            { icon: '✏️', text: 'Rename', action: () => this.editGroupTitle(group) },
            { icon: '🎨', text: 'Change Color', action: () => this.changeGroupColor(group) },
            { icon: group.collapsed ? '▼' : '▲', text: group.collapsed ? 'Expand' : 'Collapse', action: () => this.toggleGroupCollapse(group.id) },
            { separator: true },
            { icon: '📋', text: 'Copy Group', action: () => this.copyGroup(group.id) },
            { icon: '📄', text: 'Duplicate Group', action: () => this.duplicateGroup(group.id) },
            { separator: true },
            { icon: '📦', text: 'Create Subflow', action: () => this.createSubflow(group.id) },
            { icon: '💾', text: 'Export Group', action: () => this.exportGroup(group.id) },
            { separator: true },
            { icon: '🗑️', text: 'Delete Group', action: () => this.deleteGroup(group.id), danger: true }
        ];

        menu.innerHTML = menuItems.map(item => {
            if (item.separator) {
                return '<div class="menu-separator"></div>';
            }
            return `
                <div class="menu-item ${item.danger ? 'danger' : ''}" data-action="${item.text}">
                    <span class="menu-icon">${item.icon}</span>
                    ${item.text}
                </div>
            `;
        }).join('');

        document.body.appendChild(menu);

        // Add event listeners
        menu.querySelectorAll('.menu-item').forEach((item, index) => {
            const menuItem = menuItems.filter(m => !m.separator)[index];
            if (menuItem) {
                item.addEventListener('click', (e) => {
                    e.stopPropagation();
                    menuItem.action();
                    menu.remove();
                });
            }
        });

        // Close on click outside
        setTimeout(() => {
            const closeMenu = (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener('click', closeMenu);
                }
            };
            document.addEventListener('click', closeMenu);
        }, 100);
    }

    /**
     * Change group color
     * @param {Object} group - Group object
     */
    changeGroupColor(group) {
        const colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
            '#9b59b6', '#1abc9c', '#34495e', '#e67e22',
            '#f1c40f', '#95a5a6', '#ff6b6b', '#4ecdc4'
        ];

        const colorPicker = document.createElement('div');
        colorPicker.className = 'color-picker-popup';
        colorPicker.style.cssText = `
            position: fixed;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 16px;
            z-index: 10002;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        `;

        colors.forEach(color => {
            const colorButton = document.createElement('button');
            colorButton.style.cssText = `
                width: 32px;
                height: 32px;
                border: 2px solid ${color === group.color ? '#fff' : 'transparent'};
                border-radius: 6px;
                background: ${color};
                cursor: pointer;
                transition: all 0.2s ease;
            `;
            
            colorButton.addEventListener('click', () => {
                this.setGroupColor(group.id, color);
                colorPicker.remove();
            });

            colorPicker.appendChild(colorButton);
        });

        document.body.appendChild(colorPicker);

        // Close on click outside
        setTimeout(() => {
            const closePicker = (e) => {
                if (!colorPicker.contains(e.target)) {
                    colorPicker.remove();
                    document.removeEventListener('click', closePicker);
                }
            };
            document.addEventListener('click', closePicker);
        }, 100);
    }

    /**
     * Set group color
     * @param {string} groupId - Group ID
     * @param {string} color - New color
     */
    setGroupColor(groupId, color) {
        const group = this.groups.get(groupId);
        if (!group) return;

        group.color = color;
        this.updateGroupVisual(group);
        
        window.errorSystem.showSuccess('Color Changed', `Group "${group.title}" color updated`);
    }

    /**
     * Delete a group
     * @param {string} groupId - Group ID
     */
    deleteGroup(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        if (confirm(`Delete group "${group.title}"? This will not delete the nodes.`)) {
            // Remove group membership from nodes
            const nodes = this.getGroupNodes(groupId);
            nodes.forEach(node => {
                delete node.groupId;
                node.setDirtyCanvas(true);
            });

            // Remove visual element
            const groupElement = document.getElementById(`group-visual-${groupId}`);
            if (groupElement) {
                groupElement.remove();
            }

            // Remove from groups map
            this.groups.delete(groupId);
            this.selectedGroups.delete(groupId);

            window.errorSystem.showSuccess('Group Deleted', `Group "${group.title}" has been deleted`);
        }
    }

    /**
     * Create subflow from group
     * @param {string} groupId - Group ID
     */
    createSubflow(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        // TODO: Implement subflow creation
        window.errorSystem.showInfo('Subflow Creation', 'Subflow functionality will be implemented in a future update');
    }

    /**
     * Export group as JSON
     * @param {string} groupId - Group ID
     */
    exportGroup(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        const nodes = this.getGroupNodes(groupId);
        const exportData = {
            group: { ...group },
            nodes: nodes.map(node => ({
                id: node.id,
                type: node.type,
                pos: [...node.pos],
                size: [...node.size],
                properties: { ...node.properties },
                inputs: node.inputs || [],
                outputs: node.outputs || []
            })),
            connections: this.getGroupConnections(groupId),
            metadata: {
                exported: new Date().toISOString(),
                version: '1.0',
                tool: 'SD Forms'
            }
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `${group.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_group.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        window.errorSystem.showSuccess('Group Exported', `Group "${group.title}" exported successfully`);
    }

    /**
     * Get connections within a group
     * @param {string} groupId - Group ID
     * @returns {Array} Array of connections
     */
    getGroupConnections(groupId) {
        const nodes = this.getGroupNodes(groupId);
        const nodeIds = new Set(nodes.map(n => n.id));
        const connections = [];

        if (this.app.graph && this.app.graph.links) {
            Object.values(this.app.graph.links).forEach(link => {
                if (nodeIds.has(link.origin_id) || nodeIds.has(link.target_id)) {
                    connections.push({
                        id: link.id,
                        from_node: link.origin_id,
                        from_slot: link.origin_slot,
                        to_node: link.target_id,
                        to_slot: link.target_slot,
                        type: link.type
                    });
                }
            });
        }

        return connections;
    }

    /**
     * Copy group to clipboard
     * @param {string} groupId - Group ID
     */
    copyGroup(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        const nodes = this.getGroupNodes(groupId);
        const groupData = {
            group: { ...group },
            nodes: nodes.map(node => ({ ...node }))
        };

        // Store in clipboard (simplified)
        this.clipboardGroup = groupData;
        window.errorSystem.showSuccess('Group Copied', `Group "${group.title}" copied to clipboard`);
    }

    /**
     * Duplicate group
     * @param {string} groupId - Group ID
     */
    duplicateGroup(groupId) {
        const group = this.groups.get(groupId);
        if (!group) return;

        // TODO: Implement group duplication
        window.errorSystem.showInfo('Duplicate Group', 'Group duplication will be implemented in a future update');
    }

    /**
     * Get random color for new groups
     * @returns {string} Random color hex code
     */
    getRandomGroupColor() {
        const colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
            '#9b59b6', '#1abc9c', '#34495e', '#e67e22'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    /**
     * Setup CSS styles for groups
     */
    setupGroupStyles() {
        if (document.getElementById('node-groups-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'node-groups-styles';
        styles.textContent = `
            .node-group-visual {
                pointer-events: none;
                transition: all 0.3s ease;
            }

            .node-group-visual.selected {
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.5);
            }

            .group-header {
                user-select: none;
                transition: all 0.2s ease;
            }

            .group-header:hover {
                transform: scale(1.05);
            }

            .group-toggle {
                transition: transform 0.2s ease;
            }

            .group-toggle:hover {
                transform: scale(1.2);
            }

            .group-context-menu {
                min-width: 180px;
            }

            .group-context-menu .menu-item.danger {
                color: var(--accent-danger);
            }

            .group-context-menu .menu-item.danger:hover {
                background: rgba(244, 67, 54, 0.1);
            }

            .color-picker-popup {
                animation: popupShow 0.2s ease;
            }

            @keyframes popupShow {
                from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
            }
        `;
        document.head.appendChild(styles);
    }

    /**
     * Register keyboard shortcuts for grouping
     */
    registerGroupShortcuts() {
        // These will be integrated with the main keyboard shortcuts system
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'g' && !e.shiftKey) {
                e.preventDefault();
                this.createGroup();
            } else if (e.ctrlKey && e.shiftKey && e.key === 'G') {
                e.preventDefault();
                this.ungroupSelected();
            }
        });
    }

    /**
     * Ungroup selected groups
     */
    ungroupSelected() {
        if (this.selectedGroups.size === 0) {
            window.errorSystem.showWarning('No Selection', 'Please select groups to ungroup');
            return;
        }

        let ungroupedCount = 0;
        this.selectedGroups.forEach(groupId => {
            const group = this.groups.get(groupId);
            if (group) {
                this.deleteGroup(groupId);
                ungroupedCount++;
            }
        });

        if (ungroupedCount > 0) {
            window.errorSystem.showSuccess('Groups Ungrouped', `${ungroupedCount} group(s) ungrouped`);
        }
    }

    /**
     * Update all group visuals (called when canvas moves/zooms)
     */
    updateAllGroupVisuals() {
        this.groups.forEach(group => {
            this.updateGroupVisual(group);
        });
    }

    /**
     * Clean up removed nodes from groups
     */
    cleanupGroups() {
        this.groups.forEach((group, groupId) => {
            const validNodes = group.nodes.filter(nodeId => 
                this.app.graph._nodes.some(node => node.id === nodeId)
            );
            
            if (validNodes.length !== group.nodes.length) {
                if (validNodes.length === 0) {
                    // Delete empty group
                    this.deleteGroup(groupId);
                } else {
                    // Update group with valid nodes
                    group.nodes = validNodes;
                    this.updateGroupVisual(group);
                }
            }
        });
    }

    /**
     * Get all groups
     * @returns {Map} Map of all groups
     */
    getAllGroups() {
        return new Map(this.groups);
    }

    /**
     * Get group by ID
     * @param {string} groupId - Group ID
     * @returns {Object|null} Group object or null
     */
    getGroup(groupId) {
        return this.groups.get(groupId) || null;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeGroupManager;
}