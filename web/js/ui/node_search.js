/**
 * Node Search Overlay for SD Forms
 * ComfyUI-style search interface with fuzzy finding
 */

class NodeSearchOverlay {
    constructor(app) {
        this.app = app;
        this.overlay = null;
        this.searchInput = null;
        this.resultsList = null;
        this.selectedIndex = 0;
        this.currentResults = [];
        this.isVisible = false;
        this.lastSearchQuery = '';
        this.searchPosition = { x: 0, y: 0 };
        
        this.createOverlay();
        this.setupEventListeners();
    }

    /**
     * Create the search overlay DOM structure
     */
    createOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.id = 'node-search-overlay';
        this.overlay.className = 'node-search-overlay hidden';
        
        this.overlay.innerHTML = `
            <div class="search-container">
                <div class="search-header">
                    <div class="search-input-container">
                        <span class="search-icon">🔍</span>
                        <input 
                            type="text" 
                            id="node-search-input" 
                            class="search-input" 
                            placeholder="Search nodes... (fuzzy search supported)"
                            autocomplete="off"
                            spellcheck="false"
                        >
                        <button id="search-close" class="search-close">✕</button>
                    </div>
                    <div class="search-stats">
                        <span id="search-count">0 results</span>
                        <span class="search-help">↑↓ navigate • Enter select • Esc close</span>
                    </div>
                </div>
                <div class="search-results" id="search-results">
                    <div class="search-placeholder">
                        <div class="placeholder-icon">📦</div>
                        <div class="placeholder-text">Start typing to search for nodes...</div>
                        <div class="placeholder-tips">
                            <div class="tip">💡 <strong>Tips:</strong></div>
                            <div class="tip">• Type partial names (e.g., "mod" for "Model")</div>
                            <div class="tip">• Use camelCase shortcuts (e.g., "MC" for "ModelComponent")</div>
                            <div class="tip">• Search by category or property names</div>
                        </div>
                    </div>
                </div>
                <div class="search-footer">
                    <div class="recent-searches" id="recent-searches"></div>
                </div>
            </div>
        `;

        // Add styles
        if (!document.getElementById('node-search-styles')) {
            const styles = document.createElement('style');
            styles.id = 'node-search-styles';
            styles.textContent = `
                .node-search-overlay {
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
                    opacity: 0;
                    transition: opacity 0.2s ease;
                }

                .node-search-overlay:not(.hidden) {
                    opacity: 1;
                }

                .node-search-overlay.hidden {
                    display: none;
                }

                .search-container {
                    width: 600px;
                    max-width: 90vw;
                    max-height: 80vh;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-primary);
                    border-radius: 12px;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                    transform: scale(0.9);
                    transition: transform 0.2s ease;
                }

                .node-search-overlay:not(.hidden) .search-container {
                    transform: scale(1);
                }

                .search-header {
                    padding: 16px;
                    border-bottom: 1px solid var(--border-primary);
                    background: var(--bg-tertiary);
                }

                .search-input-container {
                    display: flex;
                    align-items: center;
                    background: var(--bg-primary);
                    border: 2px solid var(--border-primary);
                    border-radius: 8px;
                    padding: 0 12px;
                    transition: border-color 0.2s ease;
                }

                .search-input-container:focus-within {
                    border-color: var(--accent-primary);
                    box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.1);
                }

                .search-icon {
                    font-size: 16px;
                    margin-right: 8px;
                    color: var(--text-muted);
                }

                .search-input {
                    flex: 1;
                    padding: 12px 0;
                    background: transparent;
                    border: none;
                    outline: none;
                    color: var(--text-primary);
                    font-size: 16px;
                    font-family: inherit;
                }

                .search-input::placeholder {
                    color: var(--text-muted);
                }

                .search-close {
                    background: none;
                    border: none;
                    color: var(--text-muted);
                    cursor: pointer;
                    padding: 4px;
                    font-size: 14px;
                    border-radius: 4px;
                    transition: all 0.2s ease;
                }

                .search-close:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                .search-stats {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-top: 8px;
                    font-size: 12px;
                    color: var(--text-muted);
                }

                .search-help {
                    font-family: 'Courier New', monospace;
                }

                .search-results {
                    flex: 1;
                    overflow-y: auto;
                    max-height: 400px;
                }

                .search-placeholder {
                    padding: 40px 20px;
                    text-align: center;
                    color: var(--text-muted);
                }

                .placeholder-icon {
                    font-size: 48px;
                    margin-bottom: 16px;
                    opacity: 0.5;
                }

                .placeholder-text {
                    font-size: 16px;
                    margin-bottom: 24px;
                }

                .placeholder-tips {
                    text-align: left;
                    max-width: 300px;
                    margin: 0 auto;
                }

                .tip {
                    margin-bottom: 8px;
                    font-size: 13px;
                    line-height: 1.4;
                }

                .search-result-item {
                    display: flex;
                    align-items: center;
                    padding: 12px 16px;
                    border-bottom: 1px solid var(--border-primary);
                    cursor: pointer;
                    transition: all 0.15s ease;
                }

                .search-result-item:hover,
                .search-result-item.selected {
                    background: var(--bg-hover);
                }

                .search-result-item.selected {
                    border-left: 3px solid var(--accent-primary);
                }

                .result-icon {
                    font-size: 20px;
                    margin-right: 12px;
                    min-width: 32px;
                    text-align: center;
                }

                .result-content {
                    flex: 1;
                    min-width: 0;
                }

                .result-name {
                    font-size: 14px;
                    font-weight: 500;
                    color: var(--text-primary);
                    margin-bottom: 2px;
                    line-height: 1.3;
                }

                .result-details {
                    font-size: 12px;
                    color: var(--text-muted);
                    display: flex;
                    gap: 8px;
                    align-items: center;
                }

                .result-type {
                    font-family: 'Courier New', monospace;
                    background: var(--bg-tertiary);
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                }

                .result-category {
                    color: var(--accent-primary);
                }

                .result-score {
                    margin-left: auto;
                    font-size: 10px;
                    color: var(--text-muted);
                    font-family: 'Courier New', monospace;
                }

                .search-highlight {
                    background: var(--accent-primary);
                    color: white;
                    padding: 1px 2px;
                    border-radius: 2px;
                    font-weight: bold;
                }

                .search-footer {
                    border-top: 1px solid var(--border-primary);
                    padding: 8px 16px;
                    background: var(--bg-tertiary);
                }

                .recent-searches {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                }

                .recent-search-item {
                    background: var(--bg-primary);
                    border: 1px solid var(--border-primary);
                    border-radius: 12px;
                    padding: 4px 8px;
                    font-size: 11px;
                    color: var(--text-secondary);
                    cursor: pointer;
                    transition: all 0.15s ease;
                }

                .recent-search-item:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                /* Scrollbar styling for search results */
                .search-results::-webkit-scrollbar {
                    width: 6px;
                }

                .search-results::-webkit-scrollbar-track {
                    background: transparent;
                }

                .search-results::-webkit-scrollbar-thumb {
                    background: var(--border-secondary);
                    border-radius: 3px;
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(this.overlay);
        
        // Get references to DOM elements
        this.searchInput = document.getElementById('node-search-input');
        this.resultsList = document.getElementById('search-results');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Search input events
        this.searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });

        this.searchInput.addEventListener('keydown', (e) => {
            this.handleKeydown(e);
        });

        // Close button
        document.getElementById('search-close').addEventListener('click', () => {
            this.hide();
        });

        // Overlay click to close
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hide();
            }
        });

        // Prevent event bubbling inside search container
        this.overlay.querySelector('.search-container').addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    /**
     * Handle search input changes
     * @param {string} query - Search query
     */
    handleSearchInput(query) {
        this.lastSearchQuery = query;
        
        if (!query.trim()) {
            this.showPlaceholder();
            return;
        }

        // Perform fuzzy search
        const components = Array.from(this.app.components.values());
        const results = window.fuzzySearch.searchComponents(query, components, 20);
        
        this.currentResults = results;
        this.selectedIndex = 0;
        this.renderResults(results, query);
        this.updateSearchStats(results.length);
    }

    /**
     * Handle keyboard navigation
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeydown(e) {
        switch (e.key) {
            case 'Escape':
                e.preventDefault();
                this.hide();
                break;
                
            case 'ArrowDown':
                e.preventDefault();
                this.moveSelection(1);
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                this.moveSelection(-1);
                break;
                
            case 'Enter':
                e.preventDefault();
                this.selectCurrentItem();
                break;
                
            case 'Tab':
                e.preventDefault();
                if (this.currentResults.length > 0) {
                    this.selectCurrentItem();
                }
                break;
        }
    }

    /**
     * Move selection up or down
     * @param {number} direction - 1 for down, -1 for up
     */
    moveSelection(direction) {
        if (this.currentResults.length === 0) return;
        
        this.selectedIndex += direction;
        
        if (this.selectedIndex < 0) {
            this.selectedIndex = this.currentResults.length - 1;
        } else if (this.selectedIndex >= this.currentResults.length) {
            this.selectedIndex = 0;
        }
        
        this.updateSelectionVisual();
    }

    /**
     * Update visual selection highlighting
     */
    updateSelectionVisual() {
        const items = this.resultsList.querySelectorAll('.search-result-item');
        items.forEach((item, index) => {
            item.classList.toggle('selected', index === this.selectedIndex);
        });
        
        // Scroll selected item into view
        const selectedItem = items[this.selectedIndex];
        if (selectedItem) {
            selectedItem.scrollIntoView({
                block: 'nearest',
                behavior: 'smooth'
            });
        }
    }

    /**
     * Select and add the currently highlighted item
     */
    selectCurrentItem() {
        if (this.currentResults.length > 0 && this.selectedIndex >= 0) {
            const selectedResult = this.currentResults[this.selectedIndex];
            if (selectedResult) {
                this.addNodeToCanvas(selectedResult.component);
                window.fuzzySearch.addToHistory(this.lastSearchQuery);
                this.hide();
            }
        }
    }

    /**
     * Add node to canvas at search position
     * @param {Object} component - Component to add
     */
    addNodeToCanvas(component) {
        // Use the app's existing method to add components
        this.app.addComponentToCanvas(component.type, this.searchPosition);
    }

    /**
     * Render search results
     * @param {Array} results - Search results with scores
     * @param {string} query - Original search query
     */
    renderResults(results, query) {
        if (results.length === 0) {
            this.showNoResults(query);
            return;
        }

        const html = results.map((result, index) => {
            const { component, score, matchType } = result;
            const highlightedName = window.fuzzySearch.highlightMatches(component.display_name, query);
            
            return `
                <div class="search-result-item ${index === 0 ? 'selected' : ''}" data-index="${index}">
                    <div class="result-icon">${component.icon || '📦'}</div>
                    <div class="result-content">
                        <div class="result-name">${highlightedName}</div>
                        <div class="result-details">
                            <span class="result-type">${component.type}</span>
                            <span class="result-category">${component.category}</span>
                            <span class="result-match-type">via ${matchType}</span>
                        </div>
                    </div>
                    <div class="result-score">${Math.round(score * 100)}%</div>
                </div>
            `;
        }).join('');

        this.resultsList.innerHTML = html;

        // Add click listeners to result items
        this.resultsList.querySelectorAll('.search-result-item').forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectedIndex = index;
                this.selectCurrentItem();
            });
            
            item.addEventListener('mouseenter', () => {
                this.selectedIndex = index;
                this.updateSelectionVisual();
            });
        });
    }

    /**
     * Show placeholder content
     */
    showPlaceholder() {
        this.currentResults = [];
        this.selectedIndex = 0;
        
        const recentSearches = window.fuzzySearch.getHistory().slice(0, 5);
        let recentHtml = '';
        
        if (recentSearches.length > 0) {
            recentHtml = recentSearches.map(query => 
                `<span class="recent-search-item" data-query="${query}">${query}</span>`
            ).join('');
        }

        this.resultsList.innerHTML = `
            <div class="search-placeholder">
                <div class="placeholder-icon">📦</div>
                <div class="placeholder-text">Start typing to search for nodes...</div>
                <div class="placeholder-tips">
                    <div class="tip">💡 <strong>Tips:</strong></div>
                    <div class="tip">• Type partial names (e.g., "mod" for "Model")</div>
                    <div class="tip">• Use camelCase shortcuts (e.g., "MC" for "ModelComponent")</div>
                    <div class="tip">• Search by category or property names</div>
                </div>
            </div>
        `;

        document.getElementById('recent-searches').innerHTML = recentHtml;
        
        // Add click listeners to recent searches
        document.querySelectorAll('.recent-search-item').forEach(item => {
            item.addEventListener('click', () => {
                const query = item.dataset.query;
                this.searchInput.value = query;
                this.handleSearchInput(query);
                this.searchInput.focus();
            });
        });

        this.updateSearchStats(0);
    }

    /**
     * Show no results message
     * @param {string} query - Search query that returned no results
     */
    showNoResults(query) {
        this.resultsList.innerHTML = `
            <div class="search-placeholder">
                <div class="placeholder-icon">🔍</div>
                <div class="placeholder-text">No nodes found for "${query}"</div>
                <div class="placeholder-tips">
                    <div class="tip">💡 Try:</div>
                    <div class="tip">• Different spelling or shorter terms</div>
                    <div class="tip">• Searching by category (e.g., "core", "models")</div>
                    <div class="tip">• Using abbreviations (e.g., "SD" for "Stable Diffusion")</div>
                </div>
            </div>
        `;
    }

    /**
     * Update search statistics display
     * @param {number} count - Number of results
     */
    updateSearchStats(count) {
        const statsElement = document.getElementById('search-count');
        if (count === 0) {
            statsElement.textContent = 'No results';
        } else if (count === 1) {
            statsElement.textContent = '1 result';
        } else {
            statsElement.textContent = `${count} results`;
        }
    }

    /**
     * Show search overlay
     * @param {Object} position - Position to place new nodes {x, y}
     */
    show(position = { x: 100, y: 100 }) {
        this.searchPosition = position;
        this.isVisible = true;
        this.overlay.classList.remove('hidden');
        
        // Focus search input after animation
        setTimeout(() => {
            this.searchInput.focus();
            this.searchInput.select();
        }, 100);
        
        // Show placeholder or restore last search
        if (this.lastSearchQuery) {
            this.searchInput.value = this.lastSearchQuery;
            this.handleSearchInput(this.lastSearchQuery);
        } else {
            this.showPlaceholder();
        }
    }

    /**
     * Hide search overlay
     */
    hide() {
        this.isVisible = false;
        this.overlay.classList.add('hidden');
        this.searchInput.blur();
        
        // Clear search results but keep query
        this.currentResults = [];
        this.selectedIndex = 0;
    }

    /**
     * Toggle search overlay visibility
     * @param {Object} position - Position for new nodes
     */
    toggle(position) {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show(position);
        }
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeSearchOverlay;
}