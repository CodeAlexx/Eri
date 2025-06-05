/**
 * Model Browser UI Component
 * Handles model selection with categorization, search, and filtering
 */

class ModelBrowser {
    constructor() {
        this.models = [];
        this.filteredModels = [];
        this.selectedModel = null;
        this.activeNode = null;
        this.searchTerm = '';
        this.selectedCategory = 'all';
        
        this.modal = null;
        this.searchInput = null;
        this.categoryFilter = null;
        this.modelList = null;
        
        this.init();
    }

    init() {
        this.createModal();
        this.setupEventListeners();
    }

    createModal() {
        // Create modal backdrop
        this.modal = document.createElement('div');
        this.modal.className = 'model-browser-modal';
        this.modal.style.cssText = `
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            backdrop-filter: blur(2px);
        `;

        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'model-browser-content';
        modalContent.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            max-width: 800px;
            height: 70%;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        `;

        // Header
        const header = document.createElement('div');
        header.className = 'model-browser-header';
        header.style.cssText = `
            padding: 15px 20px;
            border-bottom: 1px solid #444;
            background: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;

        const title = document.createElement('h3');
        title.textContent = 'Select Model';
        title.style.cssText = 'margin: 0; color: #fff; font-size: 18px;';

        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        closeBtn.onclick = () => this.hide();

        header.appendChild(title);
        header.appendChild(closeBtn);

        // Controls
        const controls = document.createElement('div');
        controls.className = 'model-browser-controls';
        controls.style.cssText = `
            padding: 15px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            gap: 15px;
            align-items: center;
            background: #2a2a2a;
        `;

        // Search input
        this.searchInput = document.createElement('input');
        this.searchInput.type = 'text';
        this.searchInput.placeholder = 'Search models...';
        this.searchInput.style.cssText = `
            flex: 1;
            padding: 8px 12px;
            background: #1a1a1a;
            border: 1px solid #555;
            border-radius: 4px;
            color: #fff;
            font-size: 14px;
        `;

        // Category filter
        this.categoryFilter = document.createElement('select');
        this.categoryFilter.style.cssText = `
            padding: 8px 12px;
            background: #1a1a1a;
            border: 1px solid #555;
            border-radius: 4px;
            color: #fff;
            font-size: 14px;
            min-width: 120px;
        `;

        const categories = [
            { value: 'all', label: 'All Models' },
            { value: 'flux', label: 'Flux' },
            { value: 'sdxl', label: 'SDXL' },
            { value: 'sd15', label: 'SD 1.5' },
            { value: 'lora', label: 'LoRAs' }
        ];

        categories.forEach(cat => {
            const option = document.createElement('option');
            option.value = cat.value;
            option.textContent = cat.label;
            this.categoryFilter.appendChild(option);
        });

        controls.appendChild(this.searchInput);
        controls.appendChild(this.categoryFilter);

        // Model list
        this.modelList = document.createElement('div');
        this.modelList.className = 'model-browser-list';
        this.modelList.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            background: #1a1a1a;
        `;

        // Assemble modal
        modalContent.appendChild(header);
        modalContent.appendChild(controls);
        modalContent.appendChild(this.modelList);
        this.modal.appendChild(modalContent);

        document.body.appendChild(this.modal);
    }

    setupEventListeners() {
        // Search input
        this.searchInput.addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.filterModels();
        });

        // Category filter
        this.categoryFilter.addEventListener('change', (e) => {
            this.selectedCategory = e.target.value;
            this.filterModels();
        });

        // Close modal on backdrop click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hide();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.modal.style.display === 'block') {
                if (e.key === 'Escape') {
                    this.hide();
                }
            }
        });
    }

    async fetchModels() {
        try {
            const response = await fetch('/api/models');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.models = await response.json();
            this.filterModels();
        } catch (error) {
            console.error('Error fetching models:', error);
            this.showError('Failed to load models. Please check your connection.');
        }
    }

    filterModels() {
        this.filteredModels = this.models.filter(model => {
            // Category filter
            if (this.selectedCategory !== 'all') {
                if (this.selectedCategory !== model.category) {
                    return false;
                }
            }

            // Search filter
            if (this.searchTerm) {
                const searchableText = `${model.name} ${model.path} ${model.category}`.toLowerCase();
                if (!searchableText.includes(this.searchTerm)) {
                    return false;
                }
            }

            return true;
        });

        this.renderModelList();
    }

    renderModelList() {
        this.modelList.innerHTML = '';

        if (this.filteredModels.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.style.cssText = `
                text-align: center;
                color: #888;
                padding: 40px 20px;
                font-size: 16px;
            `;
            emptyMessage.textContent = 'No models found matching your criteria.';
            this.modelList.appendChild(emptyMessage);
            return;
        }

        // Group models by category
        const groupedModels = this.groupModelsByCategory(this.filteredModels);

        Object.entries(groupedModels).forEach(([category, models]) => {
            if (models.length === 0) return;

            // Category header (only show if multiple categories)
            if (this.selectedCategory === 'all' && Object.keys(groupedModels).length > 1) {
                const categoryHeader = document.createElement('div');
                categoryHeader.className = 'category-header';
                categoryHeader.style.cssText = `
                    padding: 10px 5px 5px 5px;
                    color: #ccc;
                    font-weight: bold;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                `;
                categoryHeader.textContent = this.getCategoryDisplayName(category);
                this.modelList.appendChild(categoryHeader);
            }

            // Model items
            models.forEach(model => {
                const modelItem = this.createModelItem(model);
                this.modelList.appendChild(modelItem);
            });
        });
    }

    createModelItem(model) {
        const item = document.createElement('div');
        item.className = 'model-item';
        item.style.cssText = `
            padding: 12px 15px;
            margin: 2px 0;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;

        // Hover effects
        item.addEventListener('mouseenter', () => {
            item.style.background = '#333';
            item.style.borderColor = '#555';
        });

        item.addEventListener('mouseleave', () => {
            item.style.background = '#2a2a2a';
            item.style.borderColor = '#444';
        });

        // Model name
        const name = document.createElement('div');
        name.className = 'model-name';
        name.style.cssText = `
            color: #fff;
            font-weight: 500;
            font-size: 14px;
        `;
        name.textContent = model.name;

        // Model path
        const path = document.createElement('div');
        path.className = 'model-path';
        path.style.cssText = `
            color: #888;
            font-size: 12px;
            font-family: monospace;
        `;
        path.textContent = model.path;

        // Model info
        const info = document.createElement('div');
        info.className = 'model-info';
        info.style.cssText = `
            display: flex;
            gap: 10px;
            font-size: 11px;
            color: #aaa;
        `;

        const category = document.createElement('span');
        category.style.cssText = `
            background: #444;
            padding: 2px 6px;
            border-radius: 3px;
            text-transform: uppercase;
            font-weight: bold;
        `;
        category.textContent = model.category;

        if (model.size) {
            const size = document.createElement('span');
            size.textContent = this.formatFileSize(model.size);
            info.appendChild(size);
        }

        info.appendChild(category);

        item.appendChild(name);
        item.appendChild(path);
        item.appendChild(info);

        // Click handlers
        item.addEventListener('click', () => this.selectModel(model));
        item.addEventListener('dblclick', () => this.applyModel(model));

        return item;
    }

    groupModelsByCategory(models) {
        const groups = {
            flux: [],
            sdxl: [],
            sd15: [],
            lora: [],
            other: []
        };

        models.forEach(model => {
            const category = model.category || 'other';
            if (groups[category]) {
                groups[category].push(model);
            } else {
                groups.other.push(model);
            }
        });

        return groups;
    }

    getCategoryDisplayName(category) {
        const names = {
            flux: 'Flux Models',
            sdxl: 'Stable Diffusion XL',
            sd15: 'Stable Diffusion 1.5',
            lora: 'LoRA Models',
            other: 'Other Models'
        };
        return names[category] || category;
    }

    formatFileSize(bytes) {
        if (!bytes) return '';
        
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        const size = (bytes / Math.pow(1024, i)).toFixed(1);
        return `${size} ${sizes[i]}`;
    }

    selectModel(model) {
        // Remove previous selection
        this.modelList.querySelectorAll('.model-item').forEach(item => {
            item.style.background = '#2a2a2a';
            item.style.borderColor = '#444';
        });

        // Highlight selected
        const selectedItem = Array.from(this.modelList.querySelectorAll('.model-item'))
            .find(item => item.querySelector('.model-name').textContent === model.name);
        
        if (selectedItem) {
            selectedItem.style.background = '#0066cc';
            selectedItem.style.borderColor = '#0088ff';
        }

        this.selectedModel = model;
    }

    applyModel(model) {
        if (this.activeNode && model) {
            // Set the model in the active node
            this.activeNode.setProperty('model_path', model.path);
            
            // Trigger any necessary updates
            if (this.activeNode.onPropertyChanged) {
                this.activeNode.onPropertyChanged('model_path', model.path);
            }

            // Close the modal
            this.hide();
        }
    }

    show(node = null) {
        this.activeNode = node;
        this.modal.style.display = 'block';
        
        // Focus search input
        setTimeout(() => {
            this.searchInput.focus();
        }, 100);
        
        // Fetch models if not already loaded
        if (this.models.length === 0) {
            this.fetchModels();
        }
    }

    hide() {
        this.modal.style.display = 'none';
        this.selectedModel = null;
        this.activeNode = null;
        
        // Reset filters
        this.searchInput.value = '';
        this.categoryFilter.value = 'all';
        this.searchTerm = '';
        this.selectedCategory = 'all';
        
        // Re-filter to show all models
        this.filterModels();
    }

    showError(message) {
        this.modelList.innerHTML = '';
        
        const errorMessage = document.createElement('div');
        errorMessage.style.cssText = `
            text-align: center;
            color: #ff6b6b;
            padding: 40px 20px;
            font-size: 16px;
        `;
        errorMessage.textContent = message;
        
        this.modelList.appendChild(errorMessage);
    }
}

// Create global instance
window.modelBrowser = new ModelBrowser();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelBrowser;
}