/**
 * Fuzzy Search Module for SD Forms
 * Provides fast fuzzy finding for components with ranking
 */

class FuzzySearch {
    constructor() {
        this.cache = new Map();
        this.searchHistory = [];
        this.maxHistorySize = 50;
    }

    /**
     * Calculate fuzzy match score between query and target string
     * @param {string} query - Search query
     * @param {string} target - Target string to match against
     * @returns {number} Score (0-1, higher is better match)
     */
    calculateScore(query, target) {
        if (!query || !target) return 0;
        
        const queryLower = query.toLowerCase();
        const targetLower = target.toLowerCase();
        
        // Exact match gets highest score
        if (targetLower === queryLower) return 1.0;
        
        // Starts with query gets high score
        if (targetLower.startsWith(queryLower)) return 0.9;
        
        // Contains query gets medium score
        if (targetLower.includes(queryLower)) return 0.7;
        
        // Fuzzy matching with character sequence
        let score = this.fuzzyMatch(queryLower, targetLower);
        
        // Boost score for camelCase matches
        if (this.camelCaseMatch(queryLower, target)) {
            score += 0.2;
        }
        
        // Boost score for acronym matches
        if (this.acronymMatch(queryLower, target)) {
            score += 0.3;
        }
        
        return Math.min(score, 1.0);
    }

    /**
     * Fuzzy character sequence matching
     * @param {string} query - Search query (lowercase)
     * @param {string} target - Target string (lowercase)
     * @returns {number} Score (0-1)
     */
    fuzzyMatch(query, target) {
        let queryIndex = 0;
        let targetIndex = 0;
        let matches = 0;
        let consecutiveMatches = 0;
        let maxConsecutive = 0;
        
        while (queryIndex < query.length && targetIndex < target.length) {
            if (query[queryIndex] === target[targetIndex]) {
                matches++;
                queryIndex++;
                consecutiveMatches++;
                maxConsecutive = Math.max(maxConsecutive, consecutiveMatches);
            } else {
                consecutiveMatches = 0;
            }
            targetIndex++;
        }
        
        if (queryIndex < query.length) return 0; // Not all query characters found
        
        const matchRatio = matches / query.length;
        const consecutiveBonus = maxConsecutive / query.length * 0.3;
        const lengthPenalty = Math.max(0, (target.length - query.length) / target.length) * 0.1;
        
        return Math.max(0, matchRatio + consecutiveBonus - lengthPenalty);
    }

    /**
     * Check for camelCase matching (e.g., "mc" matches "ModelComponent")
     * @param {string} query - Search query (lowercase)
     * @param {string} target - Target string (original case)
     * @returns {boolean} True if camelCase match found
     */
    camelCaseMatch(query, target) {
        const capitals = target.match(/[A-Z]/g);
        if (!capitals || capitals.length < 2) return false;
        
        const acronym = capitals.join('').toLowerCase();
        return acronym.includes(query) || query.includes(acronym);
    }

    /**
     * Check for acronym matching
     * @param {string} query - Search query (lowercase)
     * @param {string} target - Target string
     * @returns {boolean} True if acronym match found
     */
    acronymMatch(query, target) {
        const words = target.split(/[\s_-]+/);
        if (words.length < 2) return false;
        
        const acronym = words.map(word => word[0]).join('').toLowerCase();
        return acronym.includes(query) || query === acronym;
    }

    /**
     * Search components with fuzzy matching
     * @param {string} query - Search query
     * @param {Array} components - Array of component objects
     * @param {number} limit - Maximum results to return
     * @returns {Array} Sorted array of matching components with scores
     */
    searchComponents(query, components, limit = 20) {
        if (!query.trim()) {
            return components.slice(0, limit).map(comp => ({ component: comp, score: 1.0 }));
        }

        const cacheKey = `${query}:${components.length}`;
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        const results = [];
        
        for (const component of components) {
            const nameScore = this.calculateScore(query, component.display_name);
            const typeScore = this.calculateScore(query, component.type) * 0.8;
            const categoryScore = this.calculateScore(query, component.category) * 0.6;
            
            // Check property names for matches
            let propertyScore = 0;
            if (component.property_definitions) {
                for (const prop of component.property_definitions) {
                    const propScore = this.calculateScore(query, prop.name) * 0.4;
                    propertyScore = Math.max(propertyScore, propScore);
                }
            }
            
            const maxScore = Math.max(nameScore, typeScore, categoryScore, propertyScore);
            
            if (maxScore > 0.1) { // Minimum threshold
                results.push({
                    component: component,
                    score: maxScore,
                    matchType: this.getMatchType(nameScore, typeScore, categoryScore, propertyScore)
                });
            }
        }

        // Sort by score (descending) and then by name (ascending)
        results.sort((a, b) => {
            if (Math.abs(a.score - b.score) < 0.01) {
                return a.component.display_name.localeCompare(b.component.display_name);
            }
            return b.score - a.score;
        });

        const limitedResults = results.slice(0, limit);
        this.cache.set(cacheKey, limitedResults);
        
        // Clean cache if it gets too large
        if (this.cache.size > 100) {
            const keysToDelete = Array.from(this.cache.keys()).slice(0, 50);
            keysToDelete.forEach(key => this.cache.delete(key));
        }

        return limitedResults;
    }

    /**
     * Determine the type of match for highlighting
     * @param {number} nameScore - Name match score
     * @param {number} typeScore - Type match score  
     * @param {number} categoryScore - Category match score
     * @param {number} propertyScore - Property match score
     * @returns {string} Match type
     */
    getMatchType(nameScore, typeScore, categoryScore, propertyScore) {
        const scores = { nameScore, typeScore, categoryScore, propertyScore };
        const maxScore = Math.max(...Object.values(scores));
        
        if (nameScore === maxScore) return 'name';
        if (typeScore === maxScore) return 'type';
        if (categoryScore === maxScore) return 'category';
        if (propertyScore === maxScore) return 'property';
        return 'name';
    }

    /**
     * Add search query to history
     * @param {string} query - Search query
     */
    addToHistory(query) {
        if (!query.trim()) return;
        
        // Remove if already exists
        const index = this.searchHistory.indexOf(query);
        if (index > -1) {
            this.searchHistory.splice(index, 1);
        }
        
        // Add to beginning
        this.searchHistory.unshift(query);
        
        // Limit history size
        if (this.searchHistory.length > this.maxHistorySize) {
            this.searchHistory = this.searchHistory.slice(0, this.maxHistorySize);
        }
    }

    /**
     * Get search history
     * @returns {Array} Array of recent search queries
     */
    getHistory() {
        return [...this.searchHistory];
    }

    /**
     * Clear search cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Highlight matching characters in text
     * @param {string} text - Original text
     * @param {string} query - Search query
     * @returns {string} HTML with highlighted matches
     */
    highlightMatches(text, query) {
        if (!query || !text) return text;
        
        const queryLower = query.toLowerCase();
        const textLower = text.toLowerCase();
        
        // Find all matching character positions
        const matches = [];
        let queryIndex = 0;
        
        for (let i = 0; i < text.length && queryIndex < query.length; i++) {
            if (textLower[i] === queryLower[queryIndex]) {
                matches.push(i);
                queryIndex++;
            }
        }
        
        if (matches.length === 0) return text;
        
        // Build highlighted text
        let result = '';
        let lastIndex = 0;
        
        for (const matchIndex of matches) {
            if (matchIndex > lastIndex) {
                result += text.slice(lastIndex, matchIndex);
            }
            result += `<mark class="search-highlight">${text[matchIndex]}</mark>`;
            lastIndex = matchIndex + 1;
        }
        
        if (lastIndex < text.length) {
            result += text.slice(lastIndex);
        }
        
        return result;
    }
}

// Create global instance
window.fuzzySearch = new FuzzySearch();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FuzzySearch;
}