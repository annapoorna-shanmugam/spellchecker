/**
 * Client-side JavaScript for Neural Spellchecker Web Interface
 * 
 * This script handles the frontend functionality of the Neural Spellchecker application,
 * including file uploads, spell checking requests, and results display. It provides a
 * rich interactive interface for users to check spelling in different domains using
 * various neural models.
 * 
 * Features:
 * - File upload handling (.txt, .docx)
 * - Real-time spell checking
 * - Progress indication
 * - Domain-specific error highlighting
 * - Confidence score display

 */

class SpellChecker {
    
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.currentErrors = [];
    }

    initializeElements() {
        this.textInput = document.getElementById('text-input');
        this.domainSelect = document.getElementById('domain-select');
        this.modelSelect = document.getElementById('model-select');
        this.fileUpload = document.getElementById('file-upload');
        this.uploadBtn = document.getElementById('upload-btn');
        this.checkBtn = document.getElementById('check-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.progressContainer = document.getElementById('progress-container');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.resultsSection = document.getElementById('results-section');
        this.generalCorrections = document.getElementById('general-corrections');
        this.domainCorrections = document.getElementById('domain-corrections');
        this.highlightedText = document.getElementById('highlighted-text');
    }

    attachEventListeners() {
        this.checkBtn.addEventListener('click', () => this.checkSpelling());
        this.clearBtn.addEventListener('click', () => this.clearText());
        this.uploadBtn.addEventListener('click', () => this.uploadFile());
        this.fileUpload.addEventListener('change', () => this.handleFileSelect());
        
        this.textInput.addEventListener('input', () => {
            if (this.textInput.value.trim()) {
                this.checkBtn.disabled = false;
            } else {
                this.checkBtn.disabled = true;
            }
        });
    }

    /**
     * Handle file selection event and update UI accordingly
     * @private
     */
    handleFileSelect() {
        const file = this.fileUpload.files[0];
        if (file) {
            this.uploadBtn.disabled = false;
            this.uploadBtn.textContent = `Upload ${file.name}`;
        } else {
            this.uploadBtn.disabled = true;
            this.uploadBtn.textContent = 'Upload';
        }
    }

    /**
     * Upload and process a file for spell checking
     * Supports .txt and .docx files
     * @private
     * @async
     * @throws {Error} If file upload fails
     */
    async uploadFile() {
        const file = this.fileUpload.files[0];
        if (!file) {
            alert('Please select a file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showProgress('Uploading file...');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.textInput.value = result.text;
                this.hideProgress();
                this.fileUpload.value = '';
                this.uploadBtn.disabled = true;
                this.uploadBtn.textContent = 'Upload';
                this.checkBtn.disabled = false;
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.hideProgress();
            alert(`Upload error: ${error.message}`);
        }
    }

    /**
     * Perform spell checking on the current text
     * Sends text to server for processing using selected model and domain
     * @private
     * @async
     * @throws {Error} If spell check request fails
     */
    async checkSpelling() {
        const text = this.textInput.value.trim();
        if (!text) {
            alert('Please enter some text to check.');
            return;
        }

        const domain = this.domainSelect.value;
        const model = this.modelSelect.value;

        this.showProgress('Analyzing text...');
        this.resultsSection.style.display = 'none';

        try {
            this.updateProgress(30, 'Loading neural model...');
            
            const response = await fetch('/check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    domain: domain,
                    model: model
                })
            });

            this.updateProgress(70, 'Processing corrections...');

            const result = await response.json();
            
            if (response.ok) {
                this.updateProgress(100, 'Complete!');
                setTimeout(() => {
                    this.hideProgress();
                    this.displayResults(result);
                }, 500);
            } else {
                throw new Error(result.error || 'Spell check failed');
            }
        } catch (error) {
            this.hideProgress();
            alert(`Error: ${error.message}`);
        }
    }

    /**
     * Display the progress indicator with initial message
     * @private
     * @param {string} message - Message to display in progress indicator
     */
    showProgress(message) {
        this.progressContainer.style.display = 'block';
        this.progressText.textContent = message;
        this.progressFill.style.width = '0%';
    }

    /**
     * Update the progress indicator
     * @private
     * @param {number} percent - Progress percentage (0-100)
     * @param {string} message - Optional progress message
     */
    updateProgress(percent, message) {
        this.progressFill.style.width = `${percent}%`;
        if (message) {
            this.progressText.textContent = message;
        }
    }

    /**
     * Hide the progress indicator
     * @private
     */
    hideProgress() {
        this.progressContainer.style.display = 'none';
    }

    /**
     * Display spell check results in the UI
     * @private
     * @param {Object} result - Spell check results from server
     * @param {Array} result.errors - Array of spelling errors
     * @param {string} result.text - Original text
     */
    displayResults(result) {
        this.currentErrors = result.errors;
        
        const generalErrs = result.errors.filter(e => e.type === 'general');
        const domainErrs = result.errors.filter(e => e.type === 'domain');

        this.displayCorrections(generalErrs, this.generalCorrections);
        this.displayCorrections(domainErrs, this.domainCorrections);
        this.displayHighlightedText(result.text, result.errors);

        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Display spelling corrections in their respective containers
     * @private
     * @param {Array} errors - Array of spelling errors
     * @param {HTMLElement} container - DOM element to contain the corrections
     */
    displayCorrections(errors, container) {
        container.innerHTML = '';
        
        if (errors.length === 0) {
            container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No errors found in this category.</p>';
            return;
        }

        errors.forEach(error => {
            const item = document.createElement('div');
            item.className = 'correction-item';
            
            const suggestions = error.suggestions.slice(0, 3);
            const suggestionsHtml = suggestions.map(([word, confidence]) => 
                `<span class="suggestion">${word}</span> <span class="confidence">[${Math.round(confidence * 100)}%]</span>`
            ).join(', ');

            item.innerHTML = `
                <div>
                    <span class="error-word">${error.word}</span> → ${suggestionsHtml}
                </div>
            `;
            
            container.appendChild(item);
        });
    }

    /**
     * Display text with highlighted spelling errors
     * @private
     * @param {string} text - Original text
     * @param {Array} errors - Array of spelling errors
     */
    displayHighlightedText(text, errors) {
        let highlightedText = text;
        const words = text.split(/(\s+)/);
        let wordIndex = 0;
        
        const processedWords = words.map(word => {
            if (word.trim() === '') {
                return word;
            }
            
            const error = errors.find(e => e.position === wordIndex);
            wordIndex++;
            
            if (error) {
                const className = error.type === 'domain' ? 'domain' : 'general';
                const bestSuggestion = error.suggestions[0];
                const title = bestSuggestion ? 
                    `Suggestion: ${bestSuggestion[0]} (${Math.round(bestSuggestion[1] * 100)}% confidence)` : 
                    'No suggestions available';
                
                return `<span class="error-highlight ${className}" title="${title}">${word}</span>`;
            }
            
            return word;
        });

        this.highlightedText.innerHTML = processedWords.join('');
    }

    /**
     * Clear all text and results, reset UI to initial state
     * @private
     */
    clearText() {
        this.textInput.value = '';
        this.resultsSection.style.display = 'none';
        this.hideProgress();
        this.checkBtn.disabled = true;
        this.fileUpload.value = '';
        this.uploadBtn.disabled = true;
        this.uploadBtn.textContent = 'Upload';
    }
}

// Initialize SpellChecker when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    new SpellChecker();
});