from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yaml
import re
import docx
import os
from werkzeug.utils import secure_filename
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from spellchecker import SpellChecker


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class BasicSpellChecker:
    def __init__(self):
        # ...existing code...
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.domain_configs = {}
        self.load_domain_configs()
        self.spell = SpellChecker()
        self.domain_models = {}
    
    def load_domain_configs(self):
        with open('domain_config.yaml', 'r') as f:
            self.domain_configs = yaml.safe_load(f)

    def load_domain_models(self):
        """Load domain-specific models based on config"""
        self.domain_models = {
            'medical': ('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', None),
            'legal': ('nlpaueb/legal-bert-base-uncased', None)
        }
        # Lazy loading of models - will be loaded when needed

    def is_word_correct(self, word):
        word_lower = word.lower()
        # Check if it's in basic dictionary
        if word_lower in self.spell:
            return True
        # Check domain terms
        for domain_config in self.domain_configs.values():
            terms = domain_config.get('terms', [])
            if word_lower in [term.lower() for term in terms]:
                return True
        return False
    
    def get_suggestions(self, word, domain):
        suggestions = []
        word_lower = word.lower()
        
        # First check domain-specific terms
        domain_terms = self.domain_configs.get(domain, {}).get('terms', [])
        # for term in domain_terms:
        #     if self.levenshtein_distance(word_lower, term.lower()) <= 2:
        #         distance = self.levenshtein_distance(word_lower, term.lower())
        #         score = max(0.1, 1.0 - (distance / max(len(word), len(term))))
        #         suggestions.append((term, score))
        
        # Use neural model for suggestions
        context = self.domain_configs.get(domain, {}).get('context', '')
        masked_text = f"{context} {self.tokenizer.mask_token} {context}"
        
        # Get domain-specific model if available
        if domain in self.domain_models:
            model_name, model = self.domain_models[domain]
            if model is None:  # Lazy loading
                self.domain_models[domain] = (
                    model_name,
                    AutoModelForMaskedLM.from_pretrained(model_name)
                )
                model = self.domain_models[domain][1]
        else:
            model = self.model  # Use default BERT model
            
        # Get predictions from model
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits[0, inputs['input_ids'][0] == self.tokenizer.mask_token_id]
            
        # print("predictions shape:", predictions.shape)
        # print("predictions:", predictions)
        predictions = predictions.squeeze(0)
        # Get top 5 predictions
        top_k = 5
        probs, indices = torch.topk(torch.softmax(predictions, dim=0), top_k)
        
        
        # Generate suggestions
        for prob, index in zip(probs, indices):
            token_id = index.item()
            try:
                suggestion = self.tokenizer.decode([token_id]).strip()
                if suggestion and self.levenshtein_distance(word_lower, suggestion.lower()) <= 3:
                    suggestions.append((suggestion, float(prob)))
            except Exception:
                continue

        # Sort all suggestions by score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5] if suggestions else [(word, 0.1)]


    def check_spelling(self, text, domain="general", model_type="basic"):
        words = re.findall(r'\b\w+\b', text)
        errors = []
        
        for i, word in enumerate(words):
            if not self.is_word_correct(word):  # Skip very short words
                suggestions = self.get_suggestions(word, domain)
                
                # Determine error type
                domain_terms = self.domain_configs.get(domain, {}).get('terms', [])
                error_type = "domain" if any(sugg in domain_terms for sugg, _ in suggestions[:3]) else "general"
                
                errors.append({
                    'word': word,
                    'position': i,
                    'suggestions': suggestions,
                    'type': error_type,
                    'confidence': suggestions[0][1] if suggestions else 0.5
                })
        
        return errors
    
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

spellchecker = BasicSpellChecker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spelling():
    data = request.json
    text = data.get('text', '')
    domain = data.get('domain', 'general')
    model_type = data.get('model', 'basic')
    
    errors = spellchecker.check_spelling(text, domain, model_type)
    
    return jsonify({
        'errors': errors,
        'text': text,
        'domain': domain,
        'model': model_type
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif filename.endswith('.docx'):
            doc = docx.Document(filepath)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        os.remove(filepath)
        return jsonify({'text': text})
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400

@app.route('/domains')
def get_domains():
    return jsonify(list(spellchecker.domain_configs.keys()))

if __name__ == '__main__':
    print("Starting Neural Spellchecker webapp...")
    print("Visit: http://localhost:5000")
    app.run(debug=True)