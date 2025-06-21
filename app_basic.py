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
import torch.nn.functional as F


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
        self.load_domain_models()
    
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
    
    def get_suggestions(self, masked_text, error_words_suggestion, error_words, domain):
        # Get domain-specific model if available
        if domain in self.domain_models:
            self.tokenizer = AutoTokenizer.from_pretrained(self.domain_models[domain][0])
            self.model = AutoModelForMaskedLM.from_pretrained(self.domain_models[domain][0])
            
        # Get predictions from model
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Find all [MASK] position
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits = self.model(**inputs).logits

        top_k = 3
        for idx, mask_pos in enumerate(mask_token_indices):
            mask_logits = logits[0, mask_pos, :]
            probs = F.softmax(mask_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            word_suggestions = []
            for token_id, prob in zip(top_indices, top_probs):
                predicted_word = self.tokenizer.decode([token_id]).strip()
                word_suggestions.append((predicted_word, float(prob)))

            # Sort suggestions by Levenshtein distance to the original error word, then by probability
            word_suggestions.sort(key=lambda x: (self.levenshtein_distance(error_words[idx].lower(), x[0].lower()), -x[1]))
            print(f"Suggestions for '{error_words[idx]}': {word_suggestions}")
            # Assign suggestions to the corresponding error word
            error_words_suggestion[error_words[idx]] = word_suggestions[0]


    def check_spelling(self, text, domain="general", model_type="basic"):
        words = re.findall(r'\b\w+\b', text)
        errors = []
        masked_text = ""
        error_words = []
        error_words_suggestion = {}
        for i, word in enumerate(words):
            if self.is_word_correct(word): 
                masked_text += word + " "
            else:
                error_words.append(word)
                error_words_suggestion[word] = []
                masked_text += self.tokenizer.mask_token + " "
        self.get_suggestions(masked_text, error_words_suggestion, error_words, domain)

        for i, word in enumerate(words):
            if word in error_words: 
                # Determine error type
                domain_terms = self.domain_configs.get(domain, {}).get('terms', [])
                error_type = "domain" if any(sugg in domain_terms for sugg in error_words_suggestion[word]) else "general"
                
                errors.append({
                    'word': word,
                    'position': i,
                    'suggestions': [list(error_words_suggestion[word])],
                    'type': error_type,
                    'confidence': error_words_suggestion[word][1] if error_words_suggestion[word] else 0.5
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