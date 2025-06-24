"""
Neural Spell Checker Web Application

This Flask application provides a web interface for spell checking using neural models.
It supports both BERT and LSTM-based spell checking models, and can handle different
domains like general, medical, and legal text. The application can process both direct
text input and file uploads (txt and docx formats).

Dependencies:
    - Flask: Web framework
    - flask-cors: Cross-Origin Resource Sharing support
    - python-docx: For processing Word documents
    - Custom modules: BertSpellChecker, LSTMSpellChecker
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import docx
import os
from werkzeug.utils import secure_filename

from BertSpellChecker import BertSpellChecker
from LSTMSpellChecker import LSTMSpellChecker 

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max file size to 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize spell checker models
bert_spell_checker = BertSpellChecker()
lstm_spell_checker = LSTMSpellChecker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spelling():
    """Check spelling in the provided text using the specified model and domain.

    Expects JSON input with the following fields:
    - text: The text to check for spelling errors
    - domain: The domain of the text (e.g., 'general', 'medical', 'legal')
    - model: The model to use ('bert' or 'lstm')

    Returns:
        JSON: Dictionary containing:
        - errors: List of detected spelling errors
        - text: Original input text
        - domain: Domain used for checking
        - model: Model used for checking
    """
    data = request.json
    text = data.get('text', '')
    domain = data.get('domain', 'general')
    model_type = data.get('model', 'basic')
    errors = []

    if model_type == 'bert':
        errors = bert_spell_checker.check_spelling(text, domain, model_type)
    elif model_type == 'lstm':
        errors = lstm_spell_checker.correct_sentence(text, domain)
    
    return jsonify({
        'errors': errors,
        'text': text,
        'domain': domain,
        'model': model_type
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and extract text content for spell checking.

    Supports .txt and .docx file formats.

    Returns:
        JSON: Dictionary containing either:
        - text: Extracted text content from the file
        - error: Error message if processing failed
    """
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
        
        os.remove(filepath)  # Clean up the uploaded file
        return jsonify({'text': text})
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)  # Clean up in case of error
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400

if __name__ == '__main__':
    print("Starting Neural Spellchecker webapp...")
    print("Visit: http://localhost:5000")
    app.run(debug=True)