from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import docx
import os
from werkzeug.utils import secure_filename

from BertSpellChecker import BertSpellChecker
from LSTMSpellChecker import LSTMSpellChecker 


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

bert_spell_checker = BertSpellChecker()
lstm_spell_checker = LSTMSpellChecker()  # Placeholder for LSTM spell checker if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spelling():
    data = request.json
    text = data.get('text', '')
    domain = data.get('domain', 'general')
    model_type = data.get('model', 'basic')
    errors = []

    # print(model_type)
    if(model_type == 'bert'):
        errors = bert_spell_checker.check_spelling(text, domain, model_type)
        print(errors)
    elif(model_type == 'lstm'):
        errors = lstm_spell_checker.correct_sentence(text, domain)
    
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

# @app.route('/domains')
# def get_domains():
#     return jsonify(list(spellchecker.domain_configs.keys()))

if __name__ == '__main__':
    print("Starting Neural Spellchecker webapp...")
    print("Visit: http://localhost:5000")
    app.run(debug=True)