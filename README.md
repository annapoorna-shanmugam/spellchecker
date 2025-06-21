# Neural Spellchecker Web Application

A context-aware, domain-specific spellchecker using neural networks (BERT/DistilBERT) for medical and legal terminology correction.

## Features

- **Context-Aware Corrections**: Uses BERT/DistilBERT models for context-based suggestions
- **Domain-Specific**: Specialized correction for medical and legal terminology
- **Real-time Processing**: Interactive web interface with progress tracking
- **File Upload Support**: Upload .txt and .docx files for batch processing
- **Confidence Scores**: Shows correction confidence percentages
- **Error Classification**: Separates general vs domain-specific errors

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Select your domain (General, Medical, Legal) and model (BERT, LSTM)

4. Enter text or upload a file for spell checking

## Architecture

### Backend (Flask)
- `app.py`: Main Flask application with spellchecker logic
- `domain_config.yaml`: Domain-specific terminology configuration
- Neural model integration using HuggingFace Transformers

### Frontend
- `templates/index.html`: Main web interface
- `static/style.css`: Responsive styling
- `static/script.js`: Interactive JavaScript functionality

### Key Components
- **NeuralSpellChecker**: Core class handling BERT/DistilBERT integration
- **Context Analysis**: Uses masked language modeling for context-aware corrections
- **Domain Classification**: Prioritizes domain-specific terms over general corrections
- **Error Highlighting**: Visual feedback with confidence scores

## Domain Configuration

The `domain_config.yaml` file contains:
- Medical terminology (hypertension, diabetes, etc.)
- Legal terminology (jurisdiction, litigation, etc.)
- Common error mappings for each domain

## API Endpoints

- `POST /check`: Spell check text with domain and model selection
- `POST /upload`: Upload and process .txt/.docx files
- `GET /domains`: Get available domains

## Example Corrections

**Medical Domain:**
- "hypertention" → "hypertension [98%]"
- "diabetis" → "diabetes [96%]"

**Legal Domain:**
- "juristiction" → "jurisdiction [97%]"
- "defendent" → "defendant [95%]"