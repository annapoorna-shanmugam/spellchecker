Problem Statement: Statistical Spellchecker
Objective:


Design a neural network-based spellchecker that corrects domain-specific terminology (domain -> medical) using context-aware models. Users should analyze how the system prioritizes corrections using semantic and syntactic patterns.

Requirements
Web Interface (3 Marks)
Front-End Development:

Build an interface using HTML/CSS/JavaScript with:

Text Input: For user-provided text or file uploads (.txt, .docx).

Domain Selection: Dropdown to choose domain (e.g., medical, legal).

Model Selection: Options to switch between neural architectures (e.g., BERT, LSTM).

Display:

Highlighted errors with confidence scores (e.g., "hypertention → hypertension [98%]").

Separate outputs for general vs. domain-specific corrections.

Spellchecker Logic (3 Marks)
Back-End Implementation (Flask/Python):

Implement a neural spellchecker using:

Pre-trained models (e.g., BERT, GPT-2) fine-tuned on domain corpora (e.g., PubMed, legal case files).

Domain config file: JSON/YAML file specifying:

Features:

Context-aware corrections using attention mechanisms.

Error classification:

General errors: Detected via standard dictionaries.

Domain errors: Prioritized using config file terms.

Integration (2 Marks)
Connect front-end and back-end for real-time corrections.

Support bulk file processing with progress tracking.

