"""
BERT-based Spell Checker Module

This module implements a spell checker using BERT (Bidirectional Encoder Representations
from Transformers) models. It supports both general-purpose and domain-specific
spell checking using specialized BERT models for medical and legal domains.

The spell checker combines traditional dictionary-based checking with contextual
suggestions from BERT models to provide accurate spelling corrections.

Dependencies:
    - transformers: For BERT models and tokenizers
    - torch: For neural network operations
    - yaml: For loading domain configurations
    - spellchecker: For basic dictionary-based spell checking

"""

import yaml
import re
from werkzeug.utils import secure_filename
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from spellchecker import SpellChecker
import torch.nn.functional as F


class BertSpellChecker:
    """
    A spell checker class that uses BERT models for context-aware spell checking.
    
    This class combines traditional dictionary-based spell checking with BERT models
    to provide context-aware spelling suggestions. It supports domain-specific
    spell checking for medical and legal texts using specialized BERT models.

    Attributes:
        model_name (str): Name of the base BERT model
        tokenizer: BERT tokenizer for processing text
        model: BERT model for masked language modeling
        domain_configs (dict): Configuration for different domains
        spell: Traditional spell checker instance
        domain_models (dict): Domain-specific BERT models
    """

    def __init__(self):
        """Initialize the BERT spell checker with models and configurations."""
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
        self.domain_models = {
            'medical': ('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', None),
            'legal': ('nlpaueb/legal-bert-base-uncased', None)
        }

    def is_word_correct(self, word):
        """
        Check if a word is spelled correctly.

        Args:
            word (str): The word to check

        Returns:
            bool: True if the word is correct, False otherwise

        The check is performed against both the general dictionary
        and domain-specific term lists.
        """
        word_lower = word.lower()
        if word_lower in self.spell:
            return True
        for domain_config in self.domain_configs.values():
            terms = domain_config.get('terms', [])
            if word_lower in [term.lower() for term in terms]:
                return True
        return False
    
    def get_suggestions(self, masked_text, error_words_suggestion, error_words, domain):
        """
        Get spelling suggestions for masked words using BERT.

        Args:
            masked_text (str): Text with [MASK] tokens replacing error words
            error_words_suggestion (dict): Dictionary to store suggestions
            error_words (list): Original error words
            domain (str): Domain for specialized spell checking

        The method uses BERT's masked language modeling to predict
        likely words for each masked position, considering the context.
        """
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

        top_k = 5
        for idx, mask_pos in enumerate(mask_token_indices):
            mask_logits = logits[0, mask_pos, :]
            probs = F.softmax(mask_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            word_suggestions = []
            for token_id, prob in zip(top_indices, top_probs):
                predicted_word = self.tokenizer.decode([token_id]).strip()
                word_suggestions.append((predicted_word, float(prob)))

            # Sort suggestions by Levenshtein distance to the original error word, then by probability
            word_suggestions.sort(key=lambda x: (levenshtein_distance(error_words[idx].lower(), x[0].lower()), -x[1]))
            print(f"Suggestions for '{error_words[idx]}': {word_suggestions}")
            # Assign suggestions to the corresponding error word
            error_words_suggestion[error_words[idx]] = word_suggestions[0]

    def check_spelling(self, text, domain="general", model_type="basic"):
        """
        Check spelling in the given text and provide suggestions.

        Args:
            text (str): Input text to check
            domain (str): Domain for specialized spell checking (default: "general")
            model_type (str): Type of model to use (default: "basic")

        Returns:
            list: List of dictionaries containing error information:
                - word: The misspelled word
                - position: Position in the text
                - suggestions: List of suggested corrections
                - type: Error type (domain/general)
                - confidence: Confidence score for the suggestion
        """
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
    
def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1 (str): First string
        s2 (str): Second string

    Returns:
        int: The Levenshtein distance between s1 and s2

    The Levenshtein distance is the minimum number of single-character
    edits required to change one string into another.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
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