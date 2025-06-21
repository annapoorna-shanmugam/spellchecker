import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

class LSTMSpellChecker:
    
    def __init__(self):
        # 1. Clean sentences
        self.clean_sentences = []

        self.clean_words = []

        self.noisy_sentences = []
        self.labels = []
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_len = 0

    
    def inject_error(self, sentence):
        words = sentence.split()
        wrong_indices = []
        for i, word in enumerate(words):
            if word.lower() in self.clean_words and random.random() < 0.7:  # 70% chance of error for medical terms
                # Create more varied typos
                error_type = random.choice(['swap', 'insert', 'delete', 'replace'])
                
                if error_type == 'swap' and len(word) > 1:
                    # Swap adjacent characters
                    j = random.randint(0, len(word)-2)
                    typo = word[:j] + word[j+1] + word[j] + word[j+2:]
                elif error_type == 'insert':
                    # Insert a random character
                    j = random.randint(0, len(word))
                    chars = 'abcdefghijklmnopqrstuvwxyz'
                    typo = word[:j] + random.choice(chars) + word[j:]
                elif error_type == 'delete' and len(word) > 1:
                    # Delete a random character
                    j = random.randint(0, len(word)-1)
                    typo = word[:j] + word[j+1:]
                else:  # replace
                    # Replace a character with a similar looking one
                    similar_chars = {
                        'a': 'e', 'e': 'a', 'i': 'y', 'o': 'u',
                        's': 'z', 'c': 'k', 'm': 'n', 'v': 'w'
                    }
                    j = random.randint(0, len(word)-1)
                    if word[j].lower() in similar_chars:
                        replacement = similar_chars[word[j].lower()]
                        typo = word[:j] + replacement + word[j+1:]
                    else:
                        chars = 'abcdefghijklmnopqrstuvwxyz'
                        typo = word[:j] + random.choice(chars) + word[j+1:]
                
                wrong_indices.append(i)
                words[i] = typo
                
        wrong_sentence = " ".join(words)
        return wrong_sentence, wrong_indices
    
    def prepare_model(self):
        try:
            # 1. Generate training data with errors
            print("Generating training data...")
            for sent in self.clean_sentences:
                noisy, wrong_indices = self.inject_error(sent)
                label = [0] * len(sent.split())
                for wrong_idx in wrong_indices:
                    label[wrong_idx] = 1  # mark wrong words
                self.noisy_sentences.append(noisy)
                self.labels.append(label)

            # 2. Prepare sequences
            print("Tokenizing sequences...")
            self.tokenizer.fit_on_texts(self.noisy_sentences + self.clean_sentences)  # Include clean sentences
            vocab_size = len(self.tokenizer.word_index) + 1

            X = self.tokenizer.texts_to_sequences(self.noisy_sentences)
            self.max_len = max(len(x) for x in X)
            X = pad_sequences(X, maxlen=self.max_len, padding='post')

            # 3. Prepare labels
            y = [lbl + [0]*(self.max_len - len(lbl)) for lbl in self.labels]
            y = np.array(y)

            # Build Encoder-Decoder LSTM model
            print("Building Encoder-Decoder LSTM model...")
            
            # Encoder
            encoder_inputs = Input(shape=(self.max_len,))
            encoder_embedding = Embedding(input_dim=vocab_size, 
                                    output_dim=64)(encoder_inputs)
            encoder_lstm = Bidirectional(LSTM(128, return_state=True))
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
            
            # Combine forward and backward states
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            encoder_states = [state_h, state_c]

            # Decoder
            decoder_inputs = Input(shape=(self.max_len,))
            decoder_embedding = Embedding(input_dim=vocab_size, 
                                    output_dim=64)(decoder_inputs)
            decoder_lstm = LSTM(256, return_sequences=True)
            decoder_outputs = decoder_lstm(decoder_embedding, 
                                        initial_state=encoder_states)
            decoder_dense = Dense(vocab_size, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)

            # Define the model
            self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Prepare decoder input (shifted by one timestep)
            decoder_input = np.zeros_like(X)
            decoder_input[:, 1:] = X[:, :-1]  # Teacher forcing: shift input by one timestep

            print("Training model...")
            history = self.model.fit(
                [X, decoder_input],  # Provide both encoder and decoder inputs
                y,
                epochs=2,
                batch_size=32,
                validation_split=0.2,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            print("Model training completed!")
            return history
        except Exception as e:
            print(f"Error in prepare_model: {str(e)}")
            raise

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

    def correct_sentence(self, test_sentence, domain='general'):

        if domain == 'general':
            with open('general_sentences.txt', 'r') as f:
                self.clean_sentences = [line.strip() for line in f if line.strip()]

            self.clean_words = [
                "diagnosis", "infection", "treatment", "medication", "antibiotics",
                "inflammation", "swelling", "biopsy", "symptoms", "therapy",
                "respiratory system", "cardiovascular", "femur", "neurons",
                "liver", "tuberculosis", "pneumonia", "diabetes", "hypertension",
                "tumor", "anemia", "chemotherapy", "laparoscopic", "CT scan",
                "physical therapy", "intravenous", "dialysis", "cardiologist",
                "ECG", "heartbeat", "cancerous", "vaccination", "viral diseases",
                "joints", "nervous system", "detoxification", "contagious",
                "bacterial", "lungs", "mellitus", "stroke", "malignant",
                "surgery", "fatigue", "weakness", "MRI", "X-ray", "bleeding",
                "dehydration", "gastroenteritis", "angioplasty", "echocardiogram",
                "antidepressants", "chronic kidney disease", "antiviral",
                "bronchitis", "glaucoma", "immunosuppressive", "insulin therapy",
                "migraine", "radiation therapy", "urinalysis", "endoscopy",
                "colonoscopy", "epilepsy", "steroids", "beta-blockers",
                "behavioral therapy", "occupational therapy", "pulmonary function test",
                "rheumatoid arthritis", "liver cirrhosis", "bipolar disorder",
                "acute appendicitis", "congestive heart failure", "motion", "statute", "limitations", "appellate court", "prosecution",
                "procedural", "defendant", "jurisdiction", "tribunal", "legal counsel",
                "ruling", "dismissed", "settlement", "judge", "plaintiff",
                "witnesses", "evidence", "jurisdiction", "court", "jury",
                "appeal", "arbitrator", "claim", "lower court", "purview",
                "remanded", "withdrawn", "upheld", "overturned", "granted",
                "conflict of interest", "legal proceedings", "testimony",
                "hearing", "deposition", "injunction", "subpoena", "verdict",
                "pleading", "affidavit", "litigation", "settlement agreement",
                "damages", "judicial review", "precedent", "discovery",
                "cross-examination", "sworn testimony", "bench trial", "mediation",
                "civil case", "criminal case", "prosecuting attorney", "defense counsel",
                "burden of proof", "admissible evidence", "court order",
                "legal proceeding", "appeal process", "judicial system"
            ]

        if domain == 'medical':
            with open('medical_sentences.txt', 'r') as f:
                self.clean_sentences = [line.strip() for line in f if line.strip()]
            self.clean_words = [
                "diagnosis", "infection", "treatment", "medication", "antibiotics",
                "inflammation", "swelling", "biopsy", "symptoms", "therapy",
                "respiratory system", "cardiovascular", "femur", "neurons",
                "liver", "tuberculosis", "pneumonia", "diabetes", "hypertension",
                "tumor", "anemia", "chemotherapy", "laparoscopic", "CT scan",
                "physical therapy", "intravenous", "dialysis", "cardiologist",
                "ECG", "heartbeat", "cancerous", "vaccination", "viral diseases",
                "joints", "nervous system", "detoxification", "contagious",
                "bacterial", "lungs", "mellitus", "stroke", "malignant",
                "surgery", "fatigue", "weakness", "MRI", "X-ray", "bleeding",
                "dehydration", "gastroenteritis", "angioplasty", "echocardiogram",
                "antidepressants", "chronic kidney disease", "antiviral",
                "bronchitis", "glaucoma", "immunosuppressive", "insulin therapy",
                "migraine", "radiation therapy", "urinalysis", "endoscopy",
                "colonoscopy", "epilepsy", "steroids", "beta-blockers",
                "behavioral therapy", "occupational therapy", "pulmonary function test",
                "rheumatoid arthritis", "liver cirrhosis", "bipolar disorder",
                "acute appendicitis", "congestive heart failure"
            ]
            
        if domain == 'legal':
            with open('legal_sentences.txt', 'r') as f:
                self.clean_sentences = [line.strip() for line in f if line.strip()]
            self.clean_words = [
                "motion", "statute", "limitations", "appellate court", "prosecution",
                "procedural", "defendant", "jurisdiction", "tribunal", "legal counsel",
                "ruling", "dismissed", "settlement", "judge", "plaintiff",
                "witnesses", "evidence", "jurisdiction", "court", "jury",
                "appeal", "arbitrator", "claim", "lower court", "purview",
                "remanded", "withdrawn", "upheld", "overturned", "granted",
                "conflict of interest", "legal proceedings", "testimony",
                "hearing", "deposition", "injunction", "subpoena", "verdict",
                "pleading", "affidavit", "litigation", "settlement agreement",
                "damages", "judicial review", "precedent", "discovery",
                "cross-examination", "sworn testimony", "bench trial", "mediation",
                "civil case", "criminal case", "prosecuting attorney", "defense counsel",
                "burden of proof", "admissible evidence", "court order",
                "legal proceeding", "appeal process", "judicial system"
            ]

        # if self.model is None:
        self.prepare_model()

        seq = self.tokenizer.texts_to_sequences([test_sentence])
        X_test = pad_sequences(seq, maxlen=self.max_len, padding='post')

        # Create decoder input (shifted by one timestep for inference)
        decoder_input = np.zeros_like(X_test)
        decoder_input[:, 1:] = X_test[:, :-1]

        # Get predictions using both encoder and decoder inputs
        pred = self.model.predict([X_test, decoder_input], verbose=0)[0]
        
        # Calculate word-level probabilities differently
        # Take the mean of top k probabilities for each word
        k = 5  # Consider top 5 predictions
        word_probs = np.mean(np.sort(pred, axis=-1)[:, -k:], axis=-1)
        
        # Don't normalize probabilities globally, handle each word individually
        results = []
        words = test_sentence.split()
        for i, (word, prob) in enumerate(zip(words, word_probs)):
            # Check if word is in clean_words (case-insensitive)
            is_known = word.lower() in [w.lower() for w in self.clean_words]
            
            # Adjust thresholds
            threshold = 0.6 if is_known else 0.5
            is_wrong = prob < threshold or (not is_known and len(word) > 3)
            
            if is_wrong:  # Remove length check to consider all words
                # Find similar words using Levenshtein distance
                suggestions = []
                for clean_word in self.clean_words:
                    distance = self.levenshtein_distance(word.lower(), clean_word.lower())
                    max_allowed_distance = min(3, max(2, len(word) // 4))
                    
                    if distance <= max_allowed_distance:
                        # Calculate normalized similarity score
                        similarity = 1.0 - (distance / max(len(word), len(clean_word)))
                        length_penalty = abs(len(word) - len(clean_word)) * 0.2
                        final_score = float(max(0, min(1, similarity - length_penalty)))
                        
                        if word.lower().startswith(clean_word.lower()) or clean_word.lower().startswith(word.lower()):
                            final_score = float(min(1, final_score + 0.1))
                            
                        suggestions.append((clean_word, final_score))
                
                if suggestions and word not in self.clean_words:
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    good_suggestions = [(word, float(score)) for word, score in suggestions if score > 0.5]
                    
                    if good_suggestions:
                        top_suggestions = [s[0] for s in good_suggestions[:5]]
                        suggested_domain = 'general'
                        
                        # Calculate confidence based on multiple factors
                        # 1. How different the word is from suggestions (Levenshtein distance)
                        best_distance = min(self.levenshtein_distance(word.lower(), s.lower()) for s in top_suggestions)
                        distance_factor = 1.0 - (best_distance / len(word))
                        
                        # 2. Model's prediction probability
                        prob_factor = float(prob)
                        
                        # 3. Whether the word is known in the domain
                        domain_factor = 0.8 if is_known else 0.4
                        
                        # Combine factors for final confidence
                        confidence = float(
                            0.4 * distance_factor +  # 40% weight to string similarity
                            0.4 * (1.0 - prob_factor) +  # 40% weight to model prediction
                            0.2 * domain_factor  # 20% weight to domain knowledge
                        )
                        
                        if domain == 'medical' or domain == 'legal' and any(s in self.clean_words for s in top_suggestions):
                            suggested_domain = 'domain'
                            
                        results.append({
                            'word': word,
                            'position': i,
                            'suggestions': [[top_suggestions[0], confidence]],
                            'type': suggested_domain,
                            'confidence': confidence
                        })
        
        return results