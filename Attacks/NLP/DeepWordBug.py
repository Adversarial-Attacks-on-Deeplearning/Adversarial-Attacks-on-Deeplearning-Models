import tensorflow as tf
import numpy as np
from typing import List, Tuple
import random
import string
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DeepWordBug:
    def __init__(self, model_path: str, tokenizer, max_length: int = 200, epsilon: float = 0.1, lambda_param: float = 0.5):
        """
        Initialize DeepWordBug attack for TensorFlow model.
        
        Args:
            model_path: Path to saved Keras model
            tokenizer: Keras tokenizer used for the model
            max_length: Maximum sequence length for padding
            epsilon: Maximum allowed perturbation ratio
            lambda_param: Weight for combining temporal scoring fuction
        """
        self.model = keras.models.load_model('NLPModel.h5')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        
    def predict(self, text: str) -> float:
        """Get model prediction for a text input."""
        # Convert text to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        # Get prediction
        prediction = self.model.predict(padded, verbose=0)[0][0]
        return prediction
    
    def score_r1s(self, text: str, word_position: int) -> float:
        """Calculate Replace-1 Score for a word."""
        words = text.split()
        original_score = self.predict(text)
        
        if 0 <= word_position < len(words):
            modified_words = words.copy()
            modified_words[word_position] = '<UNK>'
            modified_text = ' '.join(modified_words)
            modified_score = self.predict(modified_text)
            return abs(original_score - modified_score)
        return 0.0
    
    def score_ths(self, text: str, word_position: int) -> float:
        """Calculate Temporal Head Score."""
        words = text.split()
        if 0 <= word_position < len(words):
            head_text = ' '.join(words[:word_position + 1])
            partial_text = ' '.join(words[:word_position])
            return abs(self.predict(head_text) - self.predict(partial_text))
        return 0.0
    
    def score_tts(self, text: str, word_position: int) -> float:
        """Calculate Temporal Tail Score."""
        words = text.split()
        if 0 <= word_position < len(words):
            tail_text = ' '.join(words[word_position:])
            partial_text = ' '.join(words[word_position + 1:])
            return abs(self.predict(tail_text) - self.predict(partial_text))
        return 0.0
    
    def score_combined(self, text: str, word_position: int) -> float:
        """Calculate Combined Score using THS and TTS."""
        ths = self.score_ths(text, word_position)
        tts = self.score_tts(text, word_position)
        return ths + self.lambda_param * tts
    
    def transform_token(self, token: str, method: str = None) -> str:
        """Transform a token using various methods."""
        if len(token) <= 1:
            return token
            
        methods = ['swap', 'substitute', 'delete', 'insert']
        method = method or random.choice(methods)
        
        if method == 'swap':
            pos = random.randint(0, len(token) - 2)
            chars = list(token)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
            
        elif method == 'substitute':
            pos = random.randint(0, len(token) - 1)
            chars = list(token)
            chars[pos] = random.choice(string.ascii_lowercase)
            return ''.join(chars)
            
        elif method == 'delete':
            pos = random.randint(0, len(token) - 1)
            return token[:pos] + token[pos + 1:]
            
        elif method == 'insert':
            pos = random.randint(0, len(token))
            char = random.choice(string.ascii_lowercase)
            return token[:pos] + char + token[pos:]
    
    def calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.calculate_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
    
    def attack(self, text: str, scoring_method: str = 'rls') -> Tuple[str, float, float]:
        """
        Generate adversarial text using DeepWordBug attack.
        
        Args:
            text: Input text to attack
            scoring_method: Method to score tokens ('r1s', 'ths', 'tts', 'combined')
        
        Returns:
            Tuple of (adversarial_text, original_score, adversarial_score)
        """
        words = text.split()
        original_score = self.predict(text)
        
        # Score each word
        scores = []
        for i, word in enumerate(words):
            if scoring_method == 'r1s':
                score = self.score_r1s(text, i)
            elif scoring_method == 'ths':
                score = self.score_ths(text, i)
            elif scoring_method == 'tts':
                score = self.score_tts(text, i)
            else:  # combined
                score = self.score_combined(text, i)
            scores.append((i, score))
        
        # Sort words by importance score
        sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Calculate maximum number of words to modify
        max_modifications = int(len(words) * self.epsilon)
        
        # Modify words
        modified_words = words.copy()
        modifications = 0
        
        for idx, _ in sorted_words:
            if modifications >= max_modifications:
                break
                
            original_word = words[idx]
            modified_word = self.transform_token(original_word)
            
            # Check if modification maintains readability
            edit_dist = self.calculate_edit_distance(original_word, modified_word)
            if edit_dist <= 2:  # Limit per-word modifications
                modified_words[idx] = modified_word
                modifications += 1
        
        adversarial_text = ' '.join(modified_words)
        adversarial_score = self.predict(adversarial_text)
        
        return adversarial_text, original_score, adversarial_score

# Example usage
if __name__ == "__main__":
    # Initialize attack with your saved model and tokenizer
    attack = DeepWordBug(
        model_path='NLPModel.h5',
        tokenizer=tokenizer,  # Your existing tokenizer
        max_length=200,      # Same as in your model
        epsilon=30 #A hyperparameter must be tuned
    )
    
    
