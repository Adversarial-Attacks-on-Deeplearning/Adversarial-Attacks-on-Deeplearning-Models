import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the trained sentiment analysis model
model = load_model("sentiment_analysis_model.h5")

# Assuming the tokenizer is preloaded (replace with actual tokenizer)
tokenizer = Tokenizer()

def predict_sentiment(model, text, tokenizer, max_length=200):
    """Predicts the sentiment of a given text using the model."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequences)[0][0]  # Assuming binary classification

    sentiment = "positive" if prediction >= 0.5 else "negative"
    confidence = float(prediction if sentiment == "positive" else 1 - prediction)
    
    return sentiment, confidence

def perturb_word(word, attack_type="random"):
    """Applies character-level perturbations to a word."""
    if len(word) < 3:
        return word  # Avoid modifying very short words
    
    if attack_type == "random":
        attack_type = random.choice(["swap", "replace", "insert"])
    
    word = list(word)
    
    if attack_type == "swap" and len(word) > 3:
        i = random.randint(1, len(word) - 2)  # Avoid first and last character
        word[i], word[i + 1] = word[i + 1], word[i]
    
    elif attack_type == "replace":
        char_map = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}
        i = random.randint(0, len(word) - 1)
        if word[i] in char_map:
            word[i] = char_map[word[i]]
    
    elif attack_type == "insert":
        i = random.randint(1, len(word) - 1)
        word.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
    
    return "".join(word)

def adversarial_attack(model, text, tokenizer, max_length=200, max_attempts=10, attack_prob=0.7, attack_type="random"):
    """Applies a character-level adversarial attack to flip the sentiment."""
    words = text.split()
    original_sentiment, confidence = predict_sentiment(model, text, tokenizer, max_length)
    
    print(f"\nOriginal Text: {text}")
    print(f"Original Sentiment: {original_sentiment} (Confidence: {confidence:.2f})")
    
    for _ in range(max_attempts):
        modified_text = words.copy()
        modifications = []
        
        for i in range(len(modified_text)):
            if random.random() < attack_prob:  # Apply attack based on probability
                original_word = modified_text[i]
                modified_text[i] = perturb_word(modified_text[i], attack_type)
                modifications.append((original_word, modified_text[i]))
        
        modified_text = " ".join(modified_text)
        
        new_sentiment, new_confidence = predict_sentiment(model, modified_text, tokenizer, max_length)
        
        if new_sentiment != original_sentiment:
            print(f"\nAdversarial Text: {modified_text}")
            print(f"Modifications: {modifications}")
            print(f"New Sentiment: {new_sentiment} (Confidence: {new_confidence:.2f})\n")
            return modified_text, new_sentiment, new_confidence
    
    print("\nAttack failed to flip sentiment.")
    return text, original_sentiment, confidence  # Return original if attack fails

# Test cases for adversarial attacks
def test_adversarial_attack(model, tokenizer, attack_type="random"):
    test_cases = [
        "The coffee was perfect, and the ambiance was cozy.",
        "I went to this place with high expectations, but I was really disappointed. "
        "The pasta was overcooked, the sauce was bland, and the service was incredibly slow. "
        "Honestly, I don't think I’ll be coming back anytime soon."
    ]
    
    for text in test_cases:
        adversarial_attack(model, text, tokenizer, attack_type=attack_type)

# Run test cases with different attack types
test_adversarial_attack(model, tokenizer, attack_type="swap")
test_adversarial_attack(model, tokenizer, attack_type="insert")
test_adversarial_attack(model, tokenizer, attack_type="replace")
