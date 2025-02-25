import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

class FGSMAttack:
    def __init__(self, model, tokenizer, max_length, epsilon=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.epsilon = epsilon
        
        # Get the embedding matrix from the model
        self.embedding_matrix = self.model.layers[0].get_weights()[0]
        # Create reverse word index
        self.reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
    def find_closest_words(self, perturbed_embedding):
        # Calculate distances between perturbed embedding and all word embeddings
        distances = cdist(perturbed_embedding, self.embedding_matrix)
        # Get indices of closest words
        closest_indices = np.argmin(distances, axis=1)
        # Convert indices to words
        closest_words = [self.reverse_word_index.get(idx, '') for idx in closest_indices]
        return closest_words

    def create_adversarial_example(self, text, target_label):
        # Clean and preprocess the input text
        cleaned_text = clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, 
                                      padding='post', truncating='post')
        
        # Store original tokens for later comparison
        original_tokens = [self.reverse_word_index.get(idx, '') for idx in padded_sequence[0] if idx != 0]
        
        # Convert to tensor and ensure proper shape
        input_tensor = tf.convert_to_tensor(padded_sequence, dtype=tf.float32)
        target = tf.reshape(tf.convert_to_tensor([target_label], dtype=tf.float32), (-1, 1))
        
        # Get the embedding layer
        embedding_layer = self.model.layers[0]
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            embedded = embedding_layer(input_tensor)
            current = embedded
            for layer in self.model.layers[1:]:
                current = layer(current)
            loss = tf.keras.losses.binary_crossentropy(target, current)
            
        gradient = tape.gradient(loss, embedded)
        perturbation = self.epsilon * tf.sign(gradient)
        adversarial_embeddings = embedded + perturbation
        
        # Find closest words to perturbed embeddings
        perturbed_words = self.find_closest_words(adversarial_embeddings.numpy()[0])
        
        # Forward pass with perturbed embeddings
        current = adversarial_embeddings
        for layer in self.model.layers[1:]:
            current = layer(current)
            
        return {
            'original_prediction': self.model.predict(padded_sequence, verbose=0)[0],
            'adversarial_prediction': current.numpy()[0],
            'original_tokens': original_tokens,
            'perturbed_words': perturbed_words,
            'perturbation': perturbation.numpy(),
            'adversarial_embeddings': adversarial_embeddings.numpy()
        }
    
    def attack_text(self, text):
        """
        Performs FGSM attack to flip the prediction
        """
        # Get original prediction
        cleaned_text = clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, 
                                      padding='post', truncating='post')
        
        padded_sequence = np.array(padded_sequence)
        orig_pred = self.model.predict(padded_sequence, verbose=0)[0][0]
        
        target = 0.0 if orig_pred > 0.5 else 1.0
        
        result = self.create_adversarial_example(text, target)
        
        return {
            'original_text': text,
            'adversarial_text': ' '.join(word for word in result['perturbed_words'] if word),
            'original_sentiment': 'Positive' if orig_pred > 0.5 else 'Negative',
            'original_confidence': float(orig_pred),
            'adversarial_sentiment': 'Positive' if result['adversarial_prediction'] > 0.5 else 'Negative',
            'adversarial_confidence': float(result['adversarial_prediction']),
            'attack_success': (orig_pred > 0.5) != (result['adversarial_prediction'] > 0.5)
        }

def demo_attack():
    # Initialize attack
    attack = FGSMAttack(model, tokenizer, max_length, epsilon=0.15)
    
    # Test sample
    sample_text = "bad movie."
    
    # Perform attack
    result = attack.attack_text(sample_text)
    
    # Print results
    print("\nOriginal Text:", result['original_text'])
    print("\nAdversarial Text:", result['adversarial_text'])
    print("\nOriginal Sentiment:", result['original_sentiment'], 
          f"(Confidence: {result['original_confidence']:.2f})")
    print("Adversarial Sentiment:", result['adversarial_sentiment'], 
          f"(Confidence: {result['adversarial_confidence']:.2f})")
    print("Attack Success:", result['attack_success'])

if __name__ == "__main__":
    demo_attack()
