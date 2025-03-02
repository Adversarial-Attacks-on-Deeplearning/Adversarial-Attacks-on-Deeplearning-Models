import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

def find_closest_word(embedding_vector, tokenizer, embeddings_index):
    min_distance = float("inf")
    closest_word = None
    
    for word in tokenizer.word_index.keys():  # Only use words in tokenizer vocabulary
        if word in embeddings_index:
            word_embedding = embeddings_index[word]
            distance = cosine(word_embedding, embedding_vector)
            if distance < min_distance:
                min_distance = distance
                closest_word = word

    return closest_word if closest_word else "UNKNOWN"


def generate_single_word_uap(model, tokenizer, embeddings_index, vocab_size, embedding_dim, max_length, alpha=0.4, num_iterations=5):
    """
    Generates a Universal Adversarial Perturbation (UAP) consisting of a **single** adversarial word.
    """
    # Ensure embedding layer is trainable
    embedding_layer = model.layers[0]
    if not embedding_layer.trainable:
        raise ValueError("Embedding layer must be trainable for gradient-based attacks.")

    # Get a random word from the vocabulary
    vocab_words = list(tokenizer.word_index.keys())
    if not vocab_words:  # Ensure vocabulary is not empty
        raise ValueError("Tokenizer vocabulary is empty. Check if the tokenizer has been properly fitted.")

    initial_word = np.random.choice(vocab_words)
    adversarial_word = initial_word

    for _ in range(num_iterations):
        # Convert adversarial word to token index
        adversarial_index = tokenizer.texts_to_sequences([adversarial_word])

        # Ensure valid sequence
        if not adversarial_index or len(adversarial_index[0]) == 0:
            print(f"Warning: '{adversarial_word}' not found in vocabulary. Skipping update.")
            break  # Exit loop if no valid tokenization

        adversarial_index = tf.keras.preprocessing.sequence.pad_sequences(
            adversarial_index, maxlen=1, padding="post"
        )  # Ensure shape (1,1)
        
        adversarial_index = tf.convert_to_tensor(adversarial_index, dtype=tf.int32)

        # Get embedding for the adversarial word
        adversarial_embedding = embedding_layer(adversarial_index)  # Shape: (1, 1, embedding_dim)

        with tf.GradientTape() as tape:
            tape.watch(adversarial_embedding)  # Ensure TensorFlow tracks this tensor
            
            # Convert adversarial embedding back to a token index sequence
            adversarial_token = tokenizer.texts_to_sequences([adversarial_word])
            adversarial_token = tf.keras.preprocessing.sequence.pad_sequences(
                adversarial_token, maxlen=max_length, padding="post"
            )  # Shape: (1, max_length)

            # Convert to tensor (ensure proper dtype)
            adversarial_token = tf.convert_to_tensor(adversarial_token, dtype=tf.int32)

            # Forward pass through the model
            pred = model(adversarial_token)  # Now passing integer token indices

            target_label = tf.argmax(pred, axis=1)  # Get predicted label
            loss = tf.keras.losses.sparse_categorical_crossentropy(target_label, pred)

        # Compute gradient w.r.t. the embedding
        grads = tape.gradient(loss, adversarial_embedding)

        if grads is None:
            raise ValueError("Gradient computation failed. Ensure the model and embeddings are trainable.")

        # Update the adversarial embedding
        updated_embedding = adversarial_embedding + alpha * grads

        # Find the closest word in the vocabulary
        closest_word = find_closest_word(updated_embedding.numpy()[0][0], tokenizer, embeddings_index)
        if closest_word:
            adversarial_word = closest_word  # Update to new adversarial word
    
    return adversarial_word

# Generate a single-word UAP and print
uap_word = generate_single_word_uap(model, tokenizer, embeddings_index, vocab_size=20000, embedding_dim=100, max_length=200)
print("Generated UAP Word:", uap_word)
