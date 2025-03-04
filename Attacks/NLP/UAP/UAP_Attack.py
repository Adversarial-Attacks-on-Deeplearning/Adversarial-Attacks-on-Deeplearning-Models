import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random

def implement_uap_attack(model, tokenizer, train_data, test_data, vocab_size=500, max_length=200, 
                         num_iterations=5, learning_rate=1.0, batch_size=8, insertion_position=0):
    """
    Implements a Universal Adversarial Perturbation (UAP) attack for text classifiers using gradients.
    
    Args:
        model: The target model to attack
        tokenizer: Tokenizer used by the model
        train_data: Training data as (texts, labels) for finding the adversarial word
        test_data: Test data as (texts, labels) for evaluation
        vocab_size: Size of the vocabulary
        max_length: Maximum sequence length
        num_iterations: Number of iterations for optimizing the adversarial word
        learning_rate: Learning rate for gradient updates
        batch_size: Batch size for training
        insertion_position: Position to insert the adversarial word (0 = beginning)
        
    Returns:
        adversarial_word: The found adversarial word
        original_accuracy: Model accuracy without the attack
        adversarial_accuracy: Model accuracy under attack
    """
    # Get embedding layer from the model
    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()[0]
    embedding_dim = embedding_weights.shape[1]
    
    # Initialize adversarial word with a random word from vocabulary
    random_word_idx = random.randint(1, vocab_size - 1)  # Skip 0 (padding)
    adv_embedding = embedding_weights[random_word_idx].copy()
    
    # Create a mapping from embedding to word (inverse of tokenizer)
    word_to_idx = tokenizer.word_index
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    # Get a subset of training data for optimization
    train_texts, train_labels = train_data
    
    # Evaluate original accuracy
    test_texts, test_labels = test_data
    original_preds = model.predict(test_texts)
    original_accuracy = accuracy_score(test_labels, np.argmax(original_preds, axis=1))
    print(f"Original model accuracy: {original_accuracy:.4f}")
    
    # Optimization loop
    best_adv_word = None
    best_attack_success_rate = 0
    
    for iteration in tqdm(range(num_iterations)):
        # Choose a random batch from training data
        batch_indices = np.random.choice(len(train_labels), batch_size, replace=False)
        batch_x = train_texts[batch_indices]
        batch_y = train_labels[batch_indices]
        
        # Create a TensorFlow variable for the adversarial embedding
        adv_embedding_var = tf.Variable(adv_embedding, dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            # Create adversarial examples by inserting the word
            adv_inputs = np.copy(batch_x)
            for i in range(len(adv_inputs)):
                adv_inputs[i, insertion_position] = random_word_idx
            
            # Compute model predictions
            embedded_inputs = embedding_layer(adv_inputs)
            predictions = model(embedded_inputs)
            
            # Maximize loss for non-targeted attack
            loss = -tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions))
        
        # Compute gradients with respect to embedding
        grads = tape.gradient(loss, adv_embedding_var)
        
        # Update adversarial embedding (gradient ascent for non-targeted attack)
        adv_embedding = adv_embedding_var.numpy() + learning_rate * grads.numpy()
        
        # Project back to vocabulary space (find the closest word)
        similarities = np.dot(embedding_weights[1:vocab_size], adv_embedding)  # Skip padding token
        closest_word_idx = np.argmax(similarities) + 1  # Offset by 1 to account for skipping 0
        adv_embedding = embedding_weights[closest_word_idx].copy()
        
        # Evaluate attack success rate periodically
        if iteration % 2 == 0 or iteration == num_iterations - 1:
            # Apply the adversarial word to test data
            test_adv_texts = np.copy(test_texts)
            for i in range(len(test_adv_texts)):
                test_adv_texts[i, insertion_position] = closest_word_idx
            
            adv_preds = model.predict(test_adv_texts)
            adv_accuracy = accuracy_score(test_labels, np.argmax(adv_preds, axis=1))
            attack_success_rate = original_accuracy - adv_accuracy
            
            print(f"Iteration {iteration}: Adversarial word = '{idx_to_word.get(closest_word_idx, '<UNK>')}', "
                  f"Attack success rate = {attack_success_rate:.4f}")
            
            if attack_success_rate > best_attack_success_rate:
                best_attack_success_rate = attack_success_rate
                best_adv_word = idx_to_word.get(closest_word_idx, '<UNK>')
    
    # Final evaluation
    print(f"\nBest adversarial word: '{best_adv_word}'")
    print(f"Original accuracy: {original_accuracy:.4f}")
    print(f"Adversarial accuracy: {original_accuracy - best_attack_success_rate:.4f}")
    print(f"Attack success rate: {best_attack_success_rate:.4f}")
    
    return best_adv_word, original_accuracy, original_accuracy - best_attack_success_rate
