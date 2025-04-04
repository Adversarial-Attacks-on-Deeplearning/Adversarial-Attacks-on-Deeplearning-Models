# SimBA-DCT Attack Implementation
def simba_dct_attack(image_tensor, true_label, model, max_queries=2000, epsilon=0.2, low_freq_fraction=1/8, step_size=0.2, print_every=10):
    """
    image_tensor: TensorFlow tensor of shape (1, H, W, 3) in [0, 255]
    true_label: Integer label (the true class of the image)
    model: TensorFlow/Keras model used for prediction
    max_queries: Maximum number of iterations (queries)
    epsilon: Perturbation magnitude for each pixel update
    low_freq_fraction: Fraction of DCT coefficients to select (default 1/8)
    step_size: Step size for perturbation in the DCT domain
    print_every: Frequency (in iterations) at which to print detailed logs
    """
    # Adapt epsilon value for the model that accepts unnormalized images
    epsilon = epsilon * 255

    # Preprocess the image: Convert to DCT domain and select low-frequency components
    dct_coeffs_tensor, low_freq_mask = preprocess_image_dct_tensor(image_tensor, low_freq_fraction)
    
    # Initialize perturbation in DCT space and other variables
    perturbation_dct = np.zeros_like(dct_coeffs_tensor.numpy())  # Initialize as zero
    current_dct = dct_coeffs_tensor.numpy() * low_freq_mask  # Keep only low-frequency components
    available_frequencies = np.argwhere(low_freq_mask)  # Indices of the low-frequency components
    query_count = 0
    original_label = true_label
    
    # Get initial prediction to compare success
    initial_pred = model.predict(image_tensor)
    original_prob = initial_pred[0][original_label]
    print(f"Initial true class probability: {original_prob:.4f}")
    
    # Start attack iterations
    for i in range(max_queries):
        if query_count >= max_queries:
            break
        
        # Select a random low-frequency component (basis vector)
        idx = np.random.choice(available_frequencies.flatten())
        basis_vector_dct = np.zeros_like(current_dct)
        basis_vector_dct[idx] = 1  # Create a unit vector in the DCT space

        # Apply positive perturbation in the DCT domain
        dct_plus = current_dct + step_size * basis_vector_dct
        perturbed_image_plus = idct2(dct_plus)
        perturbed_image_plus = np.clip(perturbed_image_plus, 0, 255)

        # Query the model with positive perturbation
        perturbed_image_plus_tensor = tf.convert_to_tensor(perturbed_image_plus, dtype=tf.float32)
        prob_original_plus = model.predict(perturbed_image_plus_tensor)[0][original_label]
        query_count += 1
        
        # Check if the positive perturbation decreased the probability
        if prob_original_plus < original_prob:
            perturbation_dct += step_size * basis_vector_dct  # Update perturbation
            current_dct = dct_plus  # Update current DCT representation
            original_prob = prob_original_plus
            print(f"Iteration {i}: +epsilon -> Basis vector {idx}, Prob: {original_prob:.6f}")
            continue
        
        # Apply negative perturbation in the DCT domain if the positive did not help
        dct_minus = current_dct - step_size * basis_vector_dct
        perturbed_image_minus = idct2(dct_minus)
        perturbed_image_minus = np.clip(perturbed_image_minus, 0, 255)

        # Query the model with negative perturbation
        perturbed_image_minus_tensor = tf.convert_to_tensor(perturbed_image_minus, dtype=tf.float32)
        prob_original_minus = model.predict(perturbed_image_minus_tensor)[0][original_label]
        query_count += 1
        
        # Check if the negative perturbation decreased the probability
        if prob_original_minus < original_prob:
            perturbation_dct -= step_size * basis_vector_dct  # Update perturbation
            current_dct = dct_minus  # Update current DCT representation
            original_prob = prob_original_minus
            print(f"Iteration {i}: -epsilon -> Basis vector {idx}, Prob: {original_prob:.6f}")
            continue
        
        # If no improvement, skip this iteration
        if i % print_every == 0:
            print(f"Iteration {i}: No improvement at Basis vector {idx}. Prob: {original_prob:.6f}")

    # Reconstruct the adversarial image from the final DCT + perturbation
    adversarial_dct = current_dct + perturbation_dct
    adversarial_image = idct2(adversarial_dct)
    adversarial_image = np.clip(adversarial_image, 0, 255)
    
    # Convert adversarial image to tensor for model input
    adversarial_image_tensor = tf.convert_to_tensor(adversarial_image, dtype=tf.float32)
    
    print(f"Attack finished after {query_count} queries. Final probability: {original_prob:.4f}")
    return adversarial_image_tensor, query_count, original_prob
