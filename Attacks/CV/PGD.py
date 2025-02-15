import tensorflow as tf

def pgd_attack_single_image(model, image, label, epsilon, alpha, iterations, normalized=True):
    """
    Perform PGD attack on a single image without normalization.
    
    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        label (tf.Tensor): True label (integer).
        epsilon (float): Maximum perturbation magnitude.
        alpha (float): Step size for each iteration.
        iterations (int): Number of PGD iterations.
        normalized (bool): For normalized images, set to True.

    Returns:
        tf.Tensor: Adversarial example in [0, 255].
    """
    # Adjust epsilon for non normalized images
    if not normalized:
        epsilon = epsilon * 255.0
    # Ensure the image has a batch dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    label = tf.expand_dims(label, axis=0)  # Add batch dimension

    # Initialize adversarial image with the original image
    adversarial_image = tf.identity(image)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            # Forward pass
            prediction = model(adversarial_image, training=False)
            # Calculate loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
        
        # Calculate gradient of the loss w.r.t. the adversarial image
        gradient = tape.gradient(loss, adversarial_image)

        # Get the sign of the gradient and update the adversarial image
        adversarial_image = adversarial_image + alpha * tf.sign(gradient)

        # Project the adversarial image into the epsilon-ball and clip to [0, 255]
        adversarial_image = tf.clip_by_value(adversarial_image, image - epsilon, image + epsilon)
        if normalized:
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        else:
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)

    return tf.squeeze(adversarial_image)
