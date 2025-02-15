import tensorflow as tf

def fgsm_attack_single_image(model, image, label, epsilon, normalized=True):
    """
    Perform FGSM attack on a single image without normalization.
    
    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        label (tf.Tensor): True label (integer).
        epsilon (float): Perturbation magnitude.
        normalized (bool): Normalized image.

    Returns:
        tf.Tensor: Adversarial example in [0, 255].
    """
    # Adjust epsilon for non normalized images
    if not normalized:
        epsilon = epsilon * 255.0

    # Ensure the image has a batch dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Add batch dimension
    label = tf.expand_dims(label, axis=0) 

    with tf.GradientTape() as tape:
        tape.watch(image)
        # Forward pass
        prediction = model(image, training=False)
        # Calculate loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

    # Calculate gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, image)

    # Get the sign of the gradient
    gradient_sign = tf.sign(gradient)

    # Generate adversarial example
    adversarial_image = image + epsilon * gradient_sign 

    # Clip values to stay in valid range [0, 255]
    if normalized:
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    else:
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)

    return tf.squeeze(adversarial_image)