import tensorflow as tf

def black_box_fn(image: tf.Tensor) -> tf.Tensor:
    """
    A black-box function that takes a single image of shape (H, W, C)
    and returns the predicted probabilities (num_classes,).
    """
    # Expand dimensions to make it (1, H, W, C)
    image_batch = tf.expand_dims(image, axis=0)

    # Model forward pass -> shape: (1, num_classes)
    logits = model(image_batch)

    # Convert logits to probabilities
    probs = tf.nn.softmax(logits[0])  # shape: (num_classes,)

    return probs
