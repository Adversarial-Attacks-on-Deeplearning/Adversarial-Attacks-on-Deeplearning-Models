import tensorflow as tf


def mi_fgsm(model, x, y, epsilon=8.0, T=10, mu=1.0):
    """
    MI-FGSM attack for models that take unnormalized inputs in [0, 255].

    Args:
        model: A tf.keras.Model that outputs logits.
        x: Input image tensor (float32) in [0, 255], shape (batch_size, H, W, C).
        y: Ground-truth label (int scalar, int vector, or one-hot),
           with shape (), (batch_size,), (num_classes,), or (batch_size, num_classes).
        epsilon: Perturbation bound (L∞ norm), e.g. 8.0 for images in [0, 255].
        T: Number of iterations.
        mu: Momentum decay factor.

    Returns:
        x_star: Adversarial image tensor in [0, 255], same shape as x.
    """
    # Scale epsilon to pixel range
    epsilon = epsilon * 255.0

    # Cast inputs
    x = tf.cast(x, tf.float32)
    x_star = tf.identity(x)
    batch_size = tf.shape(x)[0]

    # Step size per iteration
    alpha = epsilon / float(T)

    # Initialize momentum buffer
    g = tf.zeros_like(x)

    # Prepare labels
    num_classes = model.output_shape[-1]
    y = tf.cast(y, tf.int32)
    y_shape = y.shape
    # Case 1: scalar label
    if y_shape.ndims == 0:
        y = tf.expand_dims(y, 0)
        y = tf.one_hot(y, depth=num_classes)
    # Case 2: vector
    elif y_shape.ndims == 1:
        # If length equals num_classes, assume one-hot vector
        if y_shape[0] == num_classes:
            y = tf.expand_dims(tf.cast(y, tf.float32), 0)
        else:
            # integer class indices
            y = tf.one_hot(y, depth=num_classes)
    # Case 3: already (batch_size, num_classes)
    elif y_shape.ndims == 2 and y_shape[1] == num_classes:
        y = tf.cast(y, tf.float32)
    else:
        raise ValueError(f"Unsupported label shape: {y_shape}")

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Iterative attack
    for _ in range(T):
        with tf.GradientTape() as tape:
            tape.watch(x_star)
            logits = model(x_star)
            loss = loss_object(y, logits)

        # Compute gradient
        grad = tape.gradient(loss, x_star)

        # Normalize by L1 norm
        grad_norm = tf.reduce_sum(
            tf.abs(grad),
            axis=list(range(1, len(grad.shape))),
            keepdims=True
        )
        grad_norm = tf.maximum(grad_norm, 1e-8)
        normalized_grad = grad / grad_norm

        # Momentum update
        g = mu * g + normalized_grad

        # Perturbation step
        x_star = x_star + alpha * tf.sign(g)

        # Project back into epsilon-ball
        x_star = tf.clip_by_value(x_star, x - epsilon, x + epsilon)

        # Clip to valid pixel range
        x_star = tf.clip_by_value(x_star, 0.0, 255.0)

    return x_star

