import tensorflow as tf
import numpy as np

def sparsefool_attack(
    model,
    img_input,
    num_classes,
    max_iter=50,
    top_k=10,
    epsilon=20.0,
    clip_min=0.0,
    clip_max=255.0,
    verbose=True
):
    img_input = tf.convert_to_tensor(img_input, dtype=tf.float32)
    adv_image = tf.Variable(img_input, trainable=True)  # Shape: [1, H, W, C]
    original_image = tf.identity(img_input)

    # Get original prediction
    original_logits = model(original_image)
    original_class = tf.argmax(original_logits[0]).numpy()

    if verbose:
        print(f"Initial prediction: Class {original_class}")

    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            logits = model(adv_image)

        current_class = tf.argmax(logits[0]).numpy()
        if verbose:
            print(f"Iteration {iteration}: Current class = {current_class}")

        # Check for misclassification
        if current_class != original_class:
            if verbose:
                print(f"Attack succeeded at iteration {iteration}!")
            break

        # Compute gradients of ALL logits w.r.t. the input
        grads = tape.jacobian(logits, adv_image)  # Shape: [1, num_classes, 1, H, W, C]
        grads = tf.squeeze(grads, axis=[0, 2])    # Shape: [num_classes, H, W, C]

        # Get gradients for the CURRENT predicted class
        current_class_grad = grads[current_class]  # Shape: [H, W, C]
        grad_abs = tf.abs(current_class_grad)

        # Flatten and find top-k pixel indices
        flat_grad = tf.reshape(grad_abs, [-1])
        _, top_indices = tf.math.top_k(flat_grad, k=top_k)

        # Create a perturbation mask for the top-k pixels
        mask = tf.scatter_nd(
            indices=tf.expand_dims(top_indices, axis=1),
            updates=tf.ones(top_k, dtype=tf.float32),
            shape=tf.shape(flat_grad)
        )
        mask = tf.reshape(mask, grad_abs.shape)  # Shape: [H, W, C]

        # Perturb the selected pixels by ±epsilon (random sign)
        random_sign = tf.sign(tf.random.normal(mask.shape))  # ±1 randomly
        perturbation = mask * epsilon * random_sign

        # Add batch dimension to perturbation to match adv_image's shape [1, H, W, C]
        perturbation = tf.expand_dims(perturbation, axis=0)  # <--- FIX HERE

        # Update adversarial image
        adv_image.assign_add(perturbation)
        adv_image.assign(tf.clip_by_value(adv_image, clip_min, clip_max))

    if verbose and current_class == original_class:
        print("Attack failed to misclassify within max iterations.")

    return adv_image.numpy()