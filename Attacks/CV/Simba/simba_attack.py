import tensorflow as tf
import numpy as np

def simba_pixel_attack_single_image_tf(
    black_box_fn,
    x,
    y,
    epsilon=0.01,
    max_queries=10000,
    clamp_min=0.0,
    clamp_max=1.0,
    dtype=tf.float32
):
    """
    Perform the SimBA attack in pixel space for a single image using TensorFlow.

    Parameters
    ----------
    black_box_fn : callable
        A function that takes a single image (Tensor of shape (H, W, C)) and
        returns a 1D Tensor of shape (num_classes,) representing predicted probabilities.
    x : tf.Tensor
        The original (clean) image of shape (H, W, C).
    y : int
        The correct (original) label index for x.
    epsilon : float
        The step size for perturbations in each direction.
    max_queries : int
        The maximum number of model queries allowed.
    clamp_min : float
        Minimum pixel value for clamping (e.g. 0.0 if model expects [0,1]).
    clamp_max : float
        Maximum pixel value for clamping (e.g. 1.0 if model expects [0,1]).
    dtype : tf.dtypes.DType
        The data type to use for computations (e.g., tf.float32).

    Returns
    -------
    perturbed_x : tf.Tensor
        The final perturbed image of shape (H, W, C).
    delta : tf.Tensor
        The final perturbation of shape (H, W, C).
    num_queries_used : int
        Number of queries used during the attack.
    """

    # Cast x to desired dtype
    x = tf.cast(x, dtype)

    # Image dimensions
    H, W, C = x.shape
    D = H * W * C  # total number of pixels

    # Initialize delta = 0 (no perturbation)
    delta = tf.zeros_like(x, dtype=dtype)

    # 1) Query the model with the original (unperturbed) image
    probs = black_box_fn(x)  # shape: (num_classes,)
    p = probs[y].numpy()     # Probability of the correct class y
    num_queries_used = 1

    # Check if x is already misclassified
    if tf.argmax(probs).numpy() != y:
        print("Original image is already misclassified. Returning.")
        return x, delta, num_queries_used

    # Create a random permutation of all pixel indices
    all_indices = np.arange(D)
    np.random.shuffle(all_indices)

    idx_ptr = 0  # pointer to the next pixel index in the random permutation

    # 2) Main loop
    while True:
        # Check if (x + delta) is misclassified
        current_probs = black_box_fn(tf.clip_by_value(x + delta, clamp_min, clamp_max))
        current_pred = tf.argmax(current_probs).numpy()
        if current_pred != y:
            # Misclassified => stop
            break

        # Check query budget
        if num_queries_used >= max_queries or idx_ptr >= D:
            # Either we've hit the query limit or tried all pixels
            break

        # Pick the next pixel index
        idx = all_indices[idx_ptr]
        idx_ptr += 1

        # Construct a direction vector q for this pixel index
        # Flatten a zero array, set q[idx] = 1, then reshape to (H, W, C).
        q_np = np.zeros((D,), dtype=np.float32)
        q_np[idx] = 1.0
        q_np = q_np.reshape((H, W, C))

        q_tf = tf.constant(q_np, dtype=dtype)

        # We'll clamp after adding the perturbation, so define a helper:
        def clamp_image(im):
            return tf.clip_by_value(im, clamp_min, clamp_max)

        # Try +epsilon
        plus_image = clamp_image(x + delta + epsilon * q_tf)
        plus_probs = black_box_fn(plus_image)
        num_queries_used += 1
        p_plus = plus_probs[y].numpy()

        if p_plus < p:
            # Confidence in the true class went down, accept +epsilon
            delta = plus_image - x
            p = p_plus
            continue

        # Otherwise, try -epsilon
        minus_image = clamp_image(x + delta - epsilon * q_tf)
        minus_probs = black_box_fn(minus_image)
        num_queries_used += 1
        p_minus = minus_probs[y].numpy()

        if p_minus < p:
            # Confidence in the true class went down, accept -epsilon
            delta = minus_image - x
            p = p_minus
            continue

        # If neither +epsilon nor -epsilon reduces p, do not update delta

        if num_queries_used >= max_queries:
            break

    # Final perturbed image
    perturbed_x = tf.clip_by_value(x + delta, clamp_min, clamp_max)

    return perturbed_x, delta, num_queries_used
