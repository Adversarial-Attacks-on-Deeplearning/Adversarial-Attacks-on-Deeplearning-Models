import numpy as np
import tensorflow as tf
from PIL import Image
from GTSRB_utils import GTSRB_CLASSES, load_ppm_image, predict_traffic_sign

def simba_attack(image_tensor, true_label, model, num_iters=2000, epsilon=0.2, print_every=10):
    """
    image_tensor: TensorFlow tensor of shape (1, H, W, 3) in [0, 255]
    true_label: Integer label (the true class of the image)
    model: TensorFlow/Keras model used for prediction
    num_iters: Maximum number of iterations (queries)
    epsilon: Perturbation magnitude for each pixel update
    print_every: Frequency (in iterations) at which to print detailed logs
    """
    # Adapt epsilon value for models that accept unnormalized images
    epsilon = epsilon * 255
    
    # Convert image tensor to numpy array for manipulation
    adv = image_tensor.numpy()  # shape: (1, H, W, 3)
    H, W, C = adv.shape[1], adv.shape[2], adv.shape[3]
    n_dims = H * W * C

    # Generate a random permutation of pixel indices
    perm = np.random.permutation(n_dims)

   # Get the initial probability for the true class
    pred = model.predict(adv)
    true_prob = pred[0][true_label]
    print("Initial true class probability for label {}: {:.4f}".format(true_label, true_prob))

    
    # Check initial prediction (if already flipped, no need to attack)
    current_pred = np.argmax(pred[0])
    if current_pred != true_label:
        print("Initial prediction is already different from the true label.")
        return tf.convert_to_tensor(adv, dtype=tf.float32)

    # Loop over the maximum number of iterations
    for i in range(num_iters):
        # Wrap around the permutation if i >= n_dims
        idx = perm[i % n_dims]
        # Convert flat index to (row, col, channel)
        channel = idx % C
        temp = idx // C
        col = temp % W
        row = temp // W

        # Create a perturbation only at the selected pixel and channel
        perturb = np.zeros_like(adv)
        perturb[0, row, col, channel] = epsilon

        # Try positive perturbation
        adv_pos = np.clip(adv + perturb, 0, 255)
        prob_pos = model.predict(adv_pos)
        pos_true_prob = prob_pos[0][true_label]
        diff_pos = pos_true_prob - true_prob

        # For an untargeted attack, we want to lower the confidence of the true class
        if pos_true_prob < true_prob:
            adv = adv_pos
            print(f"Iteration {i}: +epsilon -> pixel=({row},{col},{channel}), "
                  f"old_prob={true_prob:.6f}, new_prob={pos_true_prob:.6f}, diff={diff_pos:.6f}")
            true_prob = pos_true_prob
        else:
            # Try negative perturbation if positive did not help
            adv_neg = np.clip(adv - perturb, 0, 255)
            prob_neg = model.predict(adv_neg)
            neg_true_prob = prob_neg[0][true_label]
            diff_neg = neg_true_prob - true_prob

            if neg_true_prob < true_prob:
                adv = adv_neg
                print(f"Iteration {i}: -epsilon -> pixel=({row},{col},{channel}), "
                      f"old_prob={true_prob:.6f}, new_prob={neg_true_prob:.6f}, diff={diff_neg:.6f}")
                true_prob = neg_true_prob
            else:
                # Optionally print status even if no update occurred
                if (i % print_every) == 0:
                    print(f"Iteration {i}: No update at pixel=({row},{col},{channel}). "
                          f"pos_diff={diff_pos:.6f}, neg_diff={diff_neg:.6f}, current_prob={true_prob:.6f}")

        # Check if the true class is no longer the highest predicted
        current_pred = np.argmax(model.predict(adv)[0])
        if current_pred != true_label:
            print(f"Attack successful at iteration {i}: predicted class flipped to {current_pred} != true label {true_label}")
            break

    print("Attack finished after {} iterations. Final true class probability: {:.4f}".format(i+1, true_prob))
    # Return the adversarial image as a TensorFlow tensor
    adv_tensor = tf.convert_to_tensor(adv, dtype=tf.float32)
    return adv_tensor
