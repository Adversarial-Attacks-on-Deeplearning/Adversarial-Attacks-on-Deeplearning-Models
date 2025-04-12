import numpy as np
import tensorflow as tf
import scipy.fftpack

def simba_dct_attack(image_tensor, true_label, model, num_iters=2000, epsilon=0.2, print_every=10, initial_freq_frac=1/8):
    """
    Performs the SimBA-DCT adversarial attack.

    Args:
        image_tensor: Tensor of shape (1, H, W, 3) in [0, 255]
        true_label: Integer, correct label of the image
        model: Black-box model to query
        num_iters: Max number of queries
        epsilon: Step size in DCT space
        print_every: Print frequency
        initial_freq_frac: Fraction of lowest DCT frequencies to use initially

    Returns:
        adv_tensor: Final adversarial image in pixel space
    """
    epsilon_scaled = epsilon * 255

    # --- Step 1: Get DCT Coefficients of the image ---
    dct_image = rgb_image_to_dct(image_tensor)      # shape: (1, H, W, 3)
    H, W, C = dct_image.shape[1:]

    # --- Step 2: Initialize delta ---
    delta = np.zeros_like(dct_image)

    # --- Step 3: Get initial prediction ---
    pred = model.predict(image_tensor)
    true_prob = pred[0][true_label]
    print("Initial true class probability: {:.4f}".format(true_prob))

    # --- Step 4: Prepare list of DCT basis directions ---
    total_freqs = H * W * C
    num_initial_freqs = int(total_freqs * initial_freq_frac)

    # All frequency indices as (row, col, channel) tuples
    all_indices = [(i // (W * C), (i // C) % W, i % C) for i in range(total_freqs)]
    np.random.shuffle(all_indices)
    Q_dct = all_indices[:num_initial_freqs]
    current_pointer = num_initial_freqs

    queries = 0
    for i in range(num_iters):
        if len(Q_dct) == 0:
            # Exhausted current directions, add more frequencies
            additional = int(total_freqs / 32)
            next_indices = all_indices[current_pointer : current_pointer + additional]
            if not next_indices:
                print("No more directions to explore.")
                break
            Q_dct.extend(next_indices)
            current_pointer += additional

        # --- Step 5: Sample direction q ---
        row, col, channel = Q_dct.pop(np.random.randint(len(Q_dct)))

        # Create perturbation in DCT space
        perturb = np.zeros_like(delta)
        perturb[0, row, col, channel] = epsilon_scaled

        # --- Step 6: Test +epsilon ---
        dct_plus = dct_image + delta + perturb
        x_plus = dct_to_rgb_image(dct_plus)
        prob_plus = model.predict(x_plus)
        queries += 1
        p_plus = prob_plus[0][true_label]

        # --- Step 7: Test -epsilon ---
        dct_minus = dct_image + delta - perturb
        x_minus = dct_to_rgb_image(dct_minus)
        prob_minus = model.predict(x_minus)
        queries += 1
        p_minus = prob_minus[0][true_label]

        # --- Step 8: Decide best update ---
        if p_plus < true_prob:
            delta += perturb
            true_prob = p_plus
            if i % print_every == 0:
                print(f"Iter {i}: +ε at ({row},{col},{channel}), new_prob={p_plus:.6f}")
        elif p_minus < true_prob:
            delta -= perturb
            true_prob = p_minus
            if i % print_every == 0:
                print(f"Iter {i}: -ε at ({row},{col},{channel}), new_prob={p_minus:.6f}")
        elif i % print_every == 0:
            print(f"Iter {i}: No improvement at ({row},{col},{channel}), prob={true_prob:.6f}")

        # Stop if attack is successful (label flipped)
        final_pred = np.argmax(model.predict(dct_to_rgb_image(dct_image + delta))[0])
        queries += 1
        if final_pred != true_label:
            print(f"Attack succeeded at iteration {i} after {queries} queries!")
            break

    adv_image = dct_to_rgb_image(dct_image + delta)
    return adv_image
