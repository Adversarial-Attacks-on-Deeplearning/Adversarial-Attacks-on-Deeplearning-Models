import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def compute_distance(x1, x2, norm="l2"):
    """
    Compute distance between two images.
    """
    if norm == "l2":
        return np.linalg.norm(x1.flatten() - x2.flatten())
    elif norm == "linf":
        return np.max(np.abs(x1 - x2))
    else:
        raise ValueError("Unsupported norm type. Use 'l2' or 'linf'.")

def init_adv_hopskip(image, model, max_attempts=20):
    """
    Generates the hopskip adversarial image with large perturbation
    @param image: input image
    @param model: model
    @param max_attempts: maximum number of attempts to generate adversarial example
    return: adversarial image
    """
    INPUT_MIN = 0
    INPUT_MAX = 255

    original_pred = model.predict(tf.expand_dims(image, axis=0), verbose=0)
    original_label = np.argmax(original_pred, axis=1)[0]
    print(f"Original label: {original_label}, confidence: {np.max(original_pred):.4f}")

    # Start with larger noise to ensure we find an adversarial example
    noise_factor = 0.2
    max_noise_factor = 1.0
    noise_step = 0.1
    
    for attempt in range(max_attempts):
        # Generate random noise
        noise = np.random.uniform(-INPUT_MAX, INPUT_MAX, image.shape)
        x_adv = image + noise_factor * noise
        
        # Clip the image to be in the valid range
        x_adv = np.clip(x_adv, INPUT_MIN, INPUT_MAX)
        
        # Check if adversarial
        adv_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
        adv_label = np.argmax(adv_pred, axis=1)[0]
        
        if adv_label != original_label:
            print(f"Found initial adversarial example (attempt {attempt+1}):")
            print(f"  Adversarial label: {adv_label}, confidence: {np.max(adv_pred):.4f}")
            print(f"  Initial distance: {compute_distance(image, x_adv, 'l2'):.4f}")
            return x_adv
        
        # Increase noise if not successful
        if (attempt + 1) % 5 == 0 and noise_factor < max_noise_factor:
            noise_factor += noise_step
            print(f"Increasing noise factor to {noise_factor}")
    
    # If we reach here, we failed to find an adversarial example
    # Try a targeted approach with uniform direction
    print("Trying targeted approach with uniform perturbation...")
    
    # Try uniform perturbation in different directions
    directions = [
        np.ones_like(image),  # All positive
        -np.ones_like(image),  # All negative
        np.random.choice([-1, 1], size=image.shape)  # Random direction
    ]
    
    for direction in directions:
        for strength in np.linspace(0.1, 1.0, 10):  # Try stronger perturbations
            x_adv = image + strength * INPUT_MAX * direction
            x_adv = np.clip(x_adv, INPUT_MIN, INPUT_MAX)
            
            adv_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
            adv_label = np.argmax(adv_pred, axis=1)[0]
            
            if adv_label != original_label:
                print(f"Found adversarial example with uniform perturbation:")
                print(f"  Adversarial label: {adv_label}, confidence: {np.max(adv_pred):.4f}")
                print(f"  Initial distance: {compute_distance(image, x_adv, 'l2'):.4f}")
                return x_adv
    
    print("Warning: Failed to generate initial adversarial example")
    # Last resort - return a modified example that's likely to be misclassified
    return (image + 0.3 * INPUT_MAX * np.random.normal(0, 1, image.shape)).clip(INPUT_MIN, INPUT_MAX)

def estimate_gradient(original_image, adv_image, model, delta, batch_size, norm="l2"):
    """
    Estimate gradient direction using finite differences.
    
    Args:
        original_image: The original image (numpy array).
        adv_image: Current adversarial image (numpy array).
        model: The target model.
        delta: Perturbation size for gradient estimation.
        batch_size: Number of random directions to sample.
        norm: Distance norm ("l2" or "linf").
    
    Returns:
        grad_estimate: Estimated gradient direction (normalized).
    """
    # Set minimum delta to avoid numerical issues
    delta = max(delta, 0.01)
    
    # Get original label
    original_pred = model.predict(tf.expand_dims(original_image, axis=0), verbose=0)
    original_label = np.argmax(original_pred, axis=1)[0]
    
    # Get adversarial label
    adv_pred = model.predict(tf.expand_dims(adv_image, axis=0), verbose=0)
    adv_label = np.argmax(adv_pred, axis=1)[0]
    
    if adv_label == original_label:
        print("Warning: Current example is not adversarial in estimate_gradient!")
        # Return a random direction as fallback
        random_dir = np.random.normal(0, 1, original_image.shape)
        if norm == "l2":
            random_dir /= np.linalg.norm(random_dir.flatten()) + 1e-10
        else:  # linf
            random_dir = np.sign(random_dir)
        return random_dir
    
    # Generate random noise
    if norm == "l2":
        # Generate noise on a unit sphere
        noise_shape = [batch_size] + list(original_image.shape)
        noise = np.random.randn(*noise_shape).astype(np.float32)
        for i in range(batch_size):
            noise[i] = noise[i] / np.linalg.norm(noise[i].flatten())
    else:  # linf
        noise = np.random.uniform(-1, 1, [batch_size] + list(original_image.shape)).astype(np.float32)
        noise = np.sign(noise)
    
    # Create perturbed samples
    perturbed_samples = np.clip(adv_image + delta * noise, 0, 255)
    
    # Get predictions
    preds = model.predict(perturbed_samples, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    # Check which perturbations stay adversarial (0) or return to original class (1)
    f_val = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        if pred_labels[i] != original_label:  # Still adversarial
            f_val[i] = -1.0  # Points toward decision boundary
        else:  # No longer adversarial
            f_val[i] = 1.0   # Points away from decision boundary
    
    # Reshape f_val for multiplication
    f_val = f_val.reshape(-1, 1, 1, 1)
    
    # If all samples are adversarial or all are not, use a different approach
    if np.mean(f_val) == -1.0:  # All adversarial
        return -np.mean(noise, axis=0)  # Move toward decision boundary
    elif np.mean(f_val) == 1.0:  # None adversarial
        return np.mean(noise, axis=0)   # Move away from decision boundary
    
    # Compute gradient estimate
    f_val -= np.mean(f_val)
    grad_estimate = np.mean(f_val * noise, axis=0)
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad_estimate.flatten())
    if grad_norm > 1e-10:
        if norm == "l2":
            grad_estimate /= grad_norm
        else:  # linf
            grad_estimate = np.sign(grad_estimate)
    else:
        print("Warning: Gradient too small, using random direction")
        if norm == "l2":
            grad_estimate = np.random.normal(0, 1, original_image.shape)
            grad_estimate /= np.linalg.norm(grad_estimate.flatten())
        else:
            grad_estimate = np.sign(np.random.uniform(-1, 1, original_image.shape))
    
    return grad_estimate

def geometric_step_search(x_adv, grad, x_orig, model, epsilon, t, norm="l2", max_trials=10):
    """
    Geometric step search to move away from decision boundary while reducing distance.
    """
    # Get original prediction
    original_pred = model.predict(tf.expand_dims(x_orig, axis=0), verbose=0)
    original_label = np.argmax(original_pred, axis=1)[0]
    
    # Get current adversarial prediction
    adv_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
    adv_label = np.argmax(adv_pred, axis=1)[0]
    
    # Ensure we're working with an adversarial example
    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in geometric_step_search!")
        return x_adv  # Return unchanged if not adversarial
    
    # Calculate initial step size (decreasing over iterations)
    init_step_size = min(0.5, epsilon / (t + 1))
    
    step_size = init_step_size
    best_candidate = x_adv.copy()
    best_dist = compute_distance(x_adv, x_orig, norm)
    
    print(f"  Geometric step search with initial step size: {step_size:.4f}")
    
    # Try multiple step sizes
    step_multipliers = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    for multiplier in step_multipliers:
        step_size = init_step_size * multiplier
        
        # Move along gradient (should be away from decision boundary, toward original)
        x_candidate = x_adv + step_size * grad  # Note: Using + because grad points toward original
        
        # Ensure it's within valid range 
        x_candidate = np.clip(x_candidate, 0, 255)
        
        # Check if still adversarial
        candidate_pred = model.predict(tf.expand_dims(x_candidate, axis=0), verbose=0)
        candidate_label = np.argmax(candidate_pred, axis=1)[0]
        
        if candidate_label != original_label:  # Still adversarial
            # Check if distance improved
            candidate_dist = compute_distance(x_candidate, x_orig, norm)
            print(f"    Step size {step_size:.4f}: Still adversarial, distance: {candidate_dist:.4f}")
            
            if candidate_dist < best_dist:
                best_candidate = x_candidate.copy()
                best_dist = candidate_dist
                print(f"    Found better candidate at distance: {best_dist:.4f}")
        else:
            print(f"    Step size {step_size:.4f}: Not adversarial anymore")
    
    # If we found a better candidate, return it
    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(f"  Geometric search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_candidate
    else:
        print("  No improvement from geometric search")
        return x_adv

def binary_search_projection(x_adv, x_orig, epsilon, model, norm="l2", max_trials=20):
    """
    Binary search to find point on decision boundary.
    """
    # Get original prediction
    original_pred = model.predict(tf.expand_dims(x_orig, axis=0), verbose=0)
    original_label = np.argmax(original_pred, axis=1)[0]
    
    # Get current adversarial prediction
    adv_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
    adv_label = np.argmax(adv_pred, axis=1)[0]
    
    # Ensure we're working with an adversarial example
    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in binary_search_projection!")
        return x_adv  # Return unchanged if not adversarial
    
    print("  Starting binary search projection")
    
    # Binary search parameters
    low = 0.0
    high = 1.0
    threshold = 0.001  # Stopping threshold
    
    best_adv = x_adv.copy()
    best_dist = compute_distance(x_adv, x_orig, norm)
    
    # Binary search loop
    for i in range(max_trials):
        # Calculate midpoint
        mid = (low + high) / 2.0
        
        # Interpolate between original and adversarial example
        x_mid = (1 - mid) * x_orig + mid * x_adv
        x_mid = np.clip(x_mid, 0, 255)
        
        # Check if interpolated point is adversarial
        mid_pred = model.predict(tf.expand_dims(x_mid, axis=0), verbose=0)
        mid_label = np.argmax(mid_pred, axis=1)[0]
        
        if mid_label != original_label:  # Still adversarial
            # This is a valid adversarial example, can move closer to original
            high = mid
            
            # Check if better than current best
            mid_dist = compute_distance(x_mid, x_orig, norm)
            if mid_dist < best_dist:
                best_adv = x_mid.copy()
                best_dist = mid_dist
                print(f"    Trial {i+1}: Better adversarial example found with α={mid:.4f}, distance: {best_dist:.4f}")
        else:  # Not adversarial
            # Need to move farther from original
            low = mid
            print(f"    Trial {i+1}: Not adversarial at α={mid:.4f}")
        
        # Check convergence
        if high - low < threshold:
            print(f"    Binary search converged after {i+1} iterations")
            break
    
    # Return best adversarial example found
    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(f"  Binary search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_adv
    else:
        print("  No improvement from binary search")
        return x_adv

def hop_skip_jump_attack(model, x_orig, epsilon=None, delta=0.1, batch_size=100, norm="l2", max_queries=5000, max_iters=40):
    """
    Implements the HopSkipJump attack algorithm.
    
    Args:
        model: The target model.
        x_orig: Original image to attack.
        epsilon: Maximum perturbation size (if None, no constraint).
        delta: Initial perturbation size for gradient estimation.
        batch_size: Number of samples for gradient estimation.
        norm: Distance norm ("l2" or "linf").
        max_queries: Maximum number of model queries.
        max_iters: Maximum number of iterations.
    
    Returns:
        x_adv: Adversarial example.
    """
    query_count = 0
    original_pred = model.predict(tf.expand_dims(x_orig, axis=0), verbose=0)
    original_label = np.argmax(original_pred, axis=1)[0]
    query_count += 1
    
    print(f"Starting attack (original label: {original_label}, confidence: {np.max(original_pred):.4f})")
    
    # Initialize adversarial example
    x_adv = init_adv_hopskip(x_orig, model)
    query_count += 20  # Approximate queries for initialization
    
    adv_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
    adv_label = np.argmax(adv_pred, axis=1)[0]
    query_count += 1
    
    if adv_label == original_label:
        print("Failed to initialize adversarial example.")
        return x_orig  # Return original if initialization fails
    
    initial_dist = compute_distance(x_adv, x_orig, norm)
    print(f"Initial distance: {initial_dist:.4f}")
    
    # Track best adversarial example
    x_best = x_adv.copy()
    best_dist = initial_dist
    
    # Early stop counters
    no_improvement_counter = 0
    max_no_improvement = 5
    
    # If epsilon is not specified, set it to the initial distance
    if epsilon is None:
        epsilon = initial_dist
    
    # Main attack loop
    for t in range(max_iters):
        if query_count >= max_queries:
            print(f"Query limit reached: {query_count}/{max_queries}")
            break
        
        dist = compute_distance(x_adv, x_orig, norm)
        print(f"\nIteration {t+1}, current distance: {dist:.4f}, queries: {query_count}/{max_queries}")
        
        # Verify adversarial status
        current_pred = model.predict(tf.expand_dims(x_adv, axis=0), verbose=0)
        current_label = np.argmax(current_pred, axis=1)[0]
        query_count += 1
        
        if current_label == original_label:
            print("Warning: Current example is no longer adversarial! Reverting to best known.")
            x_adv = x_best.copy()
            continue
        
        # Adaptive delta based on current distance
        current_delta = min(delta, dist / 10)
        
        # Increase delta if we're stuck
        if no_improvement_counter >= 3:
            current_delta *= 2.0
            print(f"No recent improvement, increasing delta to {current_delta:.4f}")
        
        print(f"Estimating gradient with delta={current_delta:.4f}, batch_size={batch_size}")
        grad = estimate_gradient(x_orig, x_adv, model, current_delta, batch_size, norm)
        query_count += batch_size
        
        # Step 1: Move away from decision boundary (geometric step)
        x_candidate = geometric_step_search(x_adv, grad, x_orig, model, epsilon, t, norm)
        query_count += 10  # Approximate queries for geometric search
        
        # Step 2: Binary search to find point close to decision boundary
        x_refined = binary_search_projection(x_candidate, x_orig, epsilon, model, norm)
        query_count += 10  # Approximate queries for binary search
        
        # Check distance improvement
        refined_dist = compute_distance(x_refined, x_orig, norm)
        
        if refined_dist < dist:
            print(f"Iteration {t+1} improved distance: {dist:.4f} -> {refined_dist:.4f}")
            x_adv = x_refined.copy()
            
            # Update best example if better
            if refined_dist < best_dist:
                x_best = x_refined.copy()
                best_dist = refined_dist
                print(f"New best distance: {best_dist:.4f}")
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            print(f"No distance improvement in iteration {t+1}")
            no_improvement_counter += 1
            
            # Random restart if stuck
            if no_improvement_counter >= max_no_improvement:
                print("No improvement for multiple iterations, trying random restart...")
                # Add small random perturbation to current best
                noise_scale = best_dist / 10
                noise = np.random.normal(0, noise_scale, x_best.shape)
                if norm == "l2":
                    noise = noise / np.linalg.norm(noise.flatten()) * noise_scale
                else:  # linf
                    noise = np.sign(noise) * noise_scale
                
                x_perturbed = np.clip(x_best + noise, 0, 255)
                
                # Verify it's still adversarial
                perturb_pred = model.predict(tf.expand_dims(x_perturbed, axis=0), verbose=0)
                perturb_label = np.argmax(perturb_pred, axis=1)[0]
                
                if perturb_label != original_label:
                    print("Random restart successful")
                    x_adv = x_perturbed.copy()
                    no_improvement_counter = 0
                else:
                    # Try in opposite direction
                    x_perturbed = np.clip(x_best - noise, 0, 255)
                    perturb_pred = model.predict(tf.expand_dims(x_perturbed, axis=0), verbose=0)
                    perturb_label = np.argmax(perturb_pred, axis=1)[0]
                    
                    if perturb_label != original_label:
                        print("Random restart in opposite direction successful")
                        x_adv = x_perturbed.copy()
                        no_improvement_counter = 0
                    else:
                        print("Random restart failed, continuing with best known")
                
    # Final verification
    final_pred = model.predict(tf.expand_dims(x_best, axis=0), verbose=0)
    final_label = np.argmax(final_pred, axis=1)[0]
    final_dist = compute_distance(x_best, x_orig, norm)
    
    print(f"\nAttack completed:")
    print(f"  Original label: {original_label}, confidence: {np.max(original_pred):.4f}")
    print(f"  Final label: {final_label}, confidence: {np.max(final_pred):.4f}")
    print(f"  Distance: {final_dist:.4f}")
    print(f"  Queries: {query_count}")
    
    return x_best

def test():
    # Load the model
    model = load_model('TrafficEffNet.keras')

    # Load the image
    image = load_img('00000_00000_00000.png', target_size=(240, 240))
    image = img_to_array(image)

    # Parameters
    epsilon = None  # Let the algorithm find the minimum distortion
    delta = 1.0  # Initial step size for gradient estimation
    batch_size = 100  # Number of samples for gradient estimation
    norm = "l2"  # Norm type (l2 or linf)
    max_queries = 5000  # Maximum model queries
    
    # Run attack
    print("\n=== Starting HopSkipJump Attack ===\n")
    adv_image = hop_skip_jump_attack(model, image, epsilon, delta, batch_size, norm, max_queries)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Adversarial Image")
    plt.imshow(adv_image.astype(np.uint8))
    plt.axis('off')
    
    # Show the difference (magnified for visibility)
    diff = np.abs(adv_image - image)   # Magnify difference by 10x for visibility
    diff = np.clip(diff, 0, 255)
    
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(diff.astype(np.uint8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hopskipjump_result.png')
    plt.show()
    
    # Print detailed classification results
    orig_pred = model.predict(tf.expand_dims(image, axis=0))
    orig_label = np.argmax(orig_pred, axis=1)[0]
    orig_conf = np.max(orig_pred)
    
    adv_pred = model.predict(tf.expand_dims(adv_image, axis=0))
    adv_label = np.argmax(adv_pred, axis=1)[0]
    adv_conf = np.max(adv_pred)
    
    print("\n=== Classification Results ===")
    print(f"Original: Class {orig_label} with {orig_conf:.4f} confidence")
    print(f"Adversarial: Class {adv_label} with {adv_conf:.4f} confidence")
    print(f"L2 Distance: {compute_distance(image, adv_image, 'l2'):.4f}")
    
    # Save adversarial image
    plt.imsave('adversarial.png', adv_image.astype(np.uint8))
    print("Adversarial image saved as 'adversarial.png'")

if __name__ == "__main__":
    test()