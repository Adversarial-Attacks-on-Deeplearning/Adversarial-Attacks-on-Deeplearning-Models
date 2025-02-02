import numpy as np
import Utils as utils

def boundary_attack_single_image(model, original_image, adversarial_image, label, max_iter=1000, epsilon=0.01, step_size=0.01):
    """
    Perform the Boundary Attack on a single image.

    Args:
        model: The target model.
        original_image: The original input image.
        adversarial_image: The initial adversarial example.
        label: The true label of the original image.
        max_iter: Maximum number of iterations.
        epsilon: Perturbation size.
        step_size: Step size for moving closer to the original input.

    Returns:
        adversarial_image: The final adversarial example.
        min_perturbation: The minimum perturbation (distance) between the original and adversarial image.
    """
    min_perturbation = np.inf  # Initialize minimum perturbation to a large value

    for i in range(max_iter):
        # 1. Generate a random perturbation
        perturbation = np.random.normal(scale=epsilon, size=original_image.shape)
        
        # 2. Perturb the adversarial example
        perturbed_image = adversarial_image + perturbation
        perturbed_image = np.clip(perturbed_image, 0, 255)  # Clip to valid pixel range
    
        # 3. Check if the perturbed example is adversarial
        if utils.is_adversarial(model, perturbed_image, label):
            # 4. Move closer to the original input
            last_updated_image = adversarial_image
            updated_image = adversarial_image + step_size * (original_image - adversarial_image)
            updated_image = np.clip(updated_image, 0, 255)  # Clip to valid pixel range
            
            # 5. Check if the updated example is still adversarial
            if utils.is_adversarial(model, updated_image, label):
                adversarial_image = updated_image  # Update the adversarial example
                # Calculate the current perturbation (distance)
                current_perturbation = np.linalg.norm(adversarial_image - original_image)
                # Update the minimum perturbation if the current one is smaller
                if current_perturbation < min_perturbation:
                    min_perturbation = current_perturbation
            else:
                print("The minimum perturbation is found")
                return last_updated_image, min_perturbation
        else:
            print("The minimum perturbation is found")
            return last_updated_image, min_perturbation
        
        # Print progress
        if i % 1 == 0:
            distance = np.linalg.norm(adversarial_image - original_image)
            print(f"Iteration {i}: Distance to original = {distance}")

    # Return the final adversarial image and the minimum perturbation
    return adversarial_image, min_perturbation


