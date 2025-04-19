
import sys
import io
import torch



def compute_distance(x1, x2, norm="l2"):
    """
    Compute distance between two images using PyTorch tensors.
    """
    if norm == "l2":
        return torch.norm(x1.flatten() - x2.flatten()).item()
    elif norm == "linf":
        return torch.max(torch.abs(x1 - x2)).item()
    else:
        raise ValueError("Unsupported norm type. Use 'l2' or 'linf'.")

def get_top_class(output):
    """
    Extract the top class from YOLO output. Assumes output is a list of detections.
    """
    if len(output[0].boxes) == 0:
        return -1  # No detections
    conf, cls = output[0].boxes.conf.max(), output[0].boxes.cls[output[0].boxes.conf.argmax()]
    return cls.item()

def get_confidence(output, label):
    """
    Get the confidence of the specified label from YOLO output.
    """
    if len(output[0].boxes) == 0:
        return 0.0
    conf = output[0].boxes.conf[output[0].boxes.cls == label].max()
    return conf.item() if conf.numel() > 0 else 0.0



# Define the SuppressPrints class
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

# Helper function to query the model with suppressed prints
def query_model(model, image):
    with SuppressPrints():
        output = model(image)
    return output

def init_adv_hopskip(image, model, max_attempts=20):
    """
    Initialize an adversarial example with large perturbation for YOLO.
    """
    INPUT_MIN = 0.0
    INPUT_MAX = 1.0

    original_output = query_model(model, image)
    original_label = get_top_class(original_output)
    original_conf = get_confidence(original_output, original_label)
    print(f"Original label: {original_label}, confidence: {original_conf:.4f}")

    noise_factor = 0.2
    max_noise_factor = 1.0
    noise_step = 0.1

    for attempt in range(max_attempts):
        noise = torch.rand_like(image) * 2 - 1  # Random noise in [-1, 1]
        x_adv = image + noise_factor * noise
        x_adv = torch.clamp(x_adv, INPUT_MIN, INPUT_MAX)

        adv_output = query_model(model, x_adv)
        adv_label = get_top_class(adv_output)

        if adv_label != original_label:
            print(f"Found initial adversarial example (attempt {attempt+1}):")
            print(f"  Adversarial label: {adv_label}, confidence: {get_confidence(adv_output, adv_label):.4f}")
            print(f"  Initial distance: {compute_distance(image, x_adv, 'l2'):.4f}")
            return x_adv

        if (attempt + 1) % 5 == 0 and noise_factor < max_noise_factor:
            noise_factor += noise_step
            print(f"Increasing noise factor to {noise_factor}")

    print("Warning: Failed to generate initial adversarial example")
    return image

def estimate_gradient(original_image, adv_image, model, delta, batch_size, norm="l2"):
    """
    Estimate gradient direction using finite differences for YOLO.
    """
    delta = max(delta, 0.01)
    original_label = get_top_class(query_model(model, original_image))
    adv_label = get_top_class(query_model(model, adv_image))

    if adv_label == original_label:
        print("Warning: Current example is not adversarial in estimate_gradient!")
        random_dir = torch.randn_like(original_image)
        if norm == "l2":
            random_dir /= torch.norm(random_dir.flatten()) + 1e-10
        else:
            random_dir = torch.sign(random_dir)
        return random_dir

    noise_shape = (batch_size,) + original_image.shape[1:]
    if norm == "l2":
        noise = torch.randn(noise_shape).to(original_image.device)
        noise = noise / torch.norm(noise.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
    else:
        noise = torch.sign(torch.randn(noise_shape).to(original_image.device))

    perturbed_samples = torch.clamp(adv_image + delta * noise, 0, 1)
    perturbed_outputs = [query_model(model, sample.unsqueeze(0)) for sample in perturbed_samples]
    f_val = torch.tensor([1.0 if get_top_class(output) != original_label else -1.0 for output in perturbed_outputs])

    if torch.all(f_val == -1.0):
        return -torch.mean(noise, dim=0)
    elif torch.all(f_val == 1.0):
        return torch.mean(noise, dim=0)

    f_val -= torch.mean(f_val)
    grad_estimate = torch.mean(f_val.view(batch_size, 1, 1, 1) * noise, dim=0)

    grad_norm = torch.norm(grad_estimate.flatten())
    if grad_norm > 1e-10:
        if norm == "l2":
            grad_estimate /= grad_norm
        else:
            grad_estimate = torch.sign(grad_estimate)
    else:
        grad_estimate = torch.randn_like(grad_estimate)
        if norm == "l2":
            grad_estimate /= torch.norm(grad_estimate.flatten())
        else:
            grad_estimate = torch.sign(grad_estimate)

    return grad_estimate

def geometric_step_search(x_adv, grad, x_orig, model, epsilon, t, norm="l2", max_trials=10):
    """
    Geometric step search to refine the adversarial example.
    """
    original_label = get_top_class(query_model(model, x_orig))
    adv_label = get_top_class(query_model(model, x_adv))

    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in geometric_step_search!")
        return x_adv

    init_step_size = min(0.5, epsilon / (t + 1))
    step_size = init_step_size
    best_candidate = x_adv.clone()
    best_dist = compute_distance(x_adv, x_orig, norm)

    step_multipliers = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    for multiplier in step_multipliers:
        step_size = init_step_size * multiplier
        x_candidate = x_adv + step_size * grad
        x_candidate = torch.clamp(x_candidate, 0, 1)

        candidate_output = query_model(model, x_candidate)
        candidate_label = get_top_class(candidate_output)

        if candidate_label != original_label:
            candidate_dist = compute_distance(x_candidate, x_orig, norm)
            print(f"    Step size {step_size:.4f}: Still adversarial, distance: {candidate_dist:.4f}")
            if candidate_dist < best_dist:
                best_candidate = x_candidate.clone()
                best_dist = candidate_dist
                print(f"    Found better candidate at distance: {best_dist:.4f}")

    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(f"  Geometric search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_candidate
    else:
        print("  No improvement from geometric search")
        return x_adv

def binary_search_projection(x_adv, x_orig, epsilon, model, norm="l2", max_trials=20):
    """
    Binary search to project the adversarial example closer to the original.
    """
    original_label = get_top_class(query_model(model, x_orig))
    adv_label = get_top_class(query_model(model, x_adv))

    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in binary_search_projection!")
        return x_adv

    print("  Starting binary search projection")

    low = 0.0
    high = 1.0
    threshold = 0.001

    best_adv = x_adv.clone()
    best_dist = compute_distance(x_adv, x_orig, norm)

    for i in range(max_trials):
        mid = (low + high) / 2.0
        x_mid = (1 - mid) * x_orig + mid * x_adv
        x_mid = torch.clamp(x_mid, 0, 1)

        mid_output = query_model(model, x_mid)
        mid_label = get_top_class(mid_output)

        if mid_label != original_label:
            high = mid
            mid_dist = compute_distance(x_mid, x_orig, norm)
            if mid_dist < best_dist:
                best_adv = x_mid.clone()
                best_dist = mid_dist
                print(f"    Trial {i+1}: Better adversarial example found with α={mid:.4f}, distance: {best_dist:.4f}")
        else:
            low = mid

        if high - low < threshold:
            print(f"    Binary search converged after {i+1} iterations")
            break

    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(f"  Binary search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_adv
    else:
        print("  No improvement from binary search")
        return x_adv

def hop_skip_jump_attack(model, x_orig, epsilon=None, delta=0.1, batch_size=100, norm="l2", max_queries=5000, max_iters=40):
    """
    HopSkipJump attack implementation for PyTorch YOLO model with suppressed model prints.
    """
    query_count = 0
    original_output = query_model(model, x_orig)
    original_label = get_top_class(original_output)
    query_count += 1

    print(f"Starting attack (original label: {original_label}, confidence: {get_confidence(original_output, original_label):.4f})")

    x_adv = init_adv_hopskip(x_orig, model)
    query_count += 20  # Approximate queries for initialization

    adv_output = query_model(model, x_adv)
    adv_label = get_top_class(adv_output)
    query_count += 1

    if adv_label == original_label:
        print("Failed to initialize adversarial example.")
        return x_orig

    initial_dist = compute_distance(x_adv, x_orig, norm)
    print(f"Initial distance: {initial_dist:.4f}")

    x_best = x_adv.clone()
    best_dist = initial_dist

    no_improvement_counter = 0
    max_no_improvement = 5

    if epsilon is None:
        epsilon = initial_dist

    for t in range(max_iters):
        if query_count >= max_queries:
            print(f"Query limit reached: {query_count}/{max_queries}")
            break

        dist = compute_distance(x_adv, x_orig, norm)
        print(f"\nIteration {t+1}, current distance: {dist:.4f}, queries: {query_count}/{max_queries}")

        current_output = query_model(model, x_adv)
        current_label = get_top_class(current_output)
        query_count += 1

        if current_label == original_label:
            print("Warning: Current example is no longer adversarial! Reverting to best known.")
            x_adv = x_best.clone()
            continue

        current_delta = min(delta, dist / 10)

        if no_improvement_counter >= 3:
            current_delta *= 2.0
            print(f"No recent improvement, increasing delta to {current_delta:.4f}")

        print(f"Estimating gradient with delta={current_delta:.4f}, batch_size={batch_size}")
        grad = estimate_gradient(x_orig, x_adv, model, current_delta, batch_size, norm)
        query_count += batch_size

        x_candidate = geometric_step_search(x_adv, grad, x_orig, model, epsilon, t, norm)
        query_count += 10  # Approximate

        x_refined = binary_search_projection(x_candidate, x_orig, epsilon, model, norm)
        query_count += 10  # Approximate

        refined_dist = compute_distance(x_refined, x_orig, norm)

        if refined_dist < dist:
            print(f"Iteration {t+1} improved distance: {dist:.4f} -> {refined_dist:.4f}")
            x_adv = x_refined.clone()
            if refined_dist < best_dist:
                x_best = x_refined.clone()
                best_dist = refined_dist
                print(f"New best distance: {best_dist:.4f}")
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            print(f"No distance improvement in iteration {t+1}")
            no_improvement_counter += 1

            if no_improvement_counter >= max_no_improvement:
                print("No improvement for multiple iterations, trying random restart...")
                noise_scale = best_dist / 10
                noise = torch.randn_like(x_best) * noise_scale
                if norm == "l2":
                    noise = noise / torch.norm(noise.flatten()) * noise_scale
                else:
                    noise = torch.sign(noise) * noise_scale

                x_perturbed = torch.clamp(x_best + noise, 0, 1)
                perturb_output = query_model(model, x_perturbed)
                perturb_label = get_top_class(perturb_output)

                if perturb_label != original_label:
                    print("Random restart successful")
                    x_adv = x_perturbed.clone()
                    no_improvement_counter = 0
                else:
                    x_perturbed = torch.clamp(x_best - noise, 0, 1)
                    perturb_output = query_model(model, x_perturbed)
                    perturb_label = get_top_class(perturb_output)
                    if perturb_label != original_label:
                        print("Random restart in opposite direction successful")
                        x_adv = x_perturbed.clone()
                        no_improvement_counter = 0

    final_output = query_model(model, x_best)
    final_label = get_top_class(final_output)
    final_dist = compute_distance(x_best, x_orig, norm)

    print(f"\nAttack completed:")
    print(f"  Original label: {original_label}, confidence: {get_confidence(original_output, original_label):.4f}")
    print(f"  Final label: {final_label}, confidence: {get_confidence(final_output, final_label):.4f}")
    print(f"  Distance: {final_dist:.4f}")
    print(f"  Queries: {query_count}")

    return x_best