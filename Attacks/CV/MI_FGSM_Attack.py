
def MI_fgsm_single_image(model, image, label, epsilon, iterations, mu):
    """
    Perform MI_FGSM attack on a single image without normalization.
    
    Args:
        model: Trained model.
        image: Input image of shape (C, H, W), Normalized.
        label: True target mask.
        epsilon (float): Perturbation magnitude.
        iterations: Number of iterations
        mu: decay factor of momentum

    Returns:
        Adversarial example (Normalized).
    """
    
    model.eval()
    # get step size per itertion
    alpha = epsilon / iterations
    
    # Initialize adversarial image
    adversarial_image = image.clone().detach().requires_grad_(True)
    
    # Define the velocity vector
    g = torch.zeros_like(adversarial_image)
    for t in range(iterations):
        model.zero_grad()

        # Forward pass
        prediction = model(adversarial_image.unsqueeze(0))
        # Calculate loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(prediction.squeeze(0), label.unsqueeze(0))
        # Backpropagate
        loss.backward()
        
        grad = adversarial_image.grad.data
        grad = grad / torch.abs(grad).sum()  # Apply L1 norm on gradient
        
        # calculate momentum
        g = mu * g + grad

        # Get the sign of the gradient and update the adversarial image
        adversarial_image = adversarial_image + alpha * torch.sign(g)

        # Project the adversarial image into the epsilon-ball and clip to [0, 1]
        adversarial_image = torch.max(torch.min(adversarial_image, image + epsilon), image - epsilon)
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # Detach and re-enable gradient computation
        adversarial_image = adversarial_image.detach().requires_grad_(True)
        print(loss.item())
    

    return adversarial_image.squeeze(0)
