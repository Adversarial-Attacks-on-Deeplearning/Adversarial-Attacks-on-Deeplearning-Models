def compute_output_grads(model, image):
    """ Compute the element-wise grads matrix of the model output with respect to the input image. """

    image.requires_grad = True
    output = torch.sigmoid(model(image.unsqueeze(0)))  # Forward pass
    grad_outputs = torch.ones_like(output.squeeze(0))  # output size to compute element-wise gradient
    element_wise_grad = torch.autograd.grad(output.squeeze(0), image, grad_outputs=grad_outputs, create_graph = True)[0]
    return element_wise_grad
    
def saliency_map(jacobian, target_label):
    """ Compute the saliency map to determine which pixels to perturb. """
    # Compute impact of modifying each pixel
    target_grad = jacobian * target_label  # Positive if it pushes towards target class    
    return target_grad  # Saliency scores

def jsma_attack(model, image, target_label, gamma, theta, epsilon):
    """
    Performs a JSMA attack with invisible perturbations on a binary segmentation model.
    
    Args:
        model: The PyTorch model.
        image: The input image (tensor of shape [C, H, W]).
        target_label: The desired segmentation output (tensor of shape [H, W]).
        theta: The perturbation step size.
        gamma: Maximum total distortion.
        epsilon: Maximum distortion per pixel added relative to the original.
    
    Returns:
        The adversarial image.
    """
    adversarial = image.clone().detach()
    accs = torch.zeros(1000)
    itr = 0
    distortion = 0 
    while (distortion < gamma):
        grads = compute_output_grads(model, adversarial)  # Compute the output grads
        saliency = saliency_map(grads, target_label)  # Compute saliency
        
        
        num_pixels = 50  # Number of pixels to modify

        # Get the max saliency value across channels
        max_val, _ = torch.max(saliency, dim=0)  # Shape becomes [H, W]

        # Flatten the saliency map 
        flat_saliency = max_val.view(-1)  # Convert to 1D
        
        # get the indices of the top num_pixels indices to add perturbations
        top_flat_indices = torch.topk(flat_saliency, num_pixels).indices  # Get top 5 indices
        
        # Add perturbations
        with torch.no_grad():
            for idx in top_flat_indices:
                i, j = torch.div(top_flat_indices, image.shape[2], rounding_mode='floor'), top_flat_indices % image.shape[2]      
            adversarial [:,i,j]= adversarial[:,i,j] + theta * saliency.sign()[:,i,j]
            
            adversarial = torch.max(torch.min(adversarial, image + epsilon * image), image - epsilon * image)
            adversarial = torch.clamp(adversarial, 0, 1)


        # Check if misclassification occurs
        pred = model(adversarial.unsqueeze(0))
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        acc[itr] = ((pred == target_label).sum()) / torch.numel(target_label)
        
        #print(f"iter {itr} acc: {acc[itr]}")
        distortion = (((adversarial != image).sum())/torch.numel(image)).float()
        #print(f"iter {itr} distortion: {distortion*100.0}")
        
        if itr > 0 and accs[itr] <= accs[itr-1]:  # stop if accuracy increase to the targeted mask stopped
            break
        itr += 1
    
    print(f"targeted mask acc: {acc[itr]}")
    return adversarial



# Generate Adversarial Example (you should define an tageted mask first )
adv_img = jsma_attack(model, image, targeted_mask,gamma=0.2, theta=0.05, epsilon=32/255)
