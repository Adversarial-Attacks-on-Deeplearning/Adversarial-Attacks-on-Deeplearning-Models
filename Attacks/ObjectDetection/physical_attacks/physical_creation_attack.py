
import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms


# Adversarial patch attack

# Define the loss function for the Creation Attack
def creation_attack_loss(raw_pred, confidences, patch_location,target_id):
    """
    Compute the Creation Attack Loss.
    Args:
        raw_pred: raw output tensor from YOLO v8 to get box location.
        confidences: classes confidences to optimize for the target class.
        patch_location: Tuple (x, y) representing the grid cell where the patch is placed.
    return: 
    Loss value
    stop flag
    """
    loss = None
    grid_cell_x, grid_cell_y = patch_location
    # loop on cells to get the patch cell 
    for pred, conf in zip(raw_pred[0], confidences):
        x_min,y_min = pred[0].int(), pred[1].int()
        if (x_min == int(grid_cell_x) and y_min == int(grid_cell_y) ): 
            loss = 1 - conf[target_id]   # Minimize this to increase the target class probability
            break
    
    return loss

# Generate an initial random patch
def preprocess_init_patch(size):
    """
    preprocess initial patch.
    Args:
        size: Tuple (width, height) for the patch size.
    return: 
        initial patch as a numpy array.
    """
    patch = Image.open('ss_patch.png').convert('RGB')
    patch = patch.resize(size)
    patch = np.array(patch) / 255.0
    patch = np.transpose(patch, (2,0,1))
    return patch
## Apply the patch to an image at a fixed random location
def apply_patch(image, patch, patch_location, patch_size):
    """
    Apply the adversarial patch to an image at a predetermined location.
    Args:
        image: Input image (tensor).
        patch: Adversarial patch (tensor).
        patch_location: Tuple (x, y) representing the location to place the patch.
    return:
        Tensor image with the patch applied.
    """

    _, _, image_width, image_height= image.shape
    patch_w, patch_h= patch_size

    x, y = patch_location
    # Ensure the patch fits within the image boundaries
    x_start = max(0, int(x) - patch_w // 2)
    y_start = max(0, int(y) - patch_h // 2)
    x_end = min(image_width, x_start + patch_w)
    y_end = min(image_height, y_start + patch_h)

    # Adjust patch slicing if the patch extends beyond the image boundaries
    patch_x_start = 0
    patch_y_start = 0
    patch_x_end = patch_w 
    patch_y_end = patch_h

    # Ensure the target region and patch have the same dimensions
    target_region = image[:, :, y_start:y_end, x_start:x_end]
    patch_slice = patch[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end]

    # Check if dimensions match
    if target_region.shape[2:] != patch_slice.shape[1:]:
        print(f"Warning: Patch dimensions {patch_slice.shape[1:]} do not match target region dimensions {target_region.shape[2:]}. Resizing patch to fit.")
        
        # Resize the patch to match the target region dimensions
        from torch.nn.functional import interpolate
        patch_slice = interpolate(
            patch_slice.unsqueeze(0),  # Add batch dimension
            size=target_region.shape[2:],  # Target size (height, width)
            mode='bilinear',  # Interpolation mode
            align_corners=False
        ).squeeze(0)  # Remove batch dimension

    # Apply the patch to the image
    image[:, :, y_start:y_end, x_start:x_end] = patch_slice
    return image

# Optimize the adversarial patch
def optimize_patch(
    image_path,
    patch, 
    target_id,
    num_iterations=200,
    raw_model_path="yolov8n_road.pt",
    learning_rate=0.06,
    conf_th = 0.5,
    device = "cuda"
):
    """
    Optimize the adversarial patch using gradient descent.
    Args:
        images: List of images for optimization.
        patch: Initial adversarial patch.
        num_iterations: Number of optimization iterations.
        raw_model_path (str): YOLOv8 weights for the raw model (for gradients).
        learning_rate: Learning rate for gradient descent.
    return: 
        Optimized adversarial patch.
    """
    # Move the patch to the model device
    if device == "cuda":
        patch = torch.tensor(patch, dtype=torch.float32, device="cuda", requires_grad=True)
    else:
        patch = torch.tensor(patch, dtype=torch.float32, requires_grad=True)
    
    # Load the yolo model
    high_level_model = YOLO(raw_model_path)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)  
    
    # Preprocess and load the input image.
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image
    original_image = image.clone().detach()

    
    # Forward pass: compute predictions using the raw model.
    raw_preds, _ = underlying_model(original_image)
    # Rearrange predictions from shape [1, channels, N_boxes] to [1, N_boxes, channels].
    raw_preds = raw_preds.permute(0, 2, 1)   
    class_logits = raw_preds[0][:,4:]            
    conf = class_logits.sigmoid()                
   
    # Select the box with the highest class probability for the target class
    best_pred_idx = torch.argmax(conf[:, target_id]).item()
    best_pred = raw_preds[0][best_pred_idx,:]
    # Get the bounding box coordinates (x, y) for the best prediction
    patch_x, patch_y = best_pred[0].item(), best_pred[1].item()
    patch_location = (patch_x, patch_y)
    patch_size = (best_pred[2].int().item(), best_pred[3].int().item())
    for iteration in range(num_iterations):
        torch.cuda.empty_cache()
        # Original loss calculation
        loss = torch.tensor(0.0, device=image.device,requires_grad=True)
        # apply patch on determined box
        patched_image = apply_patch(original_image.clone(), patch, patch_location, patch_size)
        
        # Forward pass: Compute predictions using the raw model on patched image.
        raw_preds, _ = underlying_model(patched_image)  
        raw_preds = raw_preds.permute(0, 2, 1)
        class_logits = raw_preds[0][:, 4:]             
        conf = class_logits.sigmoid()  

        # Compute the loss
        loss = creation_attack_loss(raw_preds,conf, patch_location,target_id)

        # stop when target object added
        with torch.no_grad():
            current_preds = high_level_model(patched_image, conf=conf_th)
        results = current_preds[0]
        current_class_ids = set(int(box.cls.item()) for box in results.boxes)
        # Condition: adversarial_class in current_class_ids
        if (target_id in current_class_ids):
            print(f"Iteration {iteration}: Stop condition met!")
            break

        # Backpropagate and update the patch
        if loss is not None:
            loss.backward(retain_graph=True)
            # Manually update the patch using gradient descent
            with torch.no_grad():
                grad = patch.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                if grad_norm == 0:
                    grad_norm = 1e-8
                patch -= learning_rate * (grad / grad_norm)# Update patch
                patch.grad.zero_()  # Reset gradients
                # Clip the patch values to ensure they are valid pixel values
                patch.data = torch.clamp(patch.data, 0, 1)
        
        

            # Re-enable gradients for the next iteration
            patch.requires_grad = True
        
            # Print the loss for monitoring            
            print(f"Iteration {iteration}, Loss: {loss.item()}")
        
        else:
            print(f"Iteration {iteration}: location not found")
            # change patch location 
            # Select the prediction with the highest class probability for the target class
            best_pred_idx = torch.argmax(conf[:, target_id]).item()
            best_pred = raw_preds[0][best_pred_idx,:]
            # Get the bounding box coordinates (x, y) for the best prediction
            patch_x, patch_y= best_pred[0].item(), best_pred[1].item()
            patch_location = (patch_x, patch_y)
            
            
    return patched_image.detach().cpu(),patch.detach().cpu()


