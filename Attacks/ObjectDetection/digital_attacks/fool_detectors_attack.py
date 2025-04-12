
import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms




def fool_detectors_attack(
    image_path,
    model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    lambda_reg=0.01,
    target_classes=[11],
    device='cpu'
):
    """
    Implements a DAG-like multi-iteration attack on YOLOv8 targeting specified classes,
    inspired by the paper "Adversarial Examples that Fool Detectors".
    
    The attack minimizes the mean detection confidence for the target classes in the image by
    iteratively updating the input image using the sign of the gradient (L_inf update). An L2 penalty 
    is added to ensure the adversarial image remains close to the original. The attack stops early if 
    the high-level model no longer detects any objects of the target classes.
    
    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model (used for gradient computation).
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model (used for detection).
        num_iterations (int): Number of iterations for the attack.
        gamma (float): Step size for the L_inf gradient update.
        conf_threshold (float): Confidence threshold for selecting detections.
        lambda_reg (float): Weight for the L2 penalty to maintain similarity to the original image.
        target_classes (list of int): List of COCO class IDs to target (e.g., [11] for stop signs).
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: The final adversarial image tensor.
    
    Alternative Name:
        FoolDetectorsAttack
    """
    # Use target_classes as provided.
    
    # 1. Load the YOLO models.
    # --------------------------------------------------------------------------
    # The high-level model is used to check detection results (post-processing),
    # while the underlying (raw) model is used for gradient-based updates.
    high_level_model = YOLO(model_path)
    underlying_model = YOLO(model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # 2. Preprocess and load the input image.
    # --------------------------------------------------------------------------
    # preprocess_image should return a torch.Tensor of shape [1, 3, H, W]
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image for the L2 (similarity) penalty.
    original_image = image.clone().detach()
    # Enable gradient computation on the input image.
    image.requires_grad_(True)

    # 3. Iterative attack loop.
    # --------------------------------------------------------------------------
    for iteration in range(num_iterations):
        # Forward pass: compute predictions using the raw model.
        raw_preds, _ = underlying_model(image)

        # Rearrange predictions from shape [1, channels, N_boxes] to [1, N_boxes, channels].
        raw_preds = raw_preds.permute(0, 2, 1)
        # Extract class logits; assuming the first 4 channels are box coordinates.
        class_logits = raw_preds[..., 4:]
        # Convert logits to confidence scores using the sigmoid activation.
        conf_scores = class_logits.sigmoid()  # Shape: [1, N_boxes, num_classes]

        # Extract confidence scores for the target classes.
        # This indexes dimension 2 with the list of target classes.
        target_confidences = conf_scores[..., target_classes]  # Shape: [1, N_boxes, len(target_classes)]
        # Compute the detector loss as the mean confidence score for target classes.
        loss_det = target_confidences.mean()

        # Compute the L2 penalty to ensure the adversarial image remains similar to the original.
        loss_l2 = lambda_reg * F.mse_loss(image, original_image)

        # Total loss: detector loss plus the L2 penalty.
        loss = loss_det + loss_l2

        # Optional: Use the high-level model to check if any target objects are detected.
        with torch.no_grad():
            high_level_preds = high_level_model(image, conf=conf_threshold)
        results = high_level_preds[0]
        # Extract predicted class IDs (assuming results.boxes.cls holds these).
        detected_classes = results.boxes.cls.tolist() if hasattr(results.boxes, "cls") else []
        # Early stopping: if none of the detected classes is in target_classes, break.
        if not set(target_classes).intersection(set(detected_classes)):
            print(f"Iteration {iteration}: No target objects detected. Early stopping.")
            break

        # Backpropagation: Compute gradients of the loss with respect to the image.
        if image.grad is not None:
            image.grad.zero_()  # Reset gradients.
        loss.backward()

        # Get the gradient and update the image in the negative gradient direction to minimize the loss.
        grad = image.grad.data
        with torch.no_grad():
            # Subtract the sign of the gradient scaled by gamma (L_inf update).
            image -= gamma * grad.sign()
            # Ensure the pixel values remain valid (clamped to [0, 1]).
            image.clamp_(0, 1)
        image.grad.zero_()  # Reset gradient for next iteration.

        # Print the current loss values for debugging.
        print(f"Iteration {iteration}: Loss_det={loss_det.item():.4f}, "
              f"Loss_L2={loss_l2.item():.4f}, Total_loss={loss.item():.4f}")

    # Return the final adversarial image tensor.
    return image.detach()
