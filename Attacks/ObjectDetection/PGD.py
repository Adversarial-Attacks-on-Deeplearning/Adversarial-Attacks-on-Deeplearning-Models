
import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms



def pgd_attack_detector(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    epsilon: float = 0.05,
    num_steps: int = 7,
    step_size: float = 0.005,
    conf_threshold: float = 0.5,
    random_start: bool = True,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    """
    Generates an adversarial example for an object detection model using the Projected Gradient Descent (PGD) attack.

    This implementation attacks high-confidence detections by iteratively perturbing the input image to maximize
    classification loss, projecting perturbations to stay within an L∞ ε-ball to maintain visual similarity.

    Args:
        image_path (str): Path to the input image file (JPG/PNG)
        model_path (str): Path to YOLOv8 detection model weights (default: 'yolov8n.pt')
        epsilon (float): Maximum L∞ perturbation magnitude (typically 0.01-0.1)
        num_steps (int): Number of PGD iterations (default: 7)
        step_size (float): Perturbation step size per iteration (default: 0.005, typically ε/4 or smaller)
        conf_threshold (float): Minimum confidence score to consider detections (0.0-1.0)
        random_start (bool): Whether to initialize with random noise in ε-ball (default: True)
        device (str): Computation device ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Adversarial image tensor of shape [1, 3, H, W] clamped to [0,1]
        None: If no high-confidence detections are found

    Raises:
        RuntimeError: If model output format is unexpected
    """
    
    # 1. Model and Image Preparation ===========================================
    # Load YOLOv8 detection model and move to target device
    model = YOLO(model_path).to(device)
    
    # Preprocess image (normalization, resizing) and enable gradient tracking
    # preprocess_image() should return tensor of shape [1, 3, H, W] in [0,1] range
    original_image = preprocess_image(image_path).to(device)
    image = original_image.clone().requires_grad_(True)

    # Initialize adversarial image
    adv_image = image.clone().detach()
    
    # Optional: Random start within ε-ball
    if random_start:
        random_noise = torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
        adv_image = adv_image + random_noise
        adv_image = torch.clamp(adv_image, 0, 1).detach()

    # 2. PGD Iteration Loop ===================================================
    for _ in range(num_steps):
        # Enable gradient tracking for current iteration
        adv_image.requires_grad_(True)
        
        # 3. Raw Model Output Extraction ======================================
        # Bypass post-processing to access raw predictions
        raw_outputs = model.model(adv_image)
        
        # Verify output dimensions
        if raw_outputs[0].shape[-1] < 5:
            raise RuntimeError(f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}")

        # 4. Prediction Processing ============================================
        raw_preds = raw_outputs[0]  # First detection head output
        num_classes = raw_preds.shape[-1] - 5  # Calculate number of classes

        # Extract components from raw predictions
        box_coords = raw_preds[..., :4]   # Bounding box coordinates (xywh format)
        obj_scores = raw_preds[..., 4:5]  # Objectness scores (anchor quality)
        cls_logits = raw_preds[..., 5:]   # Class prediction logits (before softmax)

        # 5. Confidence Calculation ===========================================
        # Combined confidence = objectness * max class probability
        obj_probs = obj_scores.sigmoid()                # Convert to probability [0,1]
        cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]  # Max class prob
        conf_scores = obj_probs * cls_probs             # Final detection confidence

        # 6. High-Confidence Detection Filtering ==============================
        # Create boolean mask for detections above confidence threshold
        mask = conf_scores.squeeze(-1) > conf_threshold
        
        if not mask.any():
            print(f"No detections found with confidence > {conf_threshold} in iteration")
            return None

        # 7. Loss Calculation =================================================
        # Get original predicted classes (detached from computation graph)
        target_classes = cls_logits.argmax(dim=-1)[mask].detach()
        
        # Extract relevant logits for high-confidence detections
        current_logits = cls_logits[mask]
        
        # Cross-entropy loss between current predictions and original classes
        classification_loss = torch.nn.functional.cross_entropy(
            current_logits, 
            target_classes
        )

        # Optional: Add bounding box regression loss (unchanged from FGSM)
        # target_boxes = box_coords[mask].detach()
        # box_loss = torch.nn.functional.smooth_l1_loss(box_coords[mask], target_boxes)
        # total_loss = classification_loss + box_loss
        
        total_loss = classification_loss

        # 8. Gradient Computation =============================================
        # Clear previous gradients and backpropagate
        model.zero_grad()
        total_loss.backward()
        
        # Extract gradient from adversarial image
        image_grad = adv_image.grad.data
        
        # 9. Update Adversarial Image =========================================
        # PGD update: x_adv = x_adv + α * sign(∇x J(x, y))
        adv_image = adv_image + step_size * image_grad.sign()
        
        # 10. Project to ε-ball: Clip perturbation to [-ε, ε] around original image
        adv_image = torch.clamp(adv_image, original_image - epsilon, original_image + epsilon)
        
        # 11. Ensure valid pixel range [0,1]
        adv_image = torch.clamp(adv_image, 0, 1).detach()

    # 12. Return Final Adversarial Image ======================================
    return adv_image

