import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms


def fgsm_attack_detector(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    epsilon: float = 0.05,
    conf_threshold: float = 0.5,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    """
    Generates an adversarial example for an object detection model using the Fast Gradient Sign Method (FGSM).

    This implementation attacks high-confidence detections by perturbing the input image in the direction
    that maximizes classification loss while maintaining visual similarity to the original image.

    Args:
        image_path (str): Path to the input image file (JPG/PNG)
        model_path (str): Path to YOLOv8 detection model weights (default: 'yolov8n.pt')
        epsilon (float): Magnitude of perturbation (controls attack strength, typically 0.01-0.1)
        conf_threshold (float): Minimum confidence score to consider detections (0.0-1.0)
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
    image = preprocess_image(image_path).to(device)
    image.requires_grad_(True)

    # 2. Raw Model Output Extraction ==========================================
    # Bypass post-processing to access raw predictions:
    # Shape: [batch_size, num_anchors, 4 + 1 + num_classes]
    # Where 4=box_coords (x_center, y_center, width, height)
    #       1=objectness score
    #       num_classes=class logits
    raw_outputs = model.model(image)
    
    # Verify output dimensions
    if raw_outputs[0].shape[-1] < 5:
        raise RuntimeError(f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}")

    # 3. Prediction Processing ================================================
    raw_preds = raw_outputs[0]  # First detection head output
    num_classes = raw_preds.shape[-1] - 5  # Calculate number of classes

    # Extract components from raw predictions
    box_coords = raw_preds[..., :4]   # Bounding box coordinates (xywh format)
    obj_scores = raw_preds[..., 4:5]  # Objectness scores (anchor quality)
    cls_logits = raw_preds[..., 5:]   # Class prediction logits (before softmax)

    # 4. Confidence Calculation ===============================================
    # Combined confidence = objectness * max class probability
    obj_probs = obj_scores.sigmoid()                # Convert to probability [0,1]
    cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]  # Max class prob
    conf_scores = obj_probs * cls_probs             # Final detection confidence

    # 5. High-Confidence Detection Filtering ==================================
    # Create boolean mask for detections above confidence threshold
    mask = conf_scores.squeeze(-1) > conf_threshold
    
    if not mask.any():
        print(f"No detections found with confidence > {conf_threshold}")
        return None

    # 6. Loss Calculation =====================================================
    # Get original predicted classes (detached from computation graph)
    target_classes = cls_logits.argmax(dim=-1)[mask].detach()
    
    # Extract relevant logits for high-confidence detections
    current_logits = cls_logits[mask]
    
    # Cross-entropy loss between current predictions and original classes
    classification_loss = torch.nn.functional.cross_entropy(
        current_logits, 
        target_classes
    )

    # Optional: Add bounding box regression loss
    # target_boxes = box_coords[mask].detach()
    # box_loss = torch.nn.functional.smooth_l1_loss(box_coords[mask], target_boxes)
    # total_loss = classification_loss + box_loss
    
    total_loss = classification_loss

    # 7. Adversarial Perturbation Generation ==================================
    # Clear previous gradients and backpropagate
    model.zero_grad()
    total_loss.backward()
    
    # Extract gradient from input image
    image_grad = image.grad.data
    
    # Apply FGSM: x_adv = x + ε * sign(∇x J(x, y))
    perturbed_image = image + epsilon * image_grad.sign()
    
    # Ensure valid pixel range and disable gradient tracking
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    return perturbed_image