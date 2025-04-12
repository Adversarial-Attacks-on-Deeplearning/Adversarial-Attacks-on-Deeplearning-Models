
import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms

def train_universal_attack(
    model,
    final_image_paths,
    preprocess_fn,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_epochs_uap=20,
    lambda_reg=0.01,
    epsilon=0.1,
    conf_threshold=0.2,
    lr=0.1,
    momentum=0.9
):
    """
    Trains a universal adversarial perturbation (delta) for the given YOLO model.

    :param model: The YOLO model (with .model as PyTorch module).
    :param final_image_paths: List of image paths used to train the universal perturbation.
    :param preprocess_fn: A function to preprocess a given image_path -> returns a torch.Tensor [1, C, H, W].
    :param device: Torch device (e.g., cuda or cpu).
    :param num_epochs_uap: Number of epochs to train.
    :param lambda_reg: Regularization coefficient for delta's L2 norm.
    :param epsilon: Clamping range for the universal delta in each pixel dimension.
    :param conf_threshold: Confidence threshold to filter detections.
    :param lr: Learning rate for SGD.
    :param momentum: Momentum for SGD.

    :return: The learned universal delta (torch.Tensor) and a list of (epoch_loss) for each epoch.
    """
    # Move model to device
    model.model.to(device)

    # Use one sample to get shape for delta initialization
    img_sample = preprocess_fn(final_image_paths[0]).to(device)
    delta = torch.zeros_like(img_sample, requires_grad=True, device=device)

    # Set up optimizer for delta
    optimizer = torch.optim.SGD([delta], lr=lr, momentum=momentum)

    # Record epoch losses if you want to track progress
    epoch_losses = []

    for epoch in range(num_epochs_uap):
        epoch_loss = 0.0

        for image_path in final_image_paths:
            image = preprocess_fn(image_path).to(device)

            # For universal perturbation, same delta for all images
            adv_image = image + delta

            # Forward pass on perturbed image
            raw_outputs = model.model(adv_image)
            if raw_outputs[0].shape[-1] < 5:
                raise RuntimeError(
                    f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}"
                )

            raw_preds = raw_outputs[0]
            num_classes = raw_preds.shape[-1] - 5

            obj_scores = raw_preds[..., 4:5]
            cls_logits = raw_preds[..., 5:]

            obj_probs = obj_scores.sigmoid()
            cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
            conf_scores = obj_probs * cls_probs

            mask = conf_scores.squeeze(-1) > conf_threshold
            if not mask.any():
                # No detections => skip
                continue

            target_classes = cls_logits.argmax(dim=-1)[mask].detach()
            current_logits = cls_logits[mask]

            classification_loss = torch.nn.functional.cross_entropy(
                current_logits, target_classes
            )

            # Optional L2 reg on delta
            reg_loss = lambda_reg * torch.norm(delta, p=2)
            total_loss = classification_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data = torch.clamp(delta, -epsilon, epsilon)

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(final_image_paths)
        epoch_losses.append(avg_loss)
        print(f"UAP Epoch [{epoch+1}/{num_epochs_uap}] Loss: {avg_loss:.4f}")

    print("Training complete.")
    return delta, epoch_losses

