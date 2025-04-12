
import torch
from ultralytics import YOLO
from util import preprocess_image  
import torch.nn.functional as F
import numpy as np
from typing import Optional


def square_attack_detector(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    epsilon: float = 0.05,
    patch_size: int = 16,
    num_iterations: int = 500,
    conf_threshold: float = 0.5,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    # 1. Model and Image Preparation
    model = YOLO(model_path).to(device)
    image = preprocess_image(image_path).to(device)  # [1, 3, H, W], [0,1]
    adv_image = image.clone().detach()
    H, W = image.shape[2], image.shape[3]

    # 2. Loss based on pushing confidences below threshold
    def compute_loss(img: torch.Tensor) -> float:
        raw_outputs = model.model(img)
        raw_preds = raw_outputs[0]
        obj_scores = raw_preds[..., 4:5].sigmoid()
        cls_logits = raw_preds[..., 5:]
        cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        conf_scores = (obj_scores * cls_probs).squeeze(-1)

        mask = conf_scores > conf_threshold
        if not mask.any():
            # early stop condition: no detections above threshold
            return 0.0

        target_classes = cls_logits.argmax(dim=-1)[mask].detach()
        current_logits = cls_logits[mask]
        loss = torch.nn.functional.cross_entropy(current_logits, target_classes, reduction='sum')
        return loss.item()

    # 2b. Check initial detections
    initial_loss = compute_loss(image)
    if initial_loss == 0.0:
        print(f"No detections ≥ {conf_threshold} to attack.")
        return None

    best_adv_image = adv_image.clone()
    best_loss = initial_loss

    # 3. Square Attack with early stopping
    for i in range(num_iterations):
        # random patch
        x = np.random.randint(0, W - patch_size + 1)
        y = np.random.randint(0, H - patch_size + 1)
        trial = adv_image.clone()
        perturb = torch.empty(1, 3, patch_size, patch_size, device=device).uniform_(-epsilon, epsilon)
        trial[:, :, y:y+patch_size, x:x+patch_size] += perturb

        # project & clamp
        trial = torch.clamp(trial, image - epsilon, image + epsilon)
        trial = torch.clamp(trial, 0, 1)

        trial_loss = compute_loss(trial)

        # EARLY STOP: if we've driven all confs below threshold
        if trial_loss == 0.0:
            print(f"Early stopping at iteration {i}: all detections < {conf_threshold}")
            break


        if trial_loss > best_loss:
            best_loss = trial_loss
            adv_image = trial.clone()
            best_adv_image = trial.clone()


        if i % 50 == 0:
            print(f"Iteration {i}/{num_iterations} — best loss: {best_loss:.4f}")

    return best_adv_image.detach()
