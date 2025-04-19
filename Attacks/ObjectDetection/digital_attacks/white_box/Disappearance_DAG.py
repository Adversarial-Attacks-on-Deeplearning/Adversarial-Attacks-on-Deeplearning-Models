import torch
from ultralytics import YOLO
from util import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms

def disappearance_dag_attack(
    image_path,
    model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    device='cpu'
):
    """
    Demonstration of a DAG-like multi-iteration attack on YOLOv8
    that pushes all predictions (with high confidence) away from
    their currently predicted class.

    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model.
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model.
        num_iterations (int): Number of attack iterations.
        gamma (float): L_inf step size per iteration.
        conf_threshold (float): Confidence threshold for selecting rows to attack.
        device (str): 'cuda' or 'cpu'.
    """

    # ----------------------------------------------------------------
    # 1. load the model
    # ----------------------------------------------------------------
    model = YOLO(model_path).to(device)
    

    # ----------------------------------------------------------------
    # 2. Preprocess and load the image
    # ----------------------------------------------------------------
    image = preprocess_image(image_path)  # should return a torch.Tensor of shape [1, 3, H, W]
    image = image.to(device)
    image.requires_grad_(True)

    # ----------------------------------------------------------------
    # 3. Iterative attack loop
    # ----------------------------------------------------------------
    for iteration in range(num_iterations):
        # ------------------------------------------------------------
        # (a) Forward pass through the raw underlying model
        # ------------------------------------------------------------
        # The raw model can return a tuple (preds, training_outputs), so unpack:
        raw_preds,_ = model.model(image)

        # raw_preds might be shape [1, 84, N], so we permute to [1, N, 84]
        raw_preds = raw_preds.permute(0, 2, 1)  # now [1, N, 84]
        # Split out the class logits: first 4 channels are box coords, next 80 are class scores
        class_logits = raw_preds[..., 4:]  # shape [1, N, 80]

        # Apply sigmoid if these are logits
        conf = class_logits.sigmoid()  # shape [1, N, 80]

        # Get max confidence per row and the corresponding predicted class
        pred_conf, pred_class = conf.max(dim=-1)  # both [1, N]

        # ------------------------------------------------------------
        # (b) Identify rows to attack
        # ------------------------------------------------------------
        # Let's see what the *high-level* model is currently predicting
        # (This is optional, but helps you track final bounding boxes, etc.)
        with torch.no_grad():
            high_level_preds = model(image, conf=conf_threshold)
        results = high_level_preds[0]  # 'Results' object
        pred_class_ids = results.boxes.cls.tolist()  # the classes YOLO is showing after NMS

        # Now, from the raw model side, pick all rows with conf > conf_threshold
        # whose predicted class is in pred_class_ids. (We treat them as "currently predicted")
        rows_to_attack = []
        target_classes = set(int(x) for x in pred_class_ids)
        for i in range(pred_class.shape[1]):
            row_pred_class = int(pred_class[0, i].item())
            row_pred_conf  = float(pred_conf[0, i].item())
            if row_pred_class in target_classes and row_pred_conf > conf_threshold:
                rows_to_attack.append(i)

        # If no rows to attack, we can break early
        if not rows_to_attack:
            print(f"Iteration {iteration}: no rows above threshold => stopping early.")
            break

        # ------------------------------------------------------------
        # (c) Build the loss
        # ------------------------------------------------------------
        # We'll do an untargeted approach: push the logit of the predicted class down.
        loss = torch.zeros(1, device=device)
        for i in rows_to_attack:
            c = pred_class[0, i]  # the predicted class index
            logit_c = class_logits[0, i, c]
            loss += -logit_c  # negative logit => push it down

        # ------------------------------------------------------------
        # (d) Backprop and update the image
        # ------------------------------------------------------------
        # Zero out old gradients
        if image.grad is not None:
            image.grad.zero_()

        # Backprop
        loss.backward()

        # Get the gradient on the image
        grad = image.grad.data

        # L-infinity step
        step = gamma * grad.sign()
        with torch.no_grad():
            image += step
            image.clamp_(0, 1)

        image.grad.zero_()  # reset for next iteration

        print(f"Iteration {iteration}: #rows_to_attack={len(rows_to_attack)}, loss={loss.item():.4f}")

    # ----------------------------------------------------------------
    # 4. Return or save the final adversarial image
    # ----------------------------------------------------------------
    return image.detach()