# Disappearance DAG Attack

---

## 1. Attack Idea
This attack is inspired by the **Dense Adversary Generation (DAG)** algorithm. It generates imperceptible perturbations to make object detections "disappear" by suppressing their confidence scores in YOLOv8. Unlike targeted attacks, it does NOT specify new classes for predictions.

---

## 2. Attack Steps & Loss Function

### Key Steps:
1. **Target Selection**: Find all detections with confidence > `conf_threshold`.
2. **Loss Calculation**: Minimize the model's confidence for these detections.
3. **Gradient Update**: Perturb the image iteratively using L-infinity bounded steps.

### Loss Function:
The loss aims to reduce logits (pre-sigmoid scores) of predicted classes:
$$
\text{Loss} = -\sum_{i \in \text{targets}} \text{logit}(c_i)
$$
where:
- $c_i$ = predicted class for detection $i$
- $\text{logit}(c_i)$ = raw output score for class $c_i$

---

## 3. Implementation Summary
```python
def disappearance_dag_attack(...):
    # 1. Load YOLO models (raw + high-level)
    # 2. Preprocess image to tensor
    # 3. For N iterations:
    #    a) Get raw model predictions
    #    b) Select high-confidence detections
    #    c) Compute loss over targets
    #    d) Update image: image += gamma * sign(gradient)
    # 4. Return perturbed image
```

---

## 4. Usage Example
```python
from ultralytics import YOLO

# Generate adversarial image
adv_image = disappearance_dag_attack(
    image_path="test.jpg",
    num_iterations=15,
    gamma=0.05,
    device="cuda"
)

# Compare results
model = YOLO("yolov8n.pt")
model.predict("test.jpg")[0].show()     # Original detections
model.predict(adv_image)[0].show()      # Objects disappear
```
## **paper:**
Adversarial Examples for Semantic Segmentation and Object Detection


# Targeted DAG Attack 

---

## 1. Attack Idea
This **targeted attack** modifies the DAG algorithm to push all detections toward a specific adversarial class. It combines:
1. **Suppression** of original high-confidence detections
2. **Promotion** of a chosen adversarial class
3. **Early stopping** when the adversarial class dominates

---

## 2. Attack Steps & Loss Function

### Key Stages:
1. **Initial Analysis**: Identify original high-confidence classes
2. **Targeted Perturbation**:
   - Push original detections toward adversarial class
   - If no originals remain, boost adversarial class globally
3. **Termination**: Stop when adversarial class appears and originals disappear

### Loss Function
The loss has two modes depending on remaining targets:

1. **Targeted Mode** (original detections present):
$$
\text{Loss} = \sum_{i \in \text{targets}} (\text{logit}_{\text{adv}}^{(i)} - \text{logit}_{\text{orig}}^{(i)})
$$
- $\text{logit}_{\text{adv}}$: Adversarial class score
- $\text{logit}_{\text{orig}}$: Original class score

2. **Global Promotion Mode** (no originals left):
$$
\text{Loss} = \sum_{i} \text{logit}_{\text{adv}}^{(i)}
$$

---

## 3. Implementation Summary
```python
def targeted_dag_attack(...):
    # 1. Load YOLO models (raw + high-level)
    # 2. Get original predictions to determine target classes
    # 3. For N iterations:
    #    a) Calculate class logits from raw model
    #    b) Select rows with original classes
    #    c) Compute loss based on current mode
    #    d) Update image: image += gamma * sign(gradient)
    #    e) Check stop condition (adv_class present + targets gone)
    # 4. Return perturbed image and stop status
```

---

## 4. Usage Example
```python
from ultralytics import YOLO

# Convert cat detections to "truck" (class 7)
adv_image, stopped_early = targeted_dag_attack(
    image_path="cat.jpg",
    adversarial_class=7,    # COCO class index for truck
    num_iterations=20,
    gamma=0.04,
    conf_threshold=0.3,
    device="cuda"
)

# Analyze results
model = YOLO("yolov8n.pt")
print("Original classes:", model("cat.jpg")[0].boxes.cls.unique())
print("Adversarial classes:", model(adv_image)[0].boxes.cls.unique())
```
## **paper:**
Adversarial Examples for Semantic Segmentation and Object Detection



# Fool Detectors Attack 

---

## 1. Attack Idea
This attack implements a **targeted adversarial attack** on object detectors (YOLOv8) inspired by the paper ["Adversarial Examples that Fool Detectors"](#). It generates perturbations to:
1. **Suppress target classes**: Reduce detection confidence for specified COCO classes (e.g., stop signs)
2. **Maintain stealth**: Use L2 regularization to keep adversarial images visually similar to originals
3. **Enable early stopping**: Terminate if target classes are no longer detected

---

## 2. Attack Steps & Loss Function

### Key Components from the Paper:
1. **Detector Vulnerability**: Exploits the detector's sensitivity to gradient patterns in class logits
2. **Multi-Objective Optimization**:
   - Minimize mean confidence of target classes
   - Constrain perturbations to stay near original image (L2 penalty)

### Mathematical Formulation
**Total Loss**:
$$
\text{Loss} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \text{conf}(c_{\text{target}}^{(i)}}_{\text{Detection Loss}} + \underbrace{\lambda \cdot ||\mathbf{X}_{\text{adv}} - \mathbf{X}_{\text{orig}}||_2^2}_{\text{L2 Regularization}}
$$

Where:
- $c_{\text{target}}$: Target class(es) to suppress
- $\lambda$: Regularization strength (`lambda_reg`)
- $N$: Number of bounding box proposals

---

## 3. Implementation Overview

```python
def fool_detectors_attack(...):
    # 1. Load YOLO models (raw model for gradients, high-level for detection)
    # 2. Preprocess image to tensor [1, 3, H, W]
    # 3. For N iterations:
    #    a) Compute class confidences for target classes
    #    b) Calculate detection loss + L2 penalty
    #    c) Backpropagate gradients
    #    d) Update image: X = X - γ * sign(∇Loss)
    #    e) Early stop if target classes disappear
    # 4. Return adversarial image
```

### Key Components:
| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Raw Model**           | YOLO backbone without post-processing for gradient computation             |
| **High-Level Model**    | Full YOLO with NMS to check detection results                               |
| **L-inf Update**        | `image -= gamma * grad.sign()` ensures bounded perturbations               |
| **Early Stopping**      | Monitors high-level model predictions for target class disappearance        |

---

## 4. Usage Example

```python
from ultralytics import YOLO
import torch

# Initialize attack (targeting stop signs - COCO class 11)
adv_image = fool_detectors_attack(
    image_path="stop_sign.jpg",
    target_classes=[11],          # COCO class 11 = stop sign
    num_iterations=20,
    gamma=0.05,
    lambda_reg=0.1,               # Balance stealth vs effectiveness
    conf_threshold=0.3,
    device="cuda"
)

# Compare results
model = YOLO("yolov8n.pt")

# Original image
orig_results = model("stop_sign.jpg")[0]
print("Original detections:", orig_results.boxes.cls.tolist())  # e.g., [11, 11]

# Adversarial image 
adv_results = model(adv_image)[0]
print("Adversarial detections:", adv_results.boxes.cls.tolist())  # e.g., []
```
## **paper:**
Adversarial Examples that Fool Detectors




# FGSM Attack for Object Detectors 

---

## 1. Attack Idea
This implementation adapts the **Fast Gradient Sign Method (FGSM)** for object detection models (YOLOv8). The attack:
1. **Targets High-Confidence Detections**: Focuses on predictions with confidence > specified threshold  
2. **Maximizes Classification Error**: Perturbs image to confuse class predictions  
3. **Optional Bounding Box Disruption**: Can simultaneously degrade localization accuracy (disabled by default)  
4. **Maintains Stealth**: Uses L-infinity bounded perturbations (ε typically 0.01-0.1)  
5. **One shot**: not iterative so it is fast and suitable for adversarial training

---

## 2. Attack Steps & Loss Function

### Key Components from FGSM Paper:
1. **Linear Explanation**: Exploits gradient sign direction for efficient perturbation  
2. **Single-Step Attack**: Computes perturbation in one forward/backward pass  

### Mathematical Formulation:
**Total Loss**:
$$
\mathcal{L}_{\text{total}} = \underbrace{-\sum_{i \in \mathcal{D}} \log(p(y_i|\mathbf{x}))}_{\text{Classification Loss}} + \lambda \underbrace{||\mathbf{x}_{\text{adv}} - \mathbf{x}_{\text{orig}}||_2}_{\text{Implicit L2 Constraint}}
$$

Where:
- $\mathcal{D}$ = High-confidence detections  
- $p(y_i|\mathbf{x})$ = Class probability for detection $i$  
- $\lambda$ = Implicitly controlled via ε clamping  

---

## 3. Implementation Overview

### Pipeline:
```python
1. Load YOLOv8 detection model
2. Preprocess input image (normalization + resizing)
3. Extract raw model outputs (bypass NMS/post-processing)
4. Calculate combined confidence scores (objectness × class probability)
5. Filter detections by confidence threshold
6. Compute classification loss between original and current predictions
7. Backpropagate loss to get image gradient
8. Apply perturbation: x_adv = x + ε·sign(∇x ℒ)
9. Clamp to valid pixel range [0,1]
```

### Key Components:
| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Raw Output Extraction** | Accesses model backbone outputs before non-max suppression                 |
| **Confidence Filtering**  | Combines objectness and class probability for detection quality assessment |
| **Gradient Isolation**    | Only perturbs high-confidence detections via boolean masking              |
| **Device Agnostic**       | Works on both CPU/GPU with automatic tensor placement                      |

---

## 4. Usage Example

```python
from attacks import fgsm_attack_detector
from ultralytics import YOLO

# Generate adversarial example
adv_image = fgsm_attack_detector(
    image_path="highway.jpg",
    model_path="yolov8n.pt",
    epsilon=0.08,        # Moderate perturbation
    conf_threshold=0.6,  # Target confident detections
    device="cuda"
)

# Evaluate attack effectiveness
model = YOLO("yolov8n.pt").to("cuda")
original_results = model("highway.jpg")[0]
adversarial_results = model(adv_image)[0]

print(f"Original detections: {len(original_results.boxes)}")
print(f"Adversarial detections: {len(adversarial_results.boxes)}")
```

## 8. papers

1. **Original FGSM Paper**:  
   [Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2015)](https://arxiv.org/abs/1412.6572)

2. **Object Detector Attacks**:  
   [Adversarial Examples for Object Detectors (Lu et al., 2017)](https://arxiv.org/abs/1712.08063)

3. **YOLOv8 Documentation**:  
   [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)



# UAP Attack for Object Detectors

---

## 1. Attack Idea

This implementation adapts a **Universal Adversarial Perturbation (UAP)** approach to fool object detectors (e.g., YOLOv8) across *many images* using a single, shared perturbation \(\delta\). The main points:

1. **Single Perturbation for All Images**: \(\delta\) is learned to degrade model performance for *any* input.  
2. **Feature-Level Disruption**: Exploits the detector’s internal representations to degrade classification and (optionally) localization.  
3. **Bounded Perturbation**: Clamps perturbation to \(\|\delta\|_\infty \le \epsilon\).  
4. **Mini-Batch Training**: Iteratively refines \(\delta\) by processing a set of images.  
5. **Stealth**: Maintains imperceptibility constraints (small \(\epsilon\)) while maximizing the drop in detection accuracy.

---

## 2. Attack Steps & Loss Function

### Key Components of UAP:

1. **Data-Driven**: (optional) We gather a subset of training images or use random proxies (data-independent).  
2. **Iterative Refinement**: Unlike FGSM’s single-step approach, we typically run multiple epochs.  
3. **Universal Across Images**: We accumulate gradients from *all* images and update the same \(\delta\).

### Mathematical Formulation

**Loss Function** (simplified):

\[
\mathcal{L}(\delta) \;=\; \sum_{x \in \mathcal{D}} \big(\; \mathcal{L}_{\mathrm{detect}}(x + \delta) \;+\; \lambda \|\delta\|_2 \big)
\]

Where:
- \(\mathcal{D}\) is the training set (or a chosen subset).  
- \(\mathcal{L}_{\mathrm{detect}}(\cdot)\) is the detection loss (e.g., cross-entropy on classification logits).  
- \(\|\delta\|_2\) is an optional L2 regularizer for stability (\(\lambda\) is the trade-off).  
- \(\delta\) is clamped in each update step to ensure \(\|\delta\|_\infty \le \epsilon\).

---

## 3. Implementation Overview

### Pipeline:

```python
1. Gather or sample a set of images (final_image_paths).
2. Initialize delta = 0 with shape = (1, C, H, W).
3. For each epoch:
   a. For each image x:
       i. x_adv = x + delta
       ii. raw_outputs = model.model(x_adv)
       iii. Filter out low-confidence detections
       iv. classification_loss = cross_entropy(...)
       v. reg_loss = lambda_reg * torch.norm(delta, p=2)
       vi. total_loss = classification_loss + reg_loss
       vii. total_loss.backward()
       viii. optimizer.step()
       ix. clamp delta to [-epsilon, epsilon]
   b. Print average epoch loss
4. Return final delta
```

### Key Components:

| **Component**         | **Description**                                                   |
|-----------------------|-------------------------------------------------------------------|
| **Training Set**      | Either real images or random “proxy data” (for data-independent). |
| **Iterative Updates** | Each epoch, accumulate gradients from *all* images.              |
| **Confidence Mask**   | Only compute classification loss for bounding boxes above a threshold. |
| **Clamping**          | Keep \(\delta\) within \([- \epsilon, \epsilon]\).               |

---

## 4. Usage Example

```python
from your_uap_code import train_universal_attack
from ultralytics import YOLO

# Prepare your data
images_dir = "/content/train/images"
labels_dir = "/content/train/labels"
final_image_paths = [...]  # e.g. from a stratified sampling script
model_path = "yolov8n.pt"

# Create & load model
model = YOLO(model_path)

# Train universal delta
delta, losses = train_universal_attack(
    model,
    final_image_paths,
    preprocess_image_fn,  # your function that loads & normalizes an image
    device="cuda",
    num_epochs_uap=20,
    lambda_reg=0.01,
    epsilon=0.1,
    conf_threshold=0.2,
    lr=0.1,
    momentum=0.9
)
```

---

## 5. papers

1. **Universal Adversarial Perturbations**  
   - [Moosavi-Dezfooli et al., CVPR 2017](https://arxiv.org/abs/1610.08401)  
   - [Mopuri et al., BMVC 2017](https://arxiv.org/abs/1707.01705)

2. **Attacks on Object Detectors**  
   - [Xie et al., CVPR 2019 (Translation Invariant Attack)](https://arxiv.org/abs/1904.02884)  
   - [Wei et al., AAAI 2022 (Cross-Task Transfer)](https://arxiv.org/abs/2201.08517)

3. **YOLOv8 Documentation**  
   - [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)


---

# PGD Attack for Object Detectors

---

## 1. Attack Idea
This implementation adapts the **Projected Gradient Descent (PGD)** attack for object detection models (YOLOv8). The attack:
1. **Targets High-Confidence Detections**: Focuses on predictions with confidence above a specified threshold.
2. **Maximizes Classification Error**: Iteratively perturbs the image to confuse class predictions.
3. **Optional Bounding Box Disruption**: Can degrade localization accuracy (disabled by default).
4. **Maintains Stealth**: Uses L-infinity bounded perturbations (ε typically 0.01–0.1) to ensure visual similarity.
5. **Iterative Strength**: Applies multiple smaller steps with projection, making it stronger than single-step attacks like FGSM.

---

## 2. Attack Steps & Loss Function

### Key Components from PGD Paper:
1. **Iterative Gradient Descent**: Computes gradients over multiple steps, refining perturbations for stronger attacks.
2. **Projection**: Ensures perturbations stay within an L∞ ε-ball, balancing attack strength and stealth.
3. **Random Initialization**: Optionally starts with random noise to escape local optima and enhance effectiveness.

### Mathematical Formulation:
**Total Loss**:
$$
\mathcal{L}_{\text{total}} = -\sum_{i \in \mathcal{D}} \log(p(y_i|\mathbf{x}_{\text{adv}}))
$$

**Perturbation Update** (per iteration):
$$
\mathbf{x}_{\text{adv}}^{t+1} = \Pi_{\|\mathbf{x}_{\text{adv}} - \mathbf{x}_{\text{orig}}\|_\infty \leq \epsilon} \left( \mathbf{x}_{\text{adv}}^t + \alpha \cdot \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}_{\text{adv}}^t, y)) \right)
$$

Where:
- $\mathcal{D}$ = High-confidence detections.
- $p(y_i|\mathbf{x}_{\text{adv}})$ = Class probability for detection $i$ in the adversarial image.
- $\epsilon$ = Maximum L∞ perturbation magnitude.
- $\alpha$ = Step size per iteration.
- $\Pi$ = Projection operator clipping to the ε-ball around the original image.
- Classification loss drives misclassification; L∞ constraint is enforced explicitly via projection.

---

## 3. Implementation Overview

### Pipeline:
```python
1. Load YOLOv8 detection model
2. Preprocess input image (normalization + resizing)
3. Initialize adversarial image (optionally with random noise in ε-ball)
4. For each iteration (num_steps):
   a. Extract raw model outputs (bypass NMS/post-processing)
   b. Calculate combined confidence scores (objectness × class probability)
   c. Filter detections by confidence threshold
   d. Compute classification loss between original and current predictions
   e. Backpropagate loss to get image gradient
   f. Update adversarial image: x_adv = x_adv + α·sign(∇x ℒ)
   g. Project to ε-ball: clip x_adv to [x_orig - ε, x_orig + ε]
   h. Clamp to valid pixel range [0,1]
5. Return final adversarial image
```

### Key Components:
| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Raw Output Extraction** | Accesses model backbone outputs before non-max suppression                 |
| **Confidence Filtering**  | Combines objectness and class probability for detection quality assessment |
| **Iterative Updates**     | Applies smaller perturbations over multiple steps for stronger attacks     |
| **Projection**            | Clips perturbations to enforce L∞ ε-ball constraint                       |
| **Random Start**          | Optional random noise initialization to improve attack robustness          |
| **Device Agnostic**       | Works on both CPU/GPU with automatic tensor placement                      |

---

## 4. Usage Example

```python
# Generate adversarial example
adv_image = pgd_attack_detector(
    image_path="highway.jpg",
    model_path="yolov8n.pt",
    epsilon=0.05,        # Moderate perturbation budget
    num_steps=7,         # Number of iterations
    step_size=0.005,     # Step size per iteration
    conf_threshold=0.6,  # Target confident detections
    random_start=True,   # Use random initialization
    device="cuda"
)

```

---

## 5. Papers

1. **Original PGD Paper**:  
   [Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2017)](https://arxiv.org/abs/1706.06083)

2. **Object Detector Attacks**:  
   [Adversarial Examples for Object Detectors (Lu et al., 2017)](https://arxiv.org/abs/1712.08063)
---


# Summary of results of the digital attacks on yolov8 

| Attack Technique   | Clean mAP@0.5 | Adversarial mAP@0.5 | Drop      |
|--------------------|---------------|---------------------|-----------|
| FGSM               | 0.927         | 0.312               | -0.615    |
| Fool Detectors     | 0.927         | 0.155               | -0.772    |
| UAP                | 0.927         | 0.443               | -0.484    |
| Disappearance DAG  | 0.935         | 0.547              | -0.388    |
| Targeted DAG       | 0.927         | 0.0947              | -0.8323   |



# Adversarial Physical Dissapearance Attack on YOLO Object Detection

## Overview

This project implements an adversarial attack against YOLO-based object detection models by generating optimized patches that reduce the confidence of detecting a specific target class. The attack works by strategically placing small patches within the object's bounding box to cause misclassification or disappearance.

## Attack Mechanism

The attack utilizes gradient-based optimization to craft adversarial patches that, when overlaid on an image, minimize the detection confidence of the target object. The approach includes:

1. **Target Class Identification**: The attack is focused on a specific class (default: `11`).
2. **Patch Generation**: Randomly initialized patches are optimized iteratively.
3. **Grid Search Optimization**: Patches are placed in different positions within the bounding box to find the most effective locations.
4. **Physical-World Transformations**: The attack accounts for real-world robustness by applying affine transformations, perspective warping, and brightness variations.
5. **Total Variation Regularization**: A smoothness constraint is applied to ensure the generated patches are visually plausible.

## Loss Function

The attack optimizes an adversarial loss function that consists of two key components:

### **1. Disappearance Loss (Main Loss)**

This loss function minimizes the confidence of the target class in the YOLO detection model.

#### **Mathematical Definition:**
The disappearance loss is formulated as:

```
J_d(x,y) = max_{s ∈ S^2, b ∈ B} P(s,b,y,f_θ(x))
```

where:
- `f_θ(x)` represents the output of the YOLO object detector.
- `P(s,b,y,f_θ(x))` extracts the probability of the target class `y` in grid cell `s` and bounding box `b`.
- The attack minimizes this probability until it falls below the detection threshold (typically 0.25).

#### **Code Implementation:**

```python
def calculate_disappearance_loss(self, results, target_class):
    """Calculate the loss based on the confidence of the target class."""
    max_conf = 0.0
    found_target = False
    target_confidences = []
    
    # Extract confidence scores for target class
    for i in range(len(results[0].boxes)):
        cls = int(results[0].boxes.cls[i].item())
        if cls == target_class:
            conf = results[0].boxes.conf[i]
            target_confidences.append(conf)
            found_target = True
    
    if found_target and target_confidences:
        confidences_tensor = torch.stack(target_confidences)
        max_conf_tensor = torch.max(confidences_tensor)
        max_conf = max_conf_tensor.item()
        loss = max_conf_tensor  # Minimize this loss
    else:
        max_conf = 0.0
        loss = torch.tensor(0.01, device=self.device, requires_grad=True)  # Attack success
        
    return loss, max_conf
```

✅ **Objective**: Reduce the confidence of the target class to make it disappear from the detection results.

### **2. Total Variation Loss (TV Loss)**

This loss is used as a regularization term to ensure the patch is smooth and does not contain noisy patterns that might be unrealistic.

#### **Mathematical Definition:**
The total variation loss is defined as:

```
TV(M_x ⋅ δ) = Σ_{i,j} |(M_x ⋅ δ)_{i+1,j} - (M_x ⋅ δ)_{i,j}| + |(M_x ⋅ δ)_{i,j+1} - (M_x ⋅ δ)_{i,j}|
```

where:
- `M_x` is the mask defining the patch area.
- `δ` is the adversarial perturbation.
- The summation ensures smoothness by penalizing pixel-level differences.

#### **Code Implementation:**

```python
def total_variation_loss(self, patch):
    """Calculate Total Variation loss to encourage patch smoothness."""
    # Compute differences in x and y directions
    diff_x = torch.abs(patch[:, :, :-1] - patch[:, :, 1:])
    diff_y = torch.abs(patch[:, :-1, :] - patch[:, 1:, :])
    
    # Sum all differences
    tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
    
    return tv_loss
```

✅ **Objective**: Prevent adversarial patches from having sharp edges or unrealistic pixel variations.

### **Final Combined Loss**

The final loss function combines the disappearance loss and total variation loss:

```
L_total = J_d(x,y) + λ TV(M_x ⋅ δ)
```

where `λ` is a weight parameter to balance between adversarial effectiveness and smoothness.

#### **Implementation:**

```python
tv_weight = 0.1  # Adjust based on results
tv_loss = self.total_variation_loss(patch)
total_loss = loss + tv_weight * tv_loss
```

✅ **Overall Objective:**

1. Minimize the detection confidence of the target class.
2. Ensure the patch remains smooth and natural-looking.

## Dependencies

Ensure you have the required dependencies installed:

```bash
pip install torch torchvision opencv-python matplotlib ultralytics tifffile
```

## How to Run

### 1. Training an Adversarial Patch

```bash
python Disappearance.py
```

This will generate adversarial patches for the target class and apply them to a given image.

### 2. Custom Parameters

You can modify key parameters within the script:

- **Target Class**: Change `target_class` in `AdversarialPatchGenerator`.
- **Patch Size Ratio**: Adjust `patch_ratio` to control patch size relative to the object.
- **Grid Size**: Modify `grid_size` to set the number of search positions for patches.

### 3. Example Execution

To generate patches and evaluate their impact:

```python
# Initialize attack
attack = AdversarialPatchGenerator(target_class=10, patch_ratio=0.02, grid_size=10)

# Run attack
optimized_patches, final_conf = attack.train(
    image_path="road666_png.rf.efa2191f5abb5266654b44edb18bb15a.jpg",
    num_patches=3,
    epochs_per_location=20,
    base_lr=0.1,
    refinement_epochs=20,
    patch_image_path="patch.jpeg"
)
```

## Results

- The attack generates optimized adversarial patches that minimize the detection confidence of the target object.
- The patches are saved as `.tiff` images, and the final adversarial image is visualized and saved as `final_result_multi.png`.

## Future Improvements

- Implement more advanced transformations for better robustness.
- Extend attack to different object detection models.
- Optimize runtime performance using efficient gradient computation techniques.

## References

- [YOLO Object Detection](https://github.com/ultralytics/yolov8)

---



# Creation Physical Attack 
## Overview 
The creation attack loss mechanism aims to fool the model into recognizing nonexistent objects. Similar to the 
“adversarial patch” approach, the goal is to create a physical sticker that can be added to any existing scene. 
Contrary to the adversarial patch, rather than causing a misclassification, the aim is to create a new classification 
(i.e., a new object detection) where none existed before.

The objective proposed in this attack uses a composite loss function that first aims at creating a new object 
localization, followed by a targeted “mis-classification.” However, since we are working with YOLOv8, the objective is 
to maximize the target class probability until the target class probability exceeds 50%.

let fθ (x) represent the full output tensor of YOLO v8 on input
scene x, and let P(s, b, y, fθ (x)) represent the probabil-
ity assigned to class y in box b of grid cell s. 
loss is then

Jc (x, y) = 1 - P(s, b, y, fθ (x))
 
## Attack mechanism

The attack utilizes gradient-based optimization to craft adversarial patches that, when overlaid on an image, maximize the detection confidence of the target object.
The approach includes:

1. **Target Class Identification**: The target class to be added.
2. **Patch Generation**: Use initial patch that optimized iteratively.
3. **Get patch location**: Patch is placed at the grid that has the maximum propability for the target class.
4. **Apply patch**: The patch size is modified to fit the grid.
5. **Patch optimization**: Use gradient-based optimization to the patch to decrease loss.

## results
![img0](https://github.com/user-attachments/assets/abbbf8b7-7cda-4da8-ae1f-c9be9738056b)


![det0](https://github.com/user-attachments/assets/f41df6ae-13ee-4544-a617-3e68ac1ceb32)


![img3](https://github.com/user-attachments/assets/8be3b681-4bf5-4417-8de1-70a77f9a9a33)


![det3](https://github.com/user-attachments/assets/b77fe74d-e09f-441a-8963-04a363969c2c)



### Initial patch used:

![ss_patch](https://github.com/user-attachments/assets/d19615d9-c82d-4cc2-a99a-c5eeb8dab2d1)




