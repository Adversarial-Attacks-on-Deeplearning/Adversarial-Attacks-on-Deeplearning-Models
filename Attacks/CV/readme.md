# Computer Vision Attacks

This directory contains the code for various computer vision-based adversarial attacks used to evaluate the robustness of machine learning models.

## Attacks Implemented

1. **FGSM** - Fast Gradient Sign Method  
2. **PGD** - Projected Gradient Descent  
3. **Boundary Attack** - Boundary Attack  
4. **JSMA** - Jacobian-Based Saliency map Attack
5. **MI-FGSM** - Momentum Iterative Fast Gradient Sign Method 
---

## **FGSM - Fast Gradient Sign Method**

The Fast Gradient Sign Method (FGSM) is a simple yet effective method to generate adversarial examples. Introduced by Goodfellow et al. in 2014, FGSM perturbs input examples in a way that is visually imperceptible but forces the network to make incorrect predictions.

### **Formula**
Given:
- `x`: Input image
- `y`: True label
- `J`: Loss function
- `ε`: Perturbation magnitude

The adversarial example `x_adv` is computed as:

\[ x_{adv} = x + \epsilon \cdot sign(\nabla_x J(\theta, x, y)) \]

### **Key Idea**
FGSM adds a small perturbation in the direction of the gradient to maximize the model's error, causing the model to misclassify the perturbed image.

---

## **PGD - Projected Gradient Descent**

Projected Gradient Descent (PGD) is an iterative adversarial attack that improves upon single-step methods like FGSM by applying multiple, small perturbations. It updates the input in the direction that maximizes the loss while ensuring that the perturbation remains within a predefined bound (ε-ball around the original input).

### **Formula**
Given:
- `x`: Original input
- `y`: True label
- `J`: Loss function
- `ε`: Maximum perturbation magnitude
- `α`: Step size
- `k`: Number of iterations

**Initialization:**
```python
x_adv^0 = x
x_adv^(i+1) = clip_{x, ε} { x_adv^i + α * sign(∇_x J(θ, x_adv^i, y)) }
```

### **Key Idea**
PGD iteratively applies FGSM with a small step size to generate adversarial examples while projecting the perturbed input back within the ε-ball constraint to maintain realism.

---

## **Boundary Attack**

Boundary Attack is a **black-box** adversarial attack method that does not require access to model gradients or internal architecture. Unlike gradient-based attacks such as FGSM and PGD, it operates by iteratively perturbing an adversarial example to bring it closer to the original image while ensuring the model still misclassifies it.

### **Characteristics:**
- **Black-box Attack:** Does not require access to model gradients or architecture.
- **Iterative Process:** Starts from a valid adversarial example and refines it iteratively.
- **Minimizes Perturbation:** Attempts to reduce the distance between the adversarial and original image while keeping the misclassification intact.

### **Algorithm**
1. **Initialize with an adversarial example:** The attack starts with an image that is already classified incorrectly by the model.
2. **Generate a random perturbation:** A small, random noise vector is added to the adversarial image.
3. **Validate adversarial status:** If the perturbed image remains misclassified, move it closer to the original input.
4. **Refine the adversarial example:** Reduce the perturbation iteratively to find the smallest possible modification that still leads to misclassification.
5. **Terminate upon convergence:** The attack stops when the minimum perturbation is found, ensuring that the adversarial image is as close as possible to the original while still being misclassified.

---

## **Jacobian-based Saliency Map Attack**


The JSMA is a method for crafting adversarial samples against deep neural networks (DNNs) 
introduced in the paper "The Limitations of Deep Learning in Adversarial Settings" by Nicolas Papernot et al. 
The attack leverages the forward derivative of the DNN to construct adversarial saliency maps, 
which identify the most influential input features to perturb in order to cause misclassification. 
In our work we tried to adapt it to image segmentation model with modifications to improve performance and 
adversarial effectiveness.

### **Forward Derivative (Jacobian Matrix)**

The Jacobian matrix JF(X)/JF​(X) represents how the model’s output changes with respect to small changes in the input. 
It helps identify which pixels/features have the greatest impact on the model’s decision. 

The jacobian matrix of the model's output with respect to the input image is defined as:

$$
J_F(X) = \frac{\partial F(X)}{\partial X} = 
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\frac{\partial F_2}{\partial x_1} & \cdots & \frac{\partial F_2}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}
$$

where:
- $F(X)$ is the model's output (e.g., segmentation mask in JSMA).
- $X$ is the input image.
- each entry $\frac{\partial F_i}{\partial x_j}$ represents the change in the $i$-th output w.r.t the $j$-th input pixel.

To improve performance and speed, we instead calculate the element-wise gradients of the output of the sigmoid function w.r.t input image.

The element-wise gradient matrix is defined as:

for a given output pixel $y_{i,j}$ in the model's prediction:

$$
grad_{i,j} = \frac{\partial F_{i,j}}{\partial x_{i,j}}
$$

where $\text{grad}_{i,j}$ has the same shape as $X$. this represents how sensitive each output pixel is to changes in the input.

### **Adversarial Saliency Map**
A saliency map is computed from the Jacobian to determine which input features should be perturbed to maximize the 
probability of an incorrect classification. The goal is to find pixels that, when modified, 
push the model towards the target class. To perform this, we use a tagret mask to define which pixels to keep thier gradients.

### **Threat Model**
We apply a targeted misclassification attack that Forces the DNN to classify the input into a specific target class. 
In the case of binary image segmentation it tries to make the model predict some pixels as class 1 or 0 depending on the targeted mask.

### **Algorithm steps** 
The JSMA attack iteratively perturbs input features based on the adversarial saliency map until the DNN misclassifies the input into the target class or a maximum distortion threshold is reached.


  1) Compute the element-wise gradients of the output w.r.t input image
  2) Construct the adversarial saliency map based on the targeted mask
  3) Select the most salient feature(s) to perturb).
  4) Modify the selected feature(s) by θ.
  5) Repeat until the distortion limit is reached.

---

## **MI-FGSM - Momentum Iterative Fast Gradient Sign Method**
The momentum method is a technique for accelerating gradient descent algorithms by accumulating a velocity
vector in the gradient direction of the loss function across
iterations. The memorization of previous gradients helps to
barrel through narrow valleys, small humps and poor local
minima or maxima. The momentum method also shows
its effectiveness in stochastic gradient descent to stabilize
the updates.


### **Formula**
To generate a non-targeted adversarial example x* from
a real example x, which satisfies the L∞ norm bound, the velocity vector is defined as:

$$
g_{t+1} = \mu \cdot g_t^* + \frac{\nabla J_t(x_t, y)}{\| \nabla J_t(x_t, y) \|_1} ;$$

then update x*t+1 by applying the sign gradient as:

$$
x_{t+1}^* = x_t^* + \alpha \cdot \text{sign}(g_{t+1}) ;
$$

### **Algorithm steps**
1) Initialize parameters

-   Set α = ε / T (step size per iteration).
-   Set g₀ = 0 (momentum term).
-  t x₀* = x (initial adversarial example as the original image).

2) Iterate for T steps (T is the total number of iterations):

- Compute the gradient.
- Update the momentum term: Accumulate the velocity vector in the gradient direction.
- Update the adversarial example: Apply the sign gradient update.
-  Ensure the updated adversarial example remains within the valid range and perturbation limit.
-  Return the final adversarial example







Below is an example of a Markdown documentation file (README.md) that explains the implementations of DeepFool and SparseFool attacks and includes references to the original papers.

---

# DeepFool and SparseFool Attack Implementations

This repository contains TensorFlow 2.x implementations of two adversarial attack methods:
- **DeepFool** – An untargeted attack that finds minimal perturbations to fool a classifier.
- **SparseFool** – A sparse attack that builds on DeepFool to produce perturbations that alter as few pixels as possible.

Both implementations target image classification models (e.g., EfficientNet) and can be applied to single images.

---

## Table of Contents

- [Overview](#overview)
- [DeepFool Attack](#deepfool-attack)
  - [Reference Paper](#reference-paper-for-deepfool)
  - [Usage](#usage-of-deepfool)
- [SparseFool Attack](#sparsefool-attack)
  - [Reference Paper](#reference-paper-for-sparsefool)
  - [Methodology](#methodology)
  - [Usage](#usage-of-sparsefool)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

Adversarial attacks expose vulnerabilities in deep neural networks by applying carefully crafted perturbations to input data. The **DeepFool** algorithm (Moosavi-Dezfooli et al., 2016) computes minimal \( \ell_2 \) perturbations that cause misclassification by iteratively linearizing the classifier's decision boundary. **SparseFool** (Modas et al., 2019) builds on DeepFool by finding a sparse (i.e., few pixel) perturbation that remains effective, which is particularly relevant for tasks where only minimal changes should occur.

---

## DeepFool Attack

### Reference Paper for DeepFool

- **DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks**  
  S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, CVPR 2016  
  [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)

### Usage of DeepFool

The `deepfool_attack_single` function implements DeepFool for a single image. It returns both the adversarial image and the perturbation applied.

Example:
```python
# Assuming you have a Keras classification model 'model',
# a preprocessed single image 'image' of shape [H, W, C] with pixel values in [0,1],
# and its true class label 'true_label' (an integer).
adv_image, r_adv = deepfool_attack_single(model, image, true_label, max_iter=50, overshoot=0.02)

# To visualize:
import matplotlib.pyplot as plt
plt.imshow(adv_image.numpy())
plt.title("Adversarial Image (DeepFool)")
plt.show()
```

*Note:* Ensure that your image is normalized as required by your model (e.g., EfficientNet often expects inputs scaled to [0, 1]).

---

## SparseFool Attack

### Reference Paper for SparseFool

- **SparseFool: A Few Pixels Make a Big Difference**  
  Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, CVPR 2019  
  [arXiv:1811.02248](https://arxiv.org/abs/1811.02248)

### Methodology

SparseFool refines the adversarial perturbation found by DeepFool into a sparse perturbation by:
1. **Initial Adversarial Example:**  
   Running DeepFool to obtain a perturbation \( r_{\text{adv}} \) so that \( x_B = x + r_{\text{adv}} \) lies near the decision boundary.
2. **Estimating the Decision Boundary Normal:**  
   Computing the gradient at \( x_B \) to obtain the normal vector \( w \).
3. **Iterative Coordinate-Wise Updates:**  
   Updating one coordinate at a time:
   - **Select Coordinate:** Choose the coordinate with the largest absolute gradient value that isn’t saturated.
   - **Compute Perturbation:** Use the formula
     \[
     r_d = \frac{|w^T(x_{\text{current}} - x_B)|}{|w_d|} \cdot \text{sign}(w_d)
     \]
     to determine the minimal update needed along that coordinate.
   - **Apply Projection:** Update \( x_{\text{current}} \) with the computed perturbation and ensure the updated image stays within valid bounds.
   - **Accumulate Perturbation:** Add each coordinate update to the total perturbation \( r_{\text{total}} \).
   - **Stop When Misclassified:** Continue until the classifier misclassifies \( x_{\text{current}} \).

### Usage of SparseFool

Below is an example of how to use the `sparse_fool_attack` function on a single image:

```python
# Assuming you have a Keras classification model 'model',
# a preprocessed single image 'image' of shape [H, W, C],
# and its true label 'true_label'.

adv_im, r_total = sparse_fool_attack(model, image, true_label, 
                                       deepfool_max_iter=20, 
                                       sparse_max_iter=20, 
                                       overshoot=0.02)

# Visualize the adversarial image
import matplotlib.pyplot as plt
plt.imshow(adv_im.numpy())
plt.title("Adversarial Image (SparseFool)")
plt.show()

# Visualize the sparse perturbation (scale if needed)
plt.imshow(tf.clip_by_value(r_total * 50, 0, 1).numpy())
plt.title("Scaled Sparse Perturbation")
plt.show()

# To get prediction of the adversarial example:
adv_prediction = tf.argmax(model(tf.expand_dims(adv_im, axis=0), training=False), axis=1).numpy()[0]
print("Adversarial Prediction: ", adv_prediction)
```



## References

1. **DeepFool:**  
   Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks. In *CVPR 2016*. [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)

2. **SparseFool:**  
   Modas, A., Moosavi-Dezfooli, S.-M., & Frossard, P. (2019). SparseFool: A Few Pixels Make a Big Difference. In *CVPR 2019*. [arXiv:1811.02248](https://arxiv.org/abs/1811.02248)



