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

