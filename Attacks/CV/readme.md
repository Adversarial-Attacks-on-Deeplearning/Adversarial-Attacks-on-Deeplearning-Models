# Computer Vision Attacks

This directory contains the code for various computer vision-based adversarial attacks used to evaluate the robustness of machine learning models.

## Attacks Implemented

1. **FGSM** - Fast Gradient Sign Method  
2. **PGD** - Projected Gradient Descent  
3. **Boundary Attack** - Boundary Attack  

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


