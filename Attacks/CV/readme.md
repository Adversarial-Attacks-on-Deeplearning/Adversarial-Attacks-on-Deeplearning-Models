# Computer Vision Attacks

This Directory contains the code for the computer vision attacks

## Attacks

1. **FGSM** - Fast Gradient Sign Method
2. **PGD** - Projected Gradient Descent
3. **Boundary Attack** - Boundary Attack


### FGSM

The Fast Gradient Sign Method (FGSM) is a simple yet effective method to generate adversarial examples. The FGSM attack was introduced by Goodfellow et al. in 2014. The attack is remarkably powerful, and yet intuitive. It is designed to attack neural networks by perturbing input examples in a way that the change is visually imperceptible, but the network classifies the input incorrectly.


#### Formula
Given:
- `x`: Input  
- `y`: True label  
- `J`: Loss function  
- `epsilon (ε)`: Perturbation magnitude  

The adversarial example `x_adv` is defined as:

x_adv = x + ε * sign(∇_x J(θ, x, y))

#### Key Idea
FGSM adds a small perturbation in the direction of the gradient to maximize the model's error.


### Projected Gradient Descent (PGD)

PGD is an iterative adversarial attack method that improves upon single-step methods like FGSM by applying multiple, small perturbations. At each step, it updates the input in the direction that maximizes the loss, then projects the perturbed input back onto an ε-ball around the original input. This ensures that the overall perturbation remains within a specified bound.

#### Formula
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
#### Key Idea

PGD iteratively applies FGSM with a small step size to generate adversarial examples. The perturbation is clipped to ensure it remains within the ε-ball around the original input.

### Boundary Attack


