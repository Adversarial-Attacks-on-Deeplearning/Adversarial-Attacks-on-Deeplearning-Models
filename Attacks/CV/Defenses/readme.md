# Noise Fusion: Enhancing Model Robustness  

Noise Fusion is a technique that improves model robustness against adversarial attacks by strategically blending clean images with random noise during training and inference.

## How Noise Fusion Works  

### During Training  

1. **Noise Injection**  
   Each clean training image is mixed with random noise using:  

   \[
   \text{noisy\_image} = (1 - \alpha) \times \text{image} + \alpha \times \text{noise}
   \]  

   Where:  
   - **α** ∈ [0, 0.3] (random mixing factor)  
   - **Noise** can be Gaussian, Uniform, or Poisson distributed  

2. **Training Process**  
   - Models are trained on both clean and noise-fused images  
   - Alternates between clean and noisy batches to enhance robustness  

### During Inference (Defense Mode)  

1. **Input Processing**  
   - Test images are fused with fresh random noise before prediction  
   - Uses same mixing formula as training (with newly sampled α)  

2. **Adversarial Defense**  
   - Noise randomization disrupts adversarial perturbations  
   - Improves prediction accuracy on attacked images  

## Key Benefits  
✔ Improves robustness against adversarial attacks  
✔ Maintains accuracy on clean images  
✔ Simple to implement with standard training pipelines  

## Implementation Example (Pseudocode)  
```python
def noise_fusion(image, alpha, noise_type="gaussian"):
    noise = generate_noise(image.shape, noise_type)
    return (1-alpha)*image + alpha*noise

# Training usage
alpha = random.uniform(0, 0.3)
noisy_img = noise_fusion(clean_img, alpha)
