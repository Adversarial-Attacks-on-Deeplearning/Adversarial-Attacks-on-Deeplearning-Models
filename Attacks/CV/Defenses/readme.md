# Noise Fusion Defense:

##  What is Noise Fusion Defense?

Noise Fusion is a **simple yet powerful defense** strategy against adversarial attacks. Instead of relying on complex adversarial training, Noise Fusion **injects random noise into input images** both during training and inference. This randomization makes it harder for adversarial perturbations to consistently fool the model.

Noise Fusion was inspired by the observation that adversarial examples are often finely tuned to specific pixel values. By fusing images with random noise, the adversarial perturbations become less effective, allowing the model to make more robust predictions.

---

##  How Does Noise Fusion Work?

### During Training:
- Each clean training image is **mixed** with **random noise**.
- The fusion follows a simple formula:

  \[ \text{noisy\_image} = (1 - \alpha) \times \text{image} + \alpha \times \text{noise} \]

- Where:
  - **\( \alpha \)** is a random mixing factor (e.g., \( \alpha \sim U(0, 0.3) \)).
  - **Noise** can be Gaussian, Uniform, or Poisson distributed.

- The model is trained on both **clean** and **noise-fused** images alternately to enhance robustness.

### During Inference (Defense Mode):
- Before passing a test image to the model, the image is **fused with fresh random noise**.
- The model predicts on this fused image.

This randomization disrupts adversarial perturbations and helps the model predict correctly even on attacked images.

---

##  Why Does Noise Fusion Help?
- **Random noise breaks the structure** of adversarial perturbations.
- **Fine-grained perturbations become ineffective** after noise fusion.
- It acts like "randomized smoothing," making the decision boundary less sensitive to small, crafted changes.
- **Very simple to implement** compared to adversarial training or certified defenses.

---

##  Experimental Results Summary

| **Attack**       | **Epsilon (ε)** | **Alpha (α)** | **Test Accuracy after applying attack** | **Accuracy after Noise Fusion Defensive Mechanism** |
|:----------------:|:---------------:|:-------------:|:----------------------------:|:-------------------------------:|
| **Original Model**| -               | -             | 96%                          | -                               |
| **FGSM**         | 0.007           | -             | 58%                          | 94.77%                          |
| **FGSM**         | 0.01            | -             | 44%                          | 93.80%                          |
| **FGSM**         | 0.03            | -             | 14%                          | 84.11%                          |
| **FGSM**         | 0.1             | -             | 6%                           | 52.52%                          |
| **PGD**          | 0.007           | 0.00175       | 35%                          | 94.19%                          |
| **PGD**          | 0.01            | 0.0025        | 7%                           | 86.05%                          |
| **PGD**          | 0.03            | 0.0075        | 3%                           | 78.68%                          |
| **PGD**          | 0.1             | 0.025         | 0%                           | 59.30%                          |

---

##  Key Takeaways
- **Simple**, lightweight, and **model-agnostic**.
- **Effective at small-to-moderate perturbation strengths** (small \( \epsilon \)).
- **Degrades gracefully** as attack strength increases.
- **Highly recommended** as a baseline defense or combined with other methods.

---

##  Quick Pseudocode
```python
# Noise Fusion function
def fuse_with_noise(image, alpha=0.2, stddev=255.0):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev)
    fused = tf.cast(image, tf.float32) * (1.0 - alpha) + noise * alpha
    fused = tf.clip_by_value(fused, 0.0, 255.0)
    return fused
```

---

##  References
- Gong, Z., Wang, W., & Ku, W.-S. (2017). **Noise Fusion for Detecting Adversarial Examples**. *arXiv preprint arXiv:1703.04618*.

---

