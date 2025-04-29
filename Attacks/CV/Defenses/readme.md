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

| Attack           | Distribution | ε (Epsilon) | α (Alpha) | Test Accuracy after Attack | Accuracy after Noise Fusion |
|:----------------:|:------------:|:-----------:|:---------:|:---------------------------:|:----------------------------:|
| FGSM             | Gaussian     | 0.007       | -         | 58%                         | 95.16%                      |
| FGSM             | Gaussian     | 0.01        | -         | 44%                         | 93.41%                      |
| FGSM             | Gaussian     | 0.03        | -         | 14%                         | 83.33%                      |
| FGSM             | Gaussian     | 0.1         | -         | 6%                          | 48.64%                      |
| PGD              | Gaussian     | 0.007       | 0.00175   | 35%                         | 93.22%                      |
| PGD              | Gaussian     | 0.01        | 0.0025    | 7%                          | 80.23%                      |
| PGD              | Gaussian     | 0.03        | 0.0075    | 3%                          | 67.05%                      |
| PGD              | Gaussian     | 0.1         | 0.025     | 0%                          | 42.83%                      |
| FGSM             | Uniform      | 0.007       | -         | 58%                         | 83.72%                      |
| FGSM             | Uniform      | 0.01        | -         | 44%                         | 81.59%                      |
| FGSM             | Uniform      | 0.03        | -         | 14%                         | 72.48%                      |
| FGSM             | Uniform      | 0.1         | -         | 6%                          | 42.64%                      |
| PGD              | Uniform      | 0.007       | 0.00175   | 35%                         | 68.99%                      |
| PGD              | Uniform      | 0.01        | 0.0025    | 7%                          | 39.92%                      |
| PGD              | Uniform      | 0.03        | 0.0075    | 3%                          | 25.39%                      |
| PGD              | Uniform      | 0.1         | 0.025     | 0%                          | 10.66%                      |
---

---
Poisson Results
| FGSM             | Poisson      | 0.007       | -         | 58%                          | 93.99%                      |
| FGSM             | Poisson      | 0.01        | -         | 44%                          | 92.44%                      |
| FGSM             | Poisson      | 0.03        | -         | 14%                          | 76.55%                      |
| FGSM             | Poisson      | 0.1         | -         | 6%                           | 42.83%                      |
| PGD              | Poisson      | 0.007       | 0.00175   | 35%                          | 46.12%                      |
| PGD              | Poisson      | 0.01        | 0.0025    | 7%                           | 7.36%                       |
| PGD              | Poisson      | 0.03        | 0.0075    | 3%                           | 2.91%                       |
| PGD              | Poisson      | 0.1         | 0.025     | 0%                           | 0.97%                       |


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

