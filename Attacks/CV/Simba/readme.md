**SimBA: Simple Black-box Adversarial Attack**

---

## Overview

**SimBA** (Simple Black-box Attack) is a highly query-efficient method for generating adversarial perturbations against black-box image classifiers. By leveraging only continuous confidence scores from the target model, SimBA iteratively searches along orthonormal directions to find imperceptible modifications that flip the model’s prediction.

This README provides:
- A clear description of the SimBA algorithm
- Basis selection options (pixel vs. DCT)
- Hyperparameter choices and trade-offs
- Guidance on implementation principles

---

## Algorithm Description

Given an input image **x** correctly classified as label **y**, SimBA seeks a perturbation **δ** such that the perturbed image **x′ = x + δ** is misclassified, while minimizing both the number of model queries and the distortion ‖δ‖₂. SimBA operates under the following black-box threat model:

- **Access**: Only query the model with an image to receive the confidence score _pₕ(y | x)_.
- **Budget**: Minimize the number of queries to the model.

### Key Steps

1. **Define Search Directions**: Construct an orthonormal basis **Q** of unit vectors. Common choices include the pixel basis or a subset of low-frequency DCT basis vectors.
2. **Initialize**: Set the perturbation **δ** to zero and record the original confidence _p₀ = pₕ(y | x)_.
3. **Iterative Search**:
   - Randomly select a direction **q** from **Q** (without replacement).
   - Probe the model at **x + δ + ε q** and at **x + δ − ε q**, where **ε** is a fixed step size.
   - Identify which probe most reduces the confidence of the true class.
   - Update **δ** by moving in that direction if it decreases confidence.
4. **Termination**: Stop when the model’s predicted label on **x + δ** differs from **y**, or when a maximum query budget is reached.

Because the directions are orthonormal, after **T** successful steps of size **ε**, the perturbation norm satisfies:

\[
\|δ\|₂ \le \sqrt{T}\,ε.
\]

This relationship provides direct control over the trade-off between distortion and query count.

---

## Basis Selection

SimBA can operate in any orthonormal basis. Two common choices:

1. **Pixel (Cartesian) Basis**
   - Each direction corresponds to modifying a single pixel channel.
   - Perturbations are sparse in pixel space (low L₀ norm).
   - Robust: high final success rate but may require more queries or yield higher L₂ distortion.

2. **Low-Frequency DCT Basis**
   - Directions are 2D DCT basis vectors sorted by increasing frequency.
   - Restrict to the lowest-frequency coefficients (e.g., first 1/8th), adding more if needed.
   - Perturbations are smooth and global, often producing steeper confidence drops per query.
   - May fail on some images if omitted frequencies are crucial.

**Trade-offs**:
- **Query Efficiency**: DCT often converges faster in early iterations.
- **Coverage**: Pixel basis covers all directions, ensuring high success rate.
- **Distortion**: DCT typically achieves lower L₂ norms.

---

## Hyperparameters and Trade-offs

- **Step size (ε)**: Controls the magnitude of each update. Larger ε reduces the number of iterations but increases distortion; smaller ε yields finer control at the cost of more queries.
- **Frequency fraction (r)** for DCT: Determines how many low-frequency directions to include. Smaller r focuses on very low frequencies; larger r increases directional coverage.
- **Maximum queries (B)**: Sets the budget for model interactions. Choose B based on application constraints and acceptable distortion.

---

## Implementation Guidance

- **Efficiency**: Precompute and store the orthonormal basis vectors to avoid runtime overhead.
- **Randomization**: Shuffle the search directions to prevent worst-case ordering and to improve average-case performance.
- **Early Stopping**: After each update, check if the model’s predicted label has changed to terminate the attack promptly.
- **Numerical Stability**: Clip pixel values to valid ranges (e.g., [0, 1]) after each update.

---

## References

- **SimBA Paper**: Chuan Guo et al., "Simple Black-box Adversarial Attacks" (ICML 2019).
- **Low-Frequency Perturbations**: Guo et al., "Low Frequency Adversarial Perturbations" (2018).

---


