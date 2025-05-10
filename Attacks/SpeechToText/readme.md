# Fast Gradient Sign Method (FGSM) Adversarial Attack Documentation

## Overview

The Fast Gradient Sign Method (FGSM) is a simple and computationally efficient adversarial attack technique used to generate adversarial examples that mislead machine learning models. In the context of automatic speech recognition (ASR), FGSM adds a small perturbation to an audio input to cause the model (e.g., Wav2Vec2) to produce incorrect transcriptions, either disrupting the correct output (denial-of-service, DOS) or forcing a specific target transcription (targeted attack). FGSM is a single-step attack that leverages the gradient of the model's loss function to craft the perturbation.

## How FGSM Works

FGSM generates an adversarial example by adding a scaled perturbation to the input audio, where the perturbation is derived from the sign of the gradient of the loss with respect to the input. The perturbation is constrained by a parameter `epsilon` to ensure it remains subtle and imperceptible or minimally perceptible to humans.

### Mathematical Formulation

Given:

- Input audio: ( x ) (a waveform, e.g., a 1D array of audio samples).
- Ground truth transcription: ( y_{\text{true}} ).
- Target transcription (for targeted attack): ( y_{\text{target}} ).
- Model: ( f(x) ), which outputs logits for transcription.
- Loss function: ( J(x, y) ), typically Connectionist Temporal Classification (CTC) loss for ASR.

The adversarial example ( x_{\text{adv}} ) is computed as:

[
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(x, y_{\text{target}}))
]

Where:

- ( \nabla_x J(x, y_{\text{target}}) ): Gradient of the loss with respect to the input ( x ), computed for the target transcription.
- ( \text{sign} ): Takes the sign of each gradient component (+1, -1, or 0).
- ( \epsilon ): Perturbation magnitude, controlling the strength of the attack (e.g., 0.01–0.3).

For a **non-targeted attack** (DOS), the perturbation is in the direction that maximizes the loss for the ground truth:

[
x_{\text{adv}} = x - \epsilon \cdot \text{sign}(\nabla_x J(x, y_{\text{true}}))
]

### Key Characteristics

- **Single-Step**: FGSM applies the perturbation in one step, making it fast but less precise than iterative methods like PGD.
- **L-infinity Norm**: The perturbation is bounded by ( \epsilon ) in the ( L_\infty ) norm, ensuring each audio sample changes by at most ( \epsilon ).
- **Effectiveness**: Best suited for DOS attacks, causing high Word Error Rates (WER) by disrupting transcriptions. Less effective for targeted attacks, especially on robust models like Wav2Vec2.



## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples*. arXiv preprint arXiv:1412.6572. https://arxiv.org/abs/1412.6572
2. Želasko, P., et al. (2021). *Adversarial Attacks and Defenses for Speech Recognition Systems*. arXiv preprint arXiv:2103.09095. https://arxiv.org/abs/2103.09095
3. Hugging Face Transformers Documentation. *Wav2Vec2*. https://huggingface.co/docs/transformers/model_doc/wav2vec2



--- 

# Projected Gradient Descent (PGD) Adversarial Attack Documentation

## Overview

Projected Gradient Descent (PGD) is an advanced adversarial attack technique used to generate adversarial examples that mislead machine learning models, particularly in automatic speech recognition (ASR) systems like Wav2Vec2. PGD is an iterative extension of the Fast Gradient Sign Method (FGSM), applying multiple small perturbations to the input audio to disrupt the correct transcription (denial-of-service, DOS) or force a specific target transcription (targeted attack). PGD is more powerful than FGSM due to its iterative nature, allowing finer control over the perturbation.

## How PGD Works

PGD iteratively updates the input audio by taking small steps in the direction of the gradient of the loss function, projecting the perturbation back into an allowed ( L_\infty )-norm ball defined by `epsilon`. This ensures the perturbation remains subtle while maximizing the attack's effectiveness.

### Mathematical Formulation

Given:

- Input audio: ( x ) (a waveform, e.g., a 1D array of audio samples).
- Ground truth transcription: ( y_{\text{true}} ).
- Target transcription (for targeted attack): ( y_{\text{target}} ).
- Model: ( f(x) ), which outputs logits for transcription.
- Loss function: ( J(x, y) ), typically Connectionist Temporal Classification (CTC) loss for ASR.
- Perturbation bound: ( \epsilon ).
- Step size: ( \alpha ).

The adversarial example ( x_{\text{adv}} ) is computed iteratively over ( T ) iterations:

1. Initialize: ( x_{\text{adv}}^{(0)} = x ).

2. For each iteration ( t = 1, \dots, T ):

   [

   x_{\text{adv}}^{(t)} = \text{Proj}

   {\epsilon}(x

   {\text{adv}}^{(t-1)} + \alpha \cdot \text{sign}(\nabla_x J(x_{\text{adv}}^{(t-1)}, y_{\text{target}})))

   ]

   Where:

   - ( \nabla_x J ): Gradient of the loss with respect to the current input.
   - ( \text{sign} ): Takes the sign of the gradient.
   - ( \text{Proj}*{\epsilon} ): Projects the perturbation to ensure ( ||x*{\text{adv}} - x||*\infty \leq \epsilon ), typically via clipping:
     [
     x*{\text{adv}}^{(t)} = \text{clip}(x_{\text{adv}}^{(t)}, x - \epsilon, x + \epsilon)
     ]

For a **non-targeted attack** (DOS), the perturbation maximizes the loss for the ground truth:

[
x_{\text{adv}}^{(t)} = \text{Proj}*{\epsilon}(x*{\text{adv}}^{(t-1)} - \alpha \cdot \text{sign}(\nabla_x J(x_{\text{adv}}^{(t-1)}, y_{\text{true}})))
]

### Key Characteristics

- **Iterative**: Multiple steps (e.g., 10–100 iterations) allow PGD to refine the perturbation, making it more effective than FGSM.
- **L-infinity Norm**: Perturbations are constrained within ( [-\epsilon, \epsilon] ), ensuring controlled changes.
- **Effectiveness**: Strong for both DOS (high ground truth WER) and targeted attacks (lower target WER), though targeted attacks may require more iterations or advanced techniques.




   

## References

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks*. arXiv preprint arXiv:1706.06083. https://arxiv.org/abs/1706.06083
2. Želasko, P., et al. (2021). *Adversarial Attacks and Defenses for Speech Recognition Systems*. arXiv preprint arXiv:2103.09095. https://arxiv.org/abs/2103.09095
3. Olivier, R., Abdullah, H., & Raj, B. (2023). *Transferable Adversarial Perturbations between Self-Supervised Speech Recognition Models*. arXiv preprint arXiv:2302.03487. https://arxiv.org/abs/2302.03487
4. Hugging Face Transformers Documentation. *Wav2Vec2*. https://huggingface.co/docs/transformers/model_doc/wav2vec2



# Cramér-IPM Adversarial Attack
## Theoretical Background
The Cramér Integral Probability Metric (Cramér-IPM) adversarial attack targets speech-to-text systems like Wav2Vec2 by crafting adversarial audio examples. It uses the Cramér distance, an Integral Probability Metric (IPM), to measure and minimize the statistical difference between the original and adversarial audio distributions. This ensures the adversarial audio remains perceptually similar to the original while causing the model to produce an incorrect transcription.
Key Concepts:

Adversarial Attack: Modifies inputs to mislead machine learning models, often subtly.
Cramér Distance: Quantifies the divergence between two probability distributions, constraining perturbations.
CTC Loss: Connectionist Temporal Classification loss, used to align the model's output with a target transcription.

The attack balances two goals:

Minimizing CTC loss to achieve the target transcription.
Minimizing Cramér distance to maintain audio similarity and robustness.

## Implementation Steps

Preprocessing:

Load and preprocess audio with the Wav2Vec2 processor.
Tokenize the target transcription for CTC loss.


Initialization:

Start with a copy of the original audio tensor.


Optimization Loop:

Compute CTC loss between the adversarial audio and target transcription.
Calculate Cramér distance between original and adversarial audio (e.g., using mean squared error as a proxy).
Compute gradients for both objectives.
Update the adversarial audio using gradient descent, scaled by epsilon, and clamp to valid audio range.


Post-processing:

Convert the adversarial tensor to a NumPy array for use.



## References

Müller, A. (1997). Integral probability metrics and their generating classes of functions. Advances in Applied Probability, 29(2), 429-443.
Carlini, N., & Wagner, D. (2018). Audio adversarial examples: Targeted attacks on speech-to-text. 2018 IEEE Security and Privacy Workshops (SPW), 1-7.
Baevski, A., et al. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. arXiv:2006.11477.



