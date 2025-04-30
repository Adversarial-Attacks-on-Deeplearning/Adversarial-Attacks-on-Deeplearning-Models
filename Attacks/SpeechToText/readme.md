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

## Usage

1. **Setup**:

   - Install dependencies: `pip install torch transformers jiwer numpy`.

   - Load a pre-trained Wav2Vec2 model and processor:

     ```python
     model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
     ```

2. **Prepare Data**:

   - Use a dataset like LibriSpeech (`librispeech_asr` from `datasets` library).
   - Extract audio (`audio_array`) and ground truth transcription.

3. **Run FGSM**:

   ```python
   audio_array = np.random.randn(16000)  # Example audio
   ground_truth = "This is a test"
   target_transcription = "Hello world"
   adversarial_waveform, ground_truth_wer, target_wer, adversarial_transcription = fgsm_attack(
       audio_array, ground_truth, target_transcription, model, processor, epsilon=0.3
   )
   print(f"Ground Truth WER: {ground_truth_wer:.2f}")
   print(f"Target WER: {target_wer:.2f}")
   ```

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



## Usage

1. **Setup**:

   - Install dependencies: `pip install torch transformers jiwer numpy`.

   - Load a pre-trained Wav2Vec2 model and processor:

     ```python
     model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
     ```

2. **Prepare Data**:

   - Use a dataset like LibriSpeech (`librispeech_asr` from `datasets` library).
   - Extract audio (`audio_array`) and ground truth transcription.

3. **Run PGD**:

   ```python
   audio_array = np.random.randn(16000)  # Example audio
   ground_truth = "This is a test"
   target_transcription = "Hello world"
   adversarial_waveform, ground_truth_wer, target_wer, adversarial_transcription = pgd_attack(
       audio_array, ground_truth, target_transcription, model, processor, epsilon=0.3, alpha=0.01, num_iter=10
   )
   print(f"Ground Truth WER: {ground_truth_wer:.2f}")
   print(f"Target WER: {target_wer:.2f}")
   ```

   

## References

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks*. arXiv preprint arXiv:1706.06083. https://arxiv.org/abs/1706.06083
2. Želasko, P., et al. (2021). *Adversarial Attacks and Defenses for Speech Recognition Systems*. arXiv preprint arXiv:2103.09095. https://arxiv.org/abs/2103.09095
3. Olivier, R., Abdullah, H., & Raj, B. (2023). *Transferable Adversarial Perturbations between Self-Supervised Speech Recognition Models*. arXiv preprint arXiv:2302.03487. https://arxiv.org/abs/2302.03487
4. Hugging Face Transformers Documentation. *Wav2Vec2*. https://huggingface.co/docs/transformers/model_doc/wav2vec2