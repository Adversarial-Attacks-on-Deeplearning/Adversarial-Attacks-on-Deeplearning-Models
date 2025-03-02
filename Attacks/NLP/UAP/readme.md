# Universal Adversarial Perturbation (UAP) for Text Classifiers

## Overview
This implementation focuses on generating **universal adversarial perturbations (UAPs)** for text classifiers. Unlike traditional adversarial attacks that optimize a perturbation for each input, **UAPs are data-independent** and can be applied to *any* input to fool the classifier with high probability. This approach **inserts a sequence of words** into the input text to cause misclassification.

## Key Concepts

- **Adversarial Perturbation**: A small, carefully crafted modification to the input that causes the classifier to misclassify it. In this case, the modification is the insertion of specific words.
- **Universal Adversarial Perturbation (UAP)**: A single perturbation that can be applied to *any* input to fool the classifier. It is data-independent, meaning it is not optimized for a specific input sample.
- **Targeted vs. Non-Targeted Attacks**:
  - **Non-targeted attack**: The goal is to make the classifier predict *any* incorrect label. The UAP is chosen to **maximize the loss** associated with the correct class.
  - **Targeted attack**: The goal is to make the classifier predict a *specific* incorrect label. The UAP is chosen to **minimize the loss** associated with a particular, incorrect target class.

## Algorithm

1. **Initialization**: Start with a random sequence of words as the initial adversarial perturbation.
2. **Iterative Optimization**: Repeat the following steps until a satisfactory UAP is found:
   - **Gradient Calculation**: For each word in the adversarial sequence, compute the gradient of the loss function with respect to the word's embedding. This gradient indicates the direction in the embedding space that will increase the loss for the correct class (non-targeted) or decrease the loss for the target class (targeted).
   - **Embedding Update**: Update the word's embedding by moving it in the direction of the gradient, scaled by a learning rate \(\alpha\).
   - **Projection to Vocabulary**: Project the updated embedding back into the vocabulary space by finding the nearest word in the vocabulary (based on cosine similarity) to the updated embedding. This ensures the perturbation remains a valid word.
   - **UAP Adjustment**: Replace the original word in the adversarial sequence with the word found in the vocabulary.
3. **Insertion**: Concatenate the resulting adversarial sequence to the input text at a specific location (e.g., beginning, end, or middle).

## Important Considerations

- **Vocabulary Handling**: The adversarial embedding must be mapped to a word in the vocabulary. Strategies for handling out-of-vocabulary words include falling back to a generic token (e.g., "<UNK>") or finding the next closest in-vocabulary word.
- **Learning Rate**: The learning rate \(\alpha\) controls the step size when updating word embeddings. Choosing an appropriate learning rate is crucial for convergence and the quality of the generated UAP.
- **Insertion Location**: The location where the adversarial words are inserted can affect the success of the attack. Some models are more sensitive to perturbations at the beginning of the input sequence.
- **Effectiveness**: Inserting even a single adversarial word can significantly reduce the accuracy of text classifiers. The effectiveness of the UAP depends on the architecture of the text classifier and the dataset used.

