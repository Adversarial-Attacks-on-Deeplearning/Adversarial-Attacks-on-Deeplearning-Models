# README: Character-Level Adversarial Attacks

## Overview
This notebook implements a **black-box** character-level adversarial attack to deceive sentiment analysis models by modifying text at the character level. 
This notebook explores adversarial attacks on deep learning-based sentiment analysis models. Specifically, it implements **character-level adversarial attacks** to alter input text in ways that deceive the model into flipping its sentiment prediction. These alterations exploit weaknesses in how models process textual data, often causing them to misinterpret words and misclassify sentiment. By introducing small perturbations that remain understandable to humans, the attack highlights the model's reliance on exact word structures rather than contextual meaning.

## Attack Type
### Character-Level Attack
This attack modifies individual characters in words to create adversarial examples while preserving human readability. Techniques used include:
- **Insertion**: Adding extraneous characters (e.g., `happy` → `haappy`)
- **Deletion**: Removing key characters (e.g., `great` → `gret`)
- **Substitution**: Replacing characters with visually or phonetically similar ones (e.g., `good` → `g00d`)
- **Transposition**: Swapping adjacent characters (e.g., `bad` → `bda`)

#### Algorithm:
1. Tokenize the input text into words.
2. Randomly select a subset of words for modification.
3. Apply one or more character-level perturbations per selected word.
4. Reconstruct the text and pass it to the model for evaluation.

## Effects on Test Sentences
The attacks significantly impact sentiment classification. For example:

| Original Sentence | Modified Sentence | Model Prediction Before | Model Prediction After |
|------------------|------------------|-------------------------|------------------------|
| "The food was great and service was excellent." | "The food wsa gret and service was exllent." | Positive | Negative |
| "I hated this movie, it was terrible." | "I htaed this movie, it wsa terible." | Negative | Positive |

## Conclusion
These attacks reveal vulnerabilities in sentiment analysis models. Character-level attacks exploit surface-level perturbations, demonstrating the need for adversarial training to improve model robustness.

