

### **The Simple Black-box Attack (SimBA): A Comparative Analysis of Pixel Space and DCT Space Implementations**

#### **Introduction to SimBA**  
The **Simple Black-box Attack (SimBA)** is a query-efficient algorithm designed to generate adversarial examples in a black-box setting, where the attacker has no access to the target model's internal parameters or training data. Instead, the attacker can only query the model and observe its output (e.g., confidence scores or class probabilities). SimBA operates on an iterative principle: it randomly samples a perturbation vector from a predefined orthonormal basis and applies it to the input image, either adding or subtracting the perturbation to minimize the model's confidence in the correct classification.  

The core intuition behind SimBA is that near a decision boundary, even approximate perturbation directions can effectively induce misclassification.  

#### **Algorithmic Framework**  
Formally, SimBA takes as input:  
- A target image-label pair \((x, y)\).  
- A set of orthonormal candidate vectors \(Q\).  
- A step size \(\epsilon > 0\).  

At each iteration, the algorithm:  
1. Randomly selects a vector \(q \in Q\) (without replacement).  
2. Queries the model to evaluate \(p(y | x + \delta + \epsilon q)\) and \(p(y | x + \delta - \epsilon q)\), where \(\delta\) is the accumulated perturbation.  
3. Updates \(\delta\) by adding or subtracting \(\epsilon q\) if either operation reduces \(p(y | x)\).  

This process continues until the model misclassifies the perturbed image or a query budget is exhausted. Due to the orthonormality of \(Q\), the \(L_2\)-norm of the perturbation after \(T\) updates is bounded by \(\|\delta_T\|_2 \leq \sqrt{T}\epsilon\).  

#### **Pixel Space vs. DCT Space Implementations**  
SimBA can be implemented using different orthonormal bases, with two primary variants being:  
1. **SimBA in Pixel Space**  
2. **SimBA in Discrete Cosine Transform (DCT) Space (SimBA-DCT)**  

The key distinctions between these approaches are outlined below.  

##### **1. SimBA in Pixel Space**  
- **Basis Selection**: The orthonormal basis \(Q\) is the standard basis \(I\), where each vector corresponds to a single pixel in the image.  
- **Perturbation Mechanism**: Each iteration modifies the intensity of a randomly selected pixel by \(\epsilon\).  
- **Attack Characteristics**:  
  - Operates as an implicit \(L_0\) attack, aiming to alter as few pixels as possible.  
  - Empirical studies indicate that approximately **73% of randomly sampled pixel-space directions** initially reduce the true class probability.  
  - Perturbations are **sparse but perceptually sharp**, as changes are localized to specific pixels.  

##### **2. SimBA in DCT Space (SimBA-DCT)**  
- **Basis Selection**: The orthonormal basis \(Q_{DCT}\) consists of a subset of low-frequency components derived from the Discrete Cosine Transform (DCT).  
- **Perturbation Mechanism**:  
  - Perturbs the DCT coefficients of the image, focusing on low-frequency components.  
  - Modifications are transformed back to pixel space before querying the model.  
- **Attack Characteristics**:  
  - **Higher initial efficacy**: ~**98% of low-frequency DCT directions** decrease the true class probability, with steeper descent compared to pixel space.  
  - Perturbations are **sparse in frequency space** but **diffuse in pixel space**, often resulting in subtler, less perceptible changes.  
  - **Faster initial convergence** and lower median query requirements for successful attacks.  
  - **Potential limitation**: Restricting perturbations to low frequencies may reduce success rates for some images. To mitigate this, SimBA-DCT can dynamically expand the basis by incorporating additional low-frequency components.  

#### **Comparative Summary**  
| **Aspect**               | **SimBA (Pixel Space)**          | **SimBA-DCT**                     |
|--------------------------|----------------------------------|-----------------------------------|
| **Basis**                | Standard pixel basis             | Low-frequency DCT components      |
| **Perturbation Nature**  | Sparse, pixel-level changes      | Diffuse, frequency-based changes  |
| **Initial Efficacy**     | ~73% of directions effective     | ~98% of directions effective      |
| **Convergence Speed**    | Slower initial descent           | Faster initial convergence        |
| **Success Rate**         | Higher eventual success          | Slightly lower for some images    |
| **Perceptibility**       | More perceptible (sharp changes) | Less perceptible (smooth changes) |

#### **Conclusion**  
Both variants of SimBA adhere to the same core algorithmic framework but differ significantly in their choice of perturbation basis and the resulting adversarial characteristics. **SimBA in pixel space** produces localized, sparse perturbations, while **SimBA-DCT** generates more distributed and imperceptible modifications by operating in the frequency domain. While SimBA-DCT often achieves greater query efficiency, its restriction to low-frequency components may limit its success rate in certain cases. The choice between these approaches depends on the specific adversarial objectives, including perturbation subtlety, query budget, and desired attack success rate.  

