import numpy as np
import tensorflow as tf

class ZOOAttack:
    def __init__(self, blackbox, input_shape, 
                 epsilon=0.5, lr=1.0, max_iter=1000, 
                 batch_size=128, attack_dim=32, 
                 hierarchical=True, use_adam=True):
        """
        Initialize ZOO attack with proper shape handling.
        """
        self.model = blackbox
        self.epsilon = epsilon
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.attack_dim = attack_dim
        self.hierarchical = hierarchical
        self.use_adam = use_adam
        
        # Ensure input_shape is (H, W, C) without batch dimension
        self.input_shape = tuple([dim for dim in input_shape if dim is not None][-3:])
        
        # ADAM parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _scale_perturbation(self, delta, target_shape):
        """Scale perturbation to target shape while maintaining 3 channels"""
        # Add batch dimension and convert to float32
        delta_expanded = tf.expand_dims(tf.cast(delta, tf.float32), axis=0)
        
        # Get target spatial dimensions (height, width)
        target_size = target_shape[:2]
        
        # Resize and remove batch dimension
        scaled = tf.image.resize(delta_expanded, target_size, method='bilinear')
        scaled = scaled[0].numpy()
        
        # Ensure output has the same number of channels as the input
        if scaled.shape[-1] != target_shape[-1]:
            scaled = np.repeat(scaled[..., np.newaxis], target_shape[-1], axis=-1)
        
        return scaled

    def _estimate_gradients(self, image, delta, target_class, c=1.0):
        """Batch gradient estimation with proper shape handling"""
        # Generate random indices for coordinate selection
        flat_delta = delta.reshape(-1)
        indices = np.random.choice(len(flat_delta), self.batch_size, replace=False)
        
        # Create perturbations
        delta_plus = delta.copy()
        delta_minus = delta.copy()
        
        # Apply perturbations
        delta_plus.flat[indices] += self.epsilon
        delta_minus.flat[indices] -= self.epsilon
        
        # Scale perturbations to input size
        scaled_plus = self._scale_perturbation(delta_plus, self.input_shape)
        scaled_minus = self._scale_perturbation(delta_minus, self.input_shape)
        
        # Generate perturbed images
        x_plus = np.clip(image + scaled_plus, 0, 255).astype(np.uint8)
        x_minus = np.clip(image + scaled_minus, 0, 255).astype(np.uint8)
        
        # Get predictions
        f_plus = self.model.predict(x_plus[np.newaxis, ...])
        f_minus = self.model.predict(x_minus[np.newaxis, ...])
        
        # Calculate losses (L2 in 0-255 range)
        l2_plus = np.sum(scaled_plus**2)
        l2_minus = np.sum(scaled_minus**2)
        
        # Hinge loss calculation
        hinge_plus = np.maximum(np.log(f_plus[0][:-1]).max() - np.log(f_plus[0][target_class]), -0.0)
        hinge_minus = np.maximum(np.log(f_minus[0][:-1]).max() - np.log(f_minus[0][target_class]), -0.0)
        
        # Compute gradients
        gradients = ((l2_plus + c*hinge_plus) - (l2_minus + c*hinge_minus)) / (2 * self.epsilon)
        
        # Create full gradient array
        grad = np.zeros_like(flat_delta)
        grad[indices] = gradients
        return grad.reshape(delta.shape)

    def attack(self, image, target_class):
        """Attack execution with proper shape handling"""
        # Ensure image is in 0-255 range and uint8
        original_image = np.array(image, dtype=np.uint8)
        if original_image.max() <= 1:
            original_image = (original_image * 255).astype(np.uint8)
            
        # Initialize perturbation in low-dim space
        current_dim = self.attack_dim
        delta = np.zeros((current_dim, current_dim, self.input_shape[-1]), dtype=np.float32)
        
        # ADAM states
        m = np.zeros_like(delta)
        v = np.zeros_like(delta)
        t = 0
        
        for step in range(self.max_iter):
            # Estimate gradients
            grad = self._estimate_gradients(original_image, delta, target_class)
            
            # ADAM update
            if self.use_adam:
                t += 1
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad**2)
                
                # Bias correction
                m_hat = m / (1 - self.beta1**t)
                v_hat = v / (1 - self.beta2**t)
                
                delta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                delta -= self.lr * grad
            
            # Hierarchical scaling
            if self.hierarchical and step % 200 == 0 and current_dim < self.input_shape[0]:
                new_dim = min(current_dim * 2, self.input_shape[0])
                delta = self._scale_perturbation(delta, (new_dim, new_dim, self.input_shape[-1]))
                current_dim = new_dim
                m = self._scale_perturbation(m, (new_dim, new_dim, self.input_shape[-1]))
                v = self._scale_perturbation(v, (new_dim, new_dim, self.input_shape[-1]))
            
            # Generate adversarial example
            scaled_delta = self._scale_perturbation(delta, self.input_shape)
            adversarial = np.clip(original_image.astype(np.float32) + scaled_delta, 0, 255).astype(np.uint8)
            
            # Check success
            pred = np.argmax(self.model.predict(adversarial[np.newaxis, ...]))
            if pred == target_class:
                print(f"Attack succeeded at step {step}")
                return adversarial
        
        return adversarial