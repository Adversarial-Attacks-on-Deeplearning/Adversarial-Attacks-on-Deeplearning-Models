import tensorflow as tf

class BlackBox:
    def __init__(self, model, input_size=(240, 240), num_classes=1000):
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes

    def predict(self, x):
        x = self._preprocess(x)
        probs = self.model.predict(x, batch_size=1, verbose=0)
        return probs

    def _preprocess(self, x):
        # Ensure input is a TensorFlow tensor
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x)
        
        # Add missing dimensions (batch and channels) if needed
        if len(x.shape) == 3:  # Single image (H, W, C)
            x = tf.expand_dims(x, axis=0)  # Add batch dim: (1, H, W, C)
        elif len(x.shape) == 2:  # Grayscale image (H, W)
            x = tf.expand_dims(x, axis=-1)  # Add channel dim: (H, W, 1)
            x = tf.expand_dims(x, axis=0)   # Add batch dim: (1, H, W, 1)
        
        # Convert grayscale to RGB (if needed)
        if x.shape[-1] == 1:
            x = tf.repeat(x, repeats=3, axis=-1)  # (1, H, W, 3)
        
        # Resize to the model's expected input size
        x = tf.image.resize(x, self.input_size)
        
        # Normalize pixel values (example for [0, 1] scaling)
        # x = x / 255.0
        
        return x
    


