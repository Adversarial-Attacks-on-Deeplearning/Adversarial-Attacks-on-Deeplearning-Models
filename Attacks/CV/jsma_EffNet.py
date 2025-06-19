import tensorflow as tf
import numpy as np
import pandas as pd

def compute_output_grads(model, image):
    """
    Compute the element-wise grads matrix of the model output with respect to the input image.
    Args:
        input image with dimensions: [H, W, C]
        tf model
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        output = model(tf.expand_dims(image, axis=0))  # Forward pass
    element_wise_grad = tape.jacobian(output, image, experimental_use_pfor=True)
    return tf.squeeze(element_wise_grad, axis=0)  # Remove batch dimension

def saliency_map(jacobian, target_label):
    """
    Compute the saliency map to determine which pixels to perturb.
    Args:
        the element-wise gradients matrix
        the targetted label
    """
     # Get gradient for target class [H, W, C]
    J_t = jacobian[target_label]  
    
    # Sum gradients for all other classes [H, W, C]
    sum_J_other = tf.reduce_sum(jacobian, axis=0) - J_t
    
    # Compute conditions
    cond1 = J_t < 0                          # Condition 1: J_t < 0
    cond2 = sum_J_other > 0                  # Condition 2: Sum of other grads > 0
    zero_mask = tf.logical_or(cond1, cond2)  # Where to set saliency to 0
    
    # Compute saliency values
    saliency_values = J_t * tf.abs(sum_J_other)
    
    # Apply conditions
    saliency_map = tf.where(zero_mask, 
                          tf.zeros_like(saliency_values), 
                          saliency_values)
    
    return saliency_map

def jsma_attack(model, image, target_label, gamma, theta, num_pixels):
    """
    Performs a JSMA attack on multi-class classification model.
    
    Args:
        model: The TensorFlow/Keras model.
        image: The input image (tensor of shape [H, W, C]) (un-normalized).
        target_label: The desired class output (class id).
        theta: The perturbation step size.
        gamma: Maximum total distortion (percentage of modified pixels).
        num_pixels: Number of pixels to perturb at each iteration.
    
    Returns:
        The adversarial image.
    """
    adversarial = tf.identity(image)
    itr = 0
    distortion = 0.0
    class_id = 0
    # print initial prediction
    pred = model(tf.expand_dims(image, axis=0))
    class_id = np.argmax(pred)
    print(f"prediction before attack: {class_id}")
    
    while distortion < gamma and class_id != target_label and itr < 15: # stop if maximum distortion reached or attack succeeded or max. num. of iterartions reached
        # step 1: Compute the output grads with respect to input image
        grads = compute_output_grads(model, adversarial)  
       
        # step 2: Compute saliency map to perturb most important pixels
        saliency = saliency_map(grads, target_label)  
       
        # step 3: Get maximum pixels value to perturb.
        # Flatten the saliency map to 1D and get top N values/indices
        flat_saliency = tf.reshape(saliency, [-1])  # Shape [H*W*C]
        top_values, flat_indices = tf.math.top_k(flat_saliency, k=num_pixels)
        # Convert flat indices to 3D coordinates (h, w, c)
        h = flat_indices // (saliency.shape[1] * saliency.shape[2])
        remainder = flat_indices % (saliency.shape[1] * saliency.shape[2])
        w = remainder // saliency.shape[2]
        c = remainder % saliency.shape[2]
        
        # Stack into [N, 3] tensor
        top_indices = tf.stack([h, w, c], axis=1)
        
        # Create updates tensor (shape [N])
        updates = tf.ones([tf.shape(top_indices)[0]]) * theta
        
        # Apply scatter add operation
        adversarial = tf.tensor_scatter_nd_add(
            adversarial,
            top_indices,  # [N, 3] indices
            updates       # [N] values to add
        )
        # Clip to maintain constraints 
        adversarial = tf.clip_by_value(adversarial, 0, 255)
        
        # Check if misclassification occurs
        pred = model(tf.expand_dims(adversarial, axis=0))
        class_id = np.argmax(pred)
        print(f"current prediction : {class_id}")
        if (class_id == target_label):   
            print(f"Attacked successfully, predicted class: {class_id}")
            
        itr+=1
    if (class_id != target_label):   
        print(f"Attack failed :(")
    return adversarial



adv_img = jsma_attack(model, image,target_label= 4,gamma = 0.20, theta = 255,num_pixels = 20)
# save image
tf.keras.utils.save_img(f"jsma/adversarial_images/{0:05d}.jpg", adv_img)