import tensorflow as tf
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt



def fitness_function(candidate, original_img, model, preprocess, true_label, attack_type='untargeted', target_label=None):
    """
    Fitness function for differential evolution.
    
    For an untargeted attack, it returns the probability of the true label (which we want to minimize).
    For a targeted attack, it returns the negative probability of the target label (so that maximizing
    the target label probability is equivalent to minimizing the returned value).
    
    Parameters:
        candidate (list or np.ndarray): Candidate solution [x, y, R, G, B].
        original_img (np.ndarray or tf.Tensor): Original image.
        model (tf.keras.Model): Pretrained model.
        preprocess (callable): Preprocessing function.
        true_label (int): The original (true) label index.
        attack_type (str): 'untargeted' or 'targeted'.
        target_label (int or None): The target label index (required if attack_type == 'targeted').
        
    Returns:
        float: The objective value.
    """
    perturbed_img = original_img.copy()
    h, w, _ = perturbed_img.shape

    # Determine the pixel coordinates.
    x_coord = int(np.clip(round(candidate[0]), 0, w - 1))
    y_coord = int(np.clip(round(candidate[1]), 0, h - 1))
    
    # Determine the new RGB values.
    r_val = np.clip(candidate[2], 0, 255)
    g_val = np.clip(candidate[3], 0, 255)
    b_val = np.clip(candidate[4], 0, 255)
    
    # Modify the designated pixel.
    perturbed_img[y_coord, x_coord, :] = [r_val, g_val, b_val]
    
    # Preprocess the image and get predictions.
    input_perturbed = preprocess(np.expand_dims(perturbed_img, axis=0))
    preds = model.predict(input_perturbed)
    
    if attack_type == 'targeted':
        if target_label is None:
            raise ValueError("For a targeted attack, target_label must be provided.")
        # For a targeted attack, maximize the probability of the target label.
        # Differential evolution minimizes the function, so we return negative probability.
        return -preds[0, target_label]
    else:
        # For an untargeted attack, minimize the probability of the true label.
        return preds[0, true_label]

def perform_one_pixel_attack(original_img, model, preprocess, true_label, 
                             maxiter=75, popsize=400, tol=1e-5, 
                             attack_type='untargeted', target_label=None):
    """
    Uses differential evolution to search for the optimal one-pixel perturbation.
    
    Parameters:
        original_img (np.ndarray or tf.Tensor): Original image.
        model (tf.keras.Model): Pretrained model.
        preprocess (callable): Preprocessing function.
        true_label (int): True label index.
        maxiter (int): Maximum iterations.
        popsize (int): Population size.
        tol (float): Convergence tolerance.
        attack_type (str): 'untargeted' or 'targeted'.
        target_label (int or None): The target label index (required if attack_type == 'targeted').
    
    Returns:
        result: The optimization result containing the best candidate.
    """
    h, w, _ = original_img.shape
    # Define the bounds: x, y coordinates and RGB values.
    bounds = [
        (0, w - 1),   # x-coordinate
        (0, h - 1),   # y-coordinate
        (0, 255),     # R value
        (0, 255),     # G value
        (0, 255)      # B value
    ]
    
    result = differential_evolution(
        lambda candidate: fitness_function(candidate, original_img, model, preprocess, 
                                           true_label, attack_type=attack_type, target_label=target_label),
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        disp=True
    )
    return result

def apply_attack(original_img, candidate):
    """
    Applies the one-pixel perturbation specified by 'candidate' to the original image.
    
    Parameters:
        original_img (np.ndarray or tf.Tensor): The original image.
        candidate (list or np.ndarray): Candidate solution [x, y, R, G, B].
    
    Returns:
        attacked_img (np.ndarray): The adversarially modified image.
    """
    attacked_img = original_img.copy()
    h, w, _ = attacked_img.shape
    
    x_coord = int(np.clip(round(candidate[0]), 0, w - 1))
    y_coord = int(np.clip(round(candidate[1]), 0, h - 1))
    r_val = np.clip(candidate[2], 0, 255)
    g_val = np.clip(candidate[3], 0, 255)
    b_val = np.clip(candidate[4], 0, 255)
    
    attacked_img[y_coord, x_coord, :] = [r_val, g_val, b_val]
    return attacked_img
