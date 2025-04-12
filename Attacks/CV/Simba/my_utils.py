

import matplotlib.pyplot as plt
from PIL import Image

def display_ppm_image(file_path):
    """
    Display a .ppm image using Pillow and Matplotlib.

    Parameters:
    -----------
    file_path : str
        Path to the .ppm image file.
    """
    # Open the .ppm file
    with Image.open(file_path) as img:
        # Convert to a format Matplotlib can handle (e.g., RGB)
        #img_rgb = img.convert("RGB")

        # Display using matplotlib
        plt.imshow(img)
        plt.axis('off')  # Hide axis for cleaner display
        plt.title(f"Displaying: {file_path}")
        plt.show()



def convert_jfif_to_ppm(input_path, output_path):
    """
    Convert a .jfif (JPEG) file to a .ppm file using Pillow.

    Parameters:
    -----------
    input_path : str
        Path to the input .jfif file.
    output_path : str
        Path to the output .ppm file.
    """
    # Open the JFIF image
    with Image.open(input_path) as img:
        # Convert to RGB (PPM is typically an RGB format)
        img_rgb = img.convert("RGB")
        # Save as PPM
        img_rgb.save(output_path, format="PPM")

def predict_traffic_sign_all(img_input, model, class_mapping):
    """
    Accepts a tensor (rank 3 or 4), raw image bytes, or a file path.
    Prints the probability for each class and the predicted class.
    """
    # Get predictions
    prop = model.predict(img_input)
    print("Prediction shape:", prop.shape)

    # Assuming the output is of shape (1, num_classes)
    probabilities = prop[0]

    # Print probabilities for all classes
    for i, p in enumerate(probabilities):
        print(f"Class {i} ({class_mapping[i]}): {p:.4f}")

    # Determine the predicted class
    class_id = np.argmax(probabilities)
    print("Predicted class:", class_mapping[class_id])

    return class_id
def ppm_to_tensor_tf(file_path):
    """
    Load a .ppm file and convert it to a TensorFlow tensor.
    Returns a 4D tensor of shape (1, height, width, 3).
    """
    with Image.open(file_path) as img:
        # Convert the image to RGB (just in case it's not already)
        img_rgb = img.convert("RGB")

        # Convert the PIL image to a NumPy array
        img_np = np.array(img_rgb)

        # Convert the NumPy array to a TF tensor
        # Shape: (height, width, channels)
        img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

        # Optionally, add a batch dimension at the start
        img_tensor = tf.expand_dims(img_tensor, axis=0)  # shape -> (1, H, W, 3)

    return img_tensor

#########################################################  Example usage for display_ppm_image  function ##################################################
#display_ppm_image("00000.ppm")



#########################################################  Example usage for predict_traffic_sign function ##################################################
#image_path = "00000.ppm"
# Load and preprocess the image with the correct target size (240x240)
#image = load_ppm_image(image_path, target_size=(240, 240))
# Call your prediction function with the image, model, and class mapping
#predict_traffic_sign(image, model, GTSRB_CLASSES)

#########################################################  Example usage for convert_jfif_to_ppm  function ##################################################

#convert_jfif_to_ppm("Limit20.jfif", "ppm_limit20.ppm")
#new_photo="ppm_limit20.ppm"
#display_ppm_image(new_photo)
