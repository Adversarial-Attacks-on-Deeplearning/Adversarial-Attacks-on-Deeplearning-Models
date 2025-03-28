import tensorflow as tf
from GTSRB_utils import GTSRB_CLASSES, predict_traffic_sign, create_subset_loader, load_ppm_image

# Example usage:
image_path = "ppm_limit20.ppm"
model = tf.keras.models.load_model("TrafficSigns_EfficientNetB1.keras")  # Load your trained model
image_tensor = load_ppm_image(image_path, target_size=(240, 240))
true_label = predict_traffic_sign(image_tensor, model, GTSRB_CLASSES)
adv_image_tensor = simba_attack(image_tensor, true_label, model, num_iters=2000, epsilon=50, print_every=10)
#
# # Optionally, save or visualize the adversarial image:
adv_image = adv_image_tensor.numpy()[0].astype(np.uint8)
adv_pil = Image.fromarray(adv_image)
adv_pil.save("adv_image.ppm")
