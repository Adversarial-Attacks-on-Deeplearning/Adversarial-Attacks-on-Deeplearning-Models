import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

os.environ['QT_QPA_PLATFORM'] = 'xcb'

class AdversarialPatchGenerator:
    def __init__(self, target_class=11, patch_ratio=0.2, grid_size=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = YOLO('yolov8n-seg.pt').to(self.device)
        self.model.eval()  # Ensure model is in evaluation mode
        self.target_class = target_class
        self.patch_ratio = patch_ratio
        self.grid_size = grid_size  # Number of grid cells in each dimension
        self.loss_history = []
        
        # Specific parameters for disappearance attack
        self.confidence_threshold = 0.25  # Target threshold for "disappearance"
        self.use_adaptive_learning = True  # Adjust learning rate based on progress

    def apply_patch(self, image_tensor, patch, bbox, grid_position=None):
        """
        Apply patch to image with perspective transformation for natural appearance
        
        Args:
            image_tensor: Input image tensor
            patch: Adversarial patch tensor
            bbox: Object bounding box (x1, y1, x2, y2)
            grid_position: Tuple (row, col) indicating position in the grid
                           None means center position
        """
        x1, y1, x2, y2 = bbox
        H, W = image_tensor.shape[1:]
        
        # Create a copy of the original image to avoid in-place modifications
        result_tensor = image_tensor.clone()
        
        # Calculate object dimensions and center
        obj_width = x2 - x1
        obj_height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate patch size based on ratio
        patch_size = int(obj_width * np.sqrt(self.patch_ratio))
        
        # Calculate patch placement coordinates based on grid position
        if grid_position is not None:
            row, col = grid_position
            # Calculate cell width and height
            cell_width = obj_width / self.grid_size
            cell_height = obj_height / self.grid_size
            
            # Calculate position within object based on grid
            rel_x = (col + 0.5) / self.grid_size  # Center of the cell
            rel_y = (row + 0.5) / self.grid_size  # Center of the cell
            
            # Convert relative position to absolute position
            patch_center_x = x1 + obj_width * rel_x
            patch_center_y = y1 + obj_height * rel_y
        else:
            # Default to center if no grid position specified
            patch_center_x = center_x
            patch_center_y = center_y
        
        # Create perspective transformation
        perspective_factor = 0.2 * ((patch_center_x / W) - 0.5)
        
        # Source points (original patch corners)
        src_points = torch.tensor([
            [0, 0],
            [patch_size, 0],
            [patch_size, patch_size],
            [0, patch_size]
        ], dtype=torch.float32, device=self.device)
        
        # Destination points with perspective effect
        perspective_shift = patch_size * perspective_factor
        dest_points = torch.tensor([
            [0, 0],
            [patch_size, perspective_shift],
            [patch_size, patch_size],
            [0, patch_size - perspective_shift]
        ], dtype=torch.float32, device=self.device)
        
        # Get perspective transformation matrix
        M = self._get_perspective_transform(src_points, dest_points)
        
        # Create a grid for the transformation
        grid = F.affine_grid(
            M.unsqueeze(0), 
            torch.Size((1, 3, patch_size, patch_size)),
            align_corners=True
        )
        
        # Apply perspective transformation to patch
        transformed_patch = F.grid_sample(
            patch.unsqueeze(0), 
            grid,
            align_corners=True,
            mode='bilinear'
        )[0]
        
        # Calculate patch placement coordinates (top-left corner)
        patch_x1 = int(patch_center_x - patch_size / 2)
        patch_y1 = int(patch_center_y - patch_size / 2)
        patch_x2 = patch_x1 + patch_size
        patch_y2 = patch_y1 + patch_size
        
        # Create binary mask for patch placement
        mask = torch.zeros((H, W), device=self.device)
        
        # Set mask values within patch bounds, with boundary checks
        valid_y1 = max(0, patch_y1)
        valid_y2 = min(H, patch_y2)
        valid_x1 = max(0, patch_x1)
        valid_x2 = min(W, patch_x2)
        
        # Calculate corresponding patch coordinates
        patch_valid_y1 = max(0, valid_y1 - patch_y1)
        patch_valid_y2 = patch_size - max(0, patch_y2 - valid_y2) if patch_y2 > H else patch_size
        patch_valid_x1 = max(0, valid_x1 - patch_x1)
        patch_valid_x2 = patch_size - max(0, patch_x2 - valid_x2) if patch_x2 > W else patch_size
        
        # Check if we have a valid region to work with
        if valid_y2 <= valid_y1 or valid_x2 <= valid_x1:
            # Return original image if no valid region
            return image_tensor
        
        # Create a binary mask with sharp edges
        # Set the mask to 1.0 for the entire patch area
        mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.0
        
        # Create an empty canvas for the patch
        patch_full = torch.zeros((3, H, W), device=self.device)
        
        # Place transformed patch on the canvas
        if valid_y2 > valid_y1 and valid_x2 > valid_x1:
            patch_valid_height = patch_valid_y2 - patch_valid_y1
            patch_valid_width = patch_valid_x2 - patch_valid_x1
            
            if patch_valid_height == valid_y2 - valid_y1 and patch_valid_width == valid_x2 - valid_x1:
                patch_full[:, valid_y1:valid_y2, valid_x1:valid_x2] = transformed_patch[
                    :,
                    patch_valid_y1:patch_valid_y2,
                    patch_valid_x1:patch_valid_x2
                ]
            else:
                # Resize the patch section to match the valid region
                patch_section = transformed_patch[:, patch_valid_y1:patch_valid_y2, patch_valid_x1:patch_valid_x2]
                resized_patch = F.interpolate(
                    patch_section.unsqueeze(0),
                    size=(valid_y2 - valid_y1, valid_x2 - valid_x1),
                    mode='bilinear',
                    align_corners=False
                )[0]
                patch_full[:, valid_y1:valid_y2, valid_x1:valid_x2] = resized_patch
        
        # Apply mask as a simple binary operation without any additional effects
        mask = mask.unsqueeze(0)
        result_tensor = result_tensor * (1 - mask) + patch_full * mask
        
        # Ensure values stay in valid range [0,1]
        result_tensor = torch.clamp(result_tensor, 0, 1)
        
        return result_tensor

    def _get_perspective_transform(self, src, dst):
        """Compute perspective transform matrix similar to cv2.getPerspectiveTransform"""
        # This is a simplified version that works for our case
        # Adjusting to create an affine transformation matrix for grid_sample
        src_mean = src.mean(dim=0, keepdim=True)
        dst_mean = dst.mean(dim=0, keepdim=True)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        
        # Calculate transformation
        s_xx = (src_centered[:, 0] * dst_centered[:, 0]).sum()
        s_xy = (src_centered[:, 0] * dst_centered[:, 1]).sum()
        s_yx = (src_centered[:, 1] * dst_centered[:, 0]).sum()
        s_yy = (src_centered[:, 1] * dst_centered[:, 1]).sum()
        
        s_x = src_centered[:, 0].pow(2).sum()
        s_y = src_centered[:, 1].pow(2).sum()
        
        if s_x > 0:
            a = torch.sqrt(s_xx**2 + s_xy**2) / s_x
            b = (s_xx * s_yx + s_xy * s_yy) / (s_x * s_y)
        else:
            a = 1.0
            b = 0.0
        
        if s_y > 0:
            d = torch.sqrt(s_yx**2 + s_yy**2) / s_y
        else:
            d = 1.0
        
        if s_xx < 0:
            a = -a
        if s_yy < 0:
            d = -d
        
        # Create affine matrix for grid_sample
        theta = torch.zeros(2, 3, device=self.device)
        theta[0, 0] = a
        theta[0, 1] = b
        theta[1, 0] = b
        theta[1, 1] = d
        
        # Add translation
        theta[0, 2] = dst_mean[0, 0] - (a * src_mean[0, 0] + b * src_mean[0, 1])
        theta[1, 2] = dst_mean[0, 1] - (b * src_mean[0, 0] + d * src_mean[0, 1])
        
        return theta

    def calculate_disappearance_loss(self, results, target_class):
        """
        Implement the disappearance attack loss function from the paper:
        Jd(x, y) = max_{s∈S2, b∈B} P(s, b, y, fθ(x))
        
        Where:
        - s represents grid cells 
        - b represents bounding boxes
        - y is the target class
        - fθ(x) is the output of the object detector
        - P(·) extracts the probability of the target class
        
        Returns both the loss tensor (for backprop) and the max confidence (for monitoring)
        """
        # Initialize with default values
        max_conf = 0.0
        found_target = False
        
        # Extract all boxes and their confidences for the target class
        target_confidences = []
        
        # Check all detected objects
        for i in range(len(results[0].boxes)):
            cls = int(results[0].boxes.cls[i].item())
            if cls == target_class:
                conf = results[0].boxes.conf[i]
                target_confidences.append(conf)
                found_target = True
        
        if found_target:
            # Convert to tensor if we have confidences
            if target_confidences:
                confidences_tensor = torch.stack(target_confidences)
                # Get maximum confidence as per the paper's formula
                max_conf_tensor = torch.max(confidences_tensor)
                max_conf = max_conf_tensor.item()
                
                # Return the maximum confidence as the loss
                loss = max_conf_tensor
            else:
                # This shouldn't happen if found_target is True, but just in case
                max_conf = 0.0
                loss = torch.tensor(0.01, device=self.device, requires_grad=True)
        else:
            # If target not found (already disappeared), use small loss value
            max_conf = 0.0
            loss = torch.tensor(0.01, device=self.device, requires_grad=True)
        
        return loss, max_conf

    def train_patch_at_position(self, image_tensor, bbox, grid_position, epochs, base_lr=0.1):
        """Train patch at a specific grid position"""
        patch_size = int((bbox[2] - bbox[0]) * np.sqrt(self.patch_ratio))
        
        # Initialize random patch for this position
        patch = torch.rand((3, patch_size, patch_size), requires_grad=True, device=self.device)
        
        # Track best patch and its confidence at this position
        best_patch = patch.clone()
        best_conf = 1.0
        stagnation_counter = 0
        
        # Initialize optimizer
        initial_lr = base_lr
        optimizer = torch.optim.Adam([patch], lr=initial_lr)
        
        # Training loop for this position
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Apply patch to image at current grid position
            perturbed = self.apply_patch(image_tensor, patch, bbox, grid_position)
            perturbed_clone = perturbed.clone().unsqueeze(0)
            
            # Forward pass through model
            results = self.model(perturbed_clone)
            
            # Calculate loss using the paper's disappearance loss function
            loss, conf_value = self.calculate_disappearance_loss(results, self.target_class)
            
            # Backpropagate if loss requires grad
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            # Clamp patch values to valid range
            with torch.no_grad():
                patch.clamp_(0, 1)
            
            # Adaptive learning rate based on progress
            if self.use_adaptive_learning:
                # If confidence drops significantly, adjust learning rate
                if epoch > 0 and epoch % 5 == 0:
                    if conf_value < 0.3 and conf_value > 0.05:
                        # Fine-tuning phase with smaller learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = initial_lr * 0.5
                    elif conf_value <= 0.05:
                        # Nearly disappeared, use very small learning rate for fine-tuning
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = initial_lr * 0.1
            
            # Track best patch
            if conf_value < best_conf:
                best_conf = conf_value
                best_patch = patch.clone().detach()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Early stopping if we achieve great results
            if conf_value < 0.02:  # Object effectively disappeared
                print(f"  Early stopping at epoch {epoch}: object disappeared (conf: {conf_value:.4f})")
                break
                
            # Early stopping if progress stagnates
            if stagnation_counter >= 10:
                # Try a different strategy if stuck
                if conf_value > 0.5:
                    # If we're stuck with high confidence, try more randomness
                    with torch.no_grad():
                        noise = torch.randn_like(patch) * 0.2
                        patch.add_(noise).clamp_(0, 1)
                    stagnation_counter = 0
                    print(f"  Adding noise to break stagnation at epoch {epoch}")
                elif stagnation_counter >= 15:
                    # If still stuck after noise, just stop
                    print(f"  Early stopping at epoch {epoch} due to stagnation. Best conf: {best_conf:.4f}")
                    break
            
            # Print progress periodically
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Grid {grid_position} | Epoch {epoch}/{epochs} | Conf: {conf_value:.4f} | Best: {best_conf:.4f}")
        
        return best_patch, best_conf

    def train(self, image_path, epochs_per_location=30, base_lr=0.1, refinement_epochs=50):
        # Load and prepare image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(image, (640, 640))
        image_tensor = torch.tensor(resized_img, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).to(self.device) / 255.0

        # Initial detection to find the target object
        with torch.no_grad():
            results = self.model(image_tensor.unsqueeze(0))
            # Find target class bbox
            found = False
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i].item()) == self.target_class:
                    bbox = results[0].boxes.xyxy[i].clone().cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Target class {self.target_class} not found in image")
            
            print(f"Found target at bbox: {x1}, {y1}, {x2}, {y2}")
        
        # Set up visualization
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        location_results = np.ones((self.grid_size, self.grid_size)) * 1.0  # Initialize with high values
        
        # Grid search tracking
        best_conf = 1.0  # Start with worst possible (highest confidence)
        best_location = None
        best_patch = None
        location_patches = {}  # Store patches for each location
        
        # PHASE 1: Grid search to find the best location
        print("\n=== PHASE 1: Grid Search for Best Location ===")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                grid_position = (row, col)
                print(f"\nTesting grid position {grid_position}")
                
                # Train patch at this position
                patch, conf = self.train_patch_at_position(
                    image_tensor, 
                    (x1, y1, x2, y2), 
                    grid_position, 
                    epochs_per_location, 
                    base_lr
                )
                
                # Store results for this position
                location_results[row, col] = conf
                location_patches[grid_position] = patch
                
                # Check if this is the best so far
                if conf < best_conf:
                    best_conf = conf
                    best_location = grid_position
                    best_patch = patch.clone()
                
                # Update heatmap visualization
                ax3.clear()
                im = ax3.imshow(location_results, cmap='viridis_r')  # viridis_r: dark is low confidence (better)
                plt.colorbar(im, ax=ax3)
                ax3.set_title("Confidence Heatmap (darker=better)")
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        ax3.text(c, r, f"{location_results[r, c]:.2f}", 
                                ha="center", va="center", 
                                color="white" if location_results[r, c] > 0.5 else "black")
                
                # Show current patch
                display_img = self.apply_patch(
                    image_tensor, 
                    patch, 
                    (x1, y1, x2, y2), 
                    grid_position
                ).permute(1, 2, 0).detach().cpu().numpy()
                
                display_img = np.clip(display_img, 0, 1)
                ax1.clear()
                ax1.imshow(display_img)
                ax1.set_title(f"Current: Grid {grid_position}\nConf: {conf:.4f}")
                
                # Show patch itself
                patch_img = patch.detach().cpu().permute(1, 2, 0).numpy()
                patch_img = np.clip(patch_img, 0, 1)
                ax2.clear()
                ax2.imshow(patch_img)
                ax2.set_title(f"Patch for Grid {grid_position}")
                
                plt.tight_layout()
                plt.pause(0.01)
                plt.draw()
        
        # PHASE 2: Refine the best patch with more epochs
        print(f"\n=== PHASE 2: Refining Best Patch at Grid Position {best_location} ===")
        refined_patch, refined_conf = self.train_patch_at_position(
            image_tensor, 
            (x1, y1, x2, y2), 
            best_location, 
            refinement_epochs, 
            base_lr * 0.5  # Lower learning rate for refinement
        )
        
        if refined_conf < best_conf:
            best_conf = refined_conf
            best_patch = refined_patch.clone()
            print(f"Refinement improved confidence from {best_conf:.4f} to {refined_conf:.4f}")
        else:
            print(f"Refinement did not improve confidence. Keeping original patch.")
        
        # PHASE 3: Test patches from neighboring cells at best location
        print("\n=== PHASE 3: Testing Patch Combinations ===")
        
        # Get neighboring grid positions
        best_row, best_col = best_location
        neighbors = []
        for r in range(max(0, best_row-1), min(self.grid_size, best_row+2)):
            for c in range(max(0, best_col-1), min(self.grid_size, best_col+2)):
                if (r, c) != best_location:
                    neighbors.append((r, c))
        
        # Try combining best patch with patches from neighboring cells
        combined_results = []
        for neighbor in neighbors:
            if neighbor in location_patches:
                neighbor_patch = location_patches[neighbor]
                
                # Create a combined patch (simple average for now)
                with torch.no_grad():
                    combined_patch = (best_patch + neighbor_patch) / 2.0
                
                # Test the combined patch
                perturbed = self.apply_patch(image_tensor, combined_patch, (x1, y1, x2, y2), best_location)
                results = self.model(perturbed.unsqueeze(0))
                
                # Calculate confidence using our disappearance loss function
                _, combined_conf = self.calculate_disappearance_loss(results, self.target_class)
                
                combined_results.append((neighbor, combined_patch, combined_conf))
                print(f"Combined with {neighbor}: conf = {combined_conf:.4f}")
                
                # Update best if improved
                if combined_conf < best_conf:
                    best_conf = combined_conf
                    best_patch = combined_patch.clone()
        
        # Final wrap-up: display and save the best result
        print(f"\nBest grid position: {best_location} with confidence: {best_conf:.4f}")
        
        # Display the best result
        if best_patch is not None and best_location is not None:
            final_perturbed = self.apply_patch(image_tensor, best_patch, (x1, y1, x2, y2), best_location)
            
            # Evaluation status
            if best_conf < self.confidence_threshold:
                status = "SUCCESS: Object effectively disappeared"
            else:
                status = "PARTIAL SUCCESS: Object confidence reduced"
            
            print(f"\nFinal result: {status}")
            print(f"Final confidence: {best_conf:.4f}")
            
            # Save final images
            final_patch = (best_patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite("adversarial_patch.png", cv2.cvtColor(final_patch, cv2.COLOR_RGB2BGR))
            
            final_perturbed_img = (final_perturbed.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite("perturbed_image.bmp", cv2.cvtColor(final_perturbed_img, cv2.COLOR_RGB2BGR))
            
            # Save the confidence heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(location_results, cmap='viridis_r')
            plt.colorbar(label="Confidence (lower is better)")
            plt.title("Detection Confidence by Grid Position")
            
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    plt.text(c, r, f"{location_results[r, c]:.2f}", 
                            ha="center", va="center", 
                            color="white" if location_results[r, c] > 0.5 else "black")
            
            plt.savefig("confidence_heatmap.png")
            
            # Show final best patch result
            ax1.clear()
            ax1.imshow(final_perturbed_img / 255)
            ax1.set_title(f"Best Result: Grid {best_location}\nConf: {best_conf:.4f}")
            
            ax2.clear()
            ax2.imshow(final_patch / 255)
            ax2.set_title("Final Optimized Patch")
            
            plt.tight_layout()
            plt.pause(0.1)
            plt.savefig("final_result.png")
        
        print("Disappearance attack optimization complete!")
        return best_patch, best_location, location_results

if __name__ == "__main__":
    generator = AdversarialPatchGenerator(target_class=11, patch_ratio=0.3, grid_size=10)
    best_patch, best_location, location_results = generator.train(
        "images.jpeg", 
        epochs_per_location=30,
        base_lr=0.1,
        refinement_epochs=50
    )