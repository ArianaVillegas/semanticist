#!/usr/bin/env python3
"""
Intervention and manipulation experiments for a SlotFormer model trained on CLEVR.

This script demonstrates the object-centric and compositional nature of the learned representations
by performing experiments like object removal, swapping, and attribute manipulation.
"""

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import lightning as L
from torch.utils.data import DataLoader
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import train module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_clevr import SlotFormer, CLEVR_PATH

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints_clevr/best_model_clevr.ckpt"
NUM_SLOTS = 16 # Must match the trained model
NUM_SAMPLES = 5 # Number of images to run interventions on
OUTPUT_DIR = Path("./clevr_interventions_results")

class CLEVRInterventions:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.model = self.load_model(checkpoint_path)
        self.transform = self.get_transforms()
        self.dataset = self.load_dataset()
        OUTPUT_DIR.mkdir(exist_ok=True)

    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the CLEVR model first.")
        
        model = SlotFormer(NUM_SLOTS, 6, "vit_base_patch16_dinov3").to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Adjust for Fabric-saved checkpoint
        model_state = checkpoint.get('model', checkpoint)
        model.load_state_dict(model_state)
        model.eval()
        print("✅ Model loaded successfully.")
        return model

    def get_transforms(self):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def load_dataset(self):
        # Use the validation set for interventions
        clevr_val_path = os.path.join(CLEVR_PATH, 'images', 'val')
        if not os.path.isdir(clevr_val_path):
             # Fallback to train if val doesn't exist in this structure
            clevr_val_path = os.path.join(CLEVR_PATH, 'images', 'train')
        return torchvision.datasets.ImageFolder(root=clevr_val_path, transform=self.transform)

    def get_slots_for_image(self, image_tensor):
        """Extracts the learned slots for a single image."""
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, slots = self.model(image_tensor, return_slots=True)
        return slots.squeeze(0) # Shape: [NUM_SLOTS, D]

    def reconstruct_from_slots(self, slots):
        """Reconstructs an image from a given set of slots using the model's decoder."""
        slots = slots.unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Reconstruct patch features from the modified slots
            reconstructed_patches = self.model.reconstructor(
                tgt=self.model.decoder_pos_embed.repeat(slots.size(0), 1, 1),
                memory=slots,
            )
            # Decode the patch features into a pixel image
            reconstructed_image = self.model.pixel_decoder(reconstructed_patches)
        return reconstructed_image.squeeze(0)

    def run_object_removal(self, sample_idx):
        """Performs the object removal experiment."""
        print(f"\n--- Running Object Removal for Sample {sample_idx} ---")
        image, _ = self.dataset[sample_idx]
        original_slots = self.get_slots_for_image(image)

        # Save original image and reconstruction
        save_image(image * 0.5 + 0.5, OUTPUT_DIR / f"{sample_idx}_original.png")
        full_reconstruction = self.reconstruct_from_slots(original_slots)
        save_image(full_reconstruction * 0.5 + 0.5, OUTPUT_DIR / f"{sample_idx}_reconstruction_full.png")

        # Reconstruct from each individual slot to see what it learned
        for i in range(NUM_SLOTS):
            isolated_slots = torch.zeros_like(original_slots)
            isolated_slots[i] = original_slots[i]
            recon_single = self.reconstruct_from_slots(isolated_slots)
            save_image(recon_single * 0.5 + 0.5, OUTPUT_DIR / f"{sample_idx}_reconstruction_slot_{i}.png")

        # Perform removal for each slot
        for i in range(NUM_SLOTS):
            modified_slots = original_slots.clone()
            modified_slots[i] = 0 # Zero out the slot
            recon_removed = self.reconstruct_from_slots(modified_slots)
            save_image(recon_removed * 0.5 + 0.5, OUTPUT_DIR / f"{sample_idx}_reconstruction_removed_slot_{i}.png")
            print(f"Saved reconstruction with slot {i} removed.")

    def run_object_swapping(self, sample_idx1, sample_idx2):
        """Performs the object swapping experiment."""
        print(f"\n--- Running Object Swapping for Samples {sample_idx1} and {sample_idx2} ---")
        image1, _ = self.dataset[sample_idx1]
        image2, _ = self.dataset[sample_idx2]

        slots1 = self.get_slots_for_image(image1)
        slots2 = self.get_slots_for_image(image2)

        save_image(image1 * 0.5 + 0.5, OUTPUT_DIR / f"swap_{sample_idx1}_original.png")
        save_image(image2 * 0.5 + 0.5, OUTPUT_DIR / f"swap_{sample_idx2}_original.png")

        # Swap the most prominent slot (assuming slot 0 captures the main object)
        for i in range(NUM_SLOTS // 2): # Swap a few slots
            new_slots1 = slots1.clone()
            new_slots1[i] = slots2[i] # Replace slot i from image1 with slot i from image2

            new_slots2 = slots2.clone()
            new_slots2[i] = slots1[i] # And vice-versa

            recon1 = self.reconstruct_from_slots(new_slots1)
            recon2 = self.reconstruct_from_slots(new_slots2)

            save_image(recon1 * 0.5 + 0.5, OUTPUT_DIR / f"swap_{sample_idx1}_with_slot_{i}_from_{sample_idx2}.png")
            save_image(recon2 * 0.5 + 0.5, OUTPUT_DIR / f"swap_{sample_idx2}_with_slot_{i}_from_{sample_idx1}.png")
            print(f"Saved swapped reconstructions for slot {i}.")

def main():
    intervenor = CLEVRInterventions(CHECKPOINT_PATH)

    for i in range(NUM_SAMPLES):
        intervenor.run_object_removal(i)
    
    # Run swapping for a few pairs
    intervenor.run_object_swapping(0, 1)
    intervenor.run_object_swapping(2, 3)

if __name__ == "__main__":
    print("Starting CLEVR intervention experiments...")
    main()
