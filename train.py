import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import timm
import lightning as L
from tqdm.auto import tqdm

# --- Hyperparameters ---
torch.set_float32_matmul_precision("high")
# Model
NUM_SLOTS = 128  # K: Number of slots to generate
TRANSFORMER_LAYERS = 3  # Number of processing layers in generator/reconstructor
MODELS = {
    "dino": "vit_base_patch16_224.dino",
    "dinov2": "vit_base_patch14_dinov2",
    "dinov3": "vit_base_patch16_dinov3",
}
ENCODER_NAME = MODELS["dinov3"]
# Training
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
PRECISION = "bf16-mixed"
LOG_INTERVAL = 20
COMPILE = False


class PixelDecoder(nn.Module):
    """Decodes patch features back into an image."""

    def __init__(self, input_dim=768, patch_size=16, img_size=224):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_side = img_size // patch_size

        # Project features to a higher-dimensional space for convolution
        self.proj = nn.Linear(input_dim, 256 * 4 * 4)

        # Convolutional upsampling layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x shape: (B, NumPatches, Dim)
        B, N, D = x.shape
        x = self.proj(x)
        # Reshape for convolution: (B, C, H, W)
        x = x.reshape(B, 256, self.num_patches_side // 4, self.num_patches_side // 4)
        x = self.decoder(x)
        return x


class SlotFormer(nn.Module):
    """SlotFormer model."""

    def __init__(self, num_slots: int, num_layers: int, encoder_name: str):
        super().__init__()
        self.num_slots = num_slots
        self.encoder = timm.create_model(encoder_name, pretrained=True)
        self.embed_dim = self.encoder.embed_dim

        # Freeze the vision transformer encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Slot queries and null slots (learnable)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, self.embed_dim))
        self.null_slots = nn.Parameter(torch.randn(1, num_slots, self.embed_dim))

        # Transformer decoder layer for both generator and reconstructor
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=12,  # Standard for ViT-Base
            dim_feedforward=3072,  # Standard for ViT-Base
            dropout=0.1,
            batch_first=True,
        )

        # Generator (causal cross-attention from slots to image patches)
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Reconstructor (cross-attention from slots to patch queries)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Pixel Decoder to go from features to image
        self.pixel_decoder = PixelDecoder(input_dim=self.embed_dim)

        # Positional embedding for the decoder (learnable)
        self.num_patches = self.encoder.patch_embed.num_patches
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, self.embed_dim)
        )

        # Cache a causal target mask once for the generator (compile-friendly)
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(num_slots, device=None),
        )

    def forward(self, images: Tensor, num_slots_to_use: int = None, return_slots: bool = False):
        B, C, H, W = images.shape
        if num_slots_to_use is None:
            num_slots_to_use = self.num_slots

        # 1. Encode image to patch tokens
        patch_tokens = self.encoder.forward_features(images)
        # Remove CLS token if it exists
        if hasattr(self.encoder, 'num_prefix_tokens') and self.encoder.num_prefix_tokens > 0:
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens:]

        # 2. Generate slots using causal cross-attention
        slots = self.generator(
            tgt=self.slot_queries[:, :num_slots_to_use].repeat(B, 1, 1),
            memory=patch_tokens,
            tgt_mask=self.tgt_mask[:num_slots_to_use, :num_slots_to_use],
        )

        # 3. Pad unused slots with null embeddings for reconstruction
        if num_slots_to_use < self.num_slots:
            null_padding = self.null_slots[:, num_slots_to_use:].repeat(B, 1, 1)
            padded_slots = torch.cat([slots, null_padding], dim=1)
        else:
            padded_slots = slots

        # 4. Reconstruct patch features from slots
        reconstructed_patches = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=padded_slots,
        )

        # 5. Decode patch features back to a pixel image
        reconstructed_image = self.pixel_decoder(reconstructed_patches)
        
        # During training, return the reconstructed image and the original for loss calculation
        if self.training:
            return F.mse_loss(reconstructed_image, images)

        # During inference/evaluation
        if return_slots:
            return reconstructed_image, images, slots
        
        return reconstructed_image, images
