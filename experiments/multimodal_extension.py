#!/usr/bin/env python3
"""
Multi-modal SlotFormer experiments.
Extends causal learning to text, audio, and cross-modal scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os
from transformers import AutoTokenizer, AutoModel

# Add parent directory to path to import train module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer

class MultiModalSlotFormer(nn.Module):
    """SlotFormer extended to multiple modalities."""
    
    def __init__(self, num_slots=128, modalities=['vision', 'text']):
        super().__init__()
        self.num_slots = num_slots
        self.modalities = modalities
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict()
        self.feature_dims = {}
        
        if 'vision' in modalities:
            # Vision encoder (DINOv3)
            import timm
            self.encoders['vision'] = timm.create_model('vit_base_patch16_dinov3', pretrained=True)
            self.feature_dims['vision'] = 768
            
        if 'text' in modalities:
            # Text encoder (BERT-like) - use offline cache
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
                self.encoders['text'] = AutoModel.from_pretrained('distilbert-base-uncased', local_files_only=True)
            except:
                print("⚠️ DistilBERT not found in cache, using dummy encoder")
                # Fallback to simple embedding layer
                self.tokenizer = None
                self.encoders['text'] = nn.Embedding(30000, 768)  # Simple embedding
            self.feature_dims['text'] = 768
            
        # Unified feature dimension
        self.unified_dim = 768
        
        # Modality projection layers
        self.projectors = nn.ModuleDict()
        for modality in modalities:
            if self.feature_dims[modality] != self.unified_dim:
                self.projectors[modality] = nn.Linear(self.feature_dims[modality], self.unified_dim)
            else:
                self.projectors[modality] = nn.Identity()
        
        # Shared SlotFormer components
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, self.unified_dim))
        self.null_slots = nn.Parameter(torch.randn(1, num_slots, self.unified_dim))
        
        # Cross-attention generator (causal)
        generator_layer = nn.TransformerDecoderLayer(
            d_model=self.unified_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.generator = nn.TransformerDecoder(generator_layer, num_layers=6)
        
        # Causal mask
        self.register_buffer('tgt_mask', torch.triu(torch.ones(num_slots, num_slots), diagonal=1).bool())
        
        # Modality-specific reconstructors
        self.reconstructors = nn.ModuleDict()
        for modality in modalities:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.unified_dim,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            )
            self.reconstructors[modality] = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Positional embeddings for different modalities
        self.pos_embeds = nn.ParameterDict()
        self.pos_embeds['vision'] = nn.Parameter(torch.randn(1, 196, self.unified_dim))  # 14x14 patches
        self.pos_embeds['text'] = nn.Parameter(torch.randn(1, 512, self.unified_dim))    # Max 512 tokens
    
    def encode_vision(self, images):
        """Encode vision inputs."""
        features = self.encoders['vision'].forward_features(images)
        patch_features = features[:, self.encoders['vision'].num_prefix_tokens:]  # Remove CLS token
        return self.projectors['vision'](patch_features)
    
    def encode_text(self, texts):
        """Encode text inputs."""
        if self.tokenizer is not None:
            # Use proper BERT tokenizer
            if isinstance(texts[0], str):
                encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt')
                input_ids = encoded['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
            else:
                input_ids, attention_mask = texts
            
            outputs = self.encoders['text'](input_ids=input_ids, attention_mask=attention_mask)
            token_features = outputs.last_hidden_state
            
            # Only use actual tokens (not padding)
            valid_features = []
            for i, mask in enumerate(attention_mask):
                valid_length = mask.sum().item()
                valid_features.append(token_features[i, :valid_length])
            
            # Pad to consistent length
            max_len = max(len(f) for f in valid_features)
            padded_features = torch.zeros(len(valid_features), max_len, token_features.size(-1)).to(token_features.device)
            
            for i, features in enumerate(valid_features):
                padded_features[i, :len(features)] = features
        else:
            # Fallback: simple word-level embedding
            device = next(self.parameters()).device
            # Simple tokenization by spaces
            all_tokens = []
            for text in texts:
                tokens = text.lower().split()[:50]  # Max 50 words
                token_ids = [hash(token) % 30000 for token in tokens]  # Simple hash-based vocab
                all_tokens.append(token_ids)
            
            # Pad sequences
            max_len = max(len(tokens) for tokens in all_tokens)
            padded_ids = torch.zeros(len(all_tokens), max_len, dtype=torch.long).to(device)
            
            for i, tokens in enumerate(all_tokens):
                padded_ids[i, :len(tokens)] = torch.tensor(tokens)
            
            padded_features = self.encoders['text'](padded_ids)
        
        return self.projectors['text'](padded_features)
    
    def forward(self, inputs, modality, num_slots=None):
        """Forward pass for specific modality."""
        if num_slots is None:
            num_slots = self.num_slots
        
        # Encode input
        if modality == 'vision':
            features = self.encode_vision(inputs)
        elif modality == 'text':
            features = self.encode_text(inputs)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        B = features.size(0)
        
        # Generate slots with causal attention
        slots = self.generator(
            tgt=self.slot_queries[:, :num_slots].repeat(B, 1, 1),
            memory=features,
            tgt_mask=self.tgt_mask[:num_slots, :num_slots],
            tgt_is_causal=True,
        )
        
        # Reconstruct in same modality
        pos_embed = self.pos_embeds[modality][:, :features.size(1)]
        reconstructed = self.reconstructors[modality](
            tgt=pos_embed.repeat(B, 1, 1),
            memory=slots,
        )
        
        return slots, reconstructed, features

class MultiModalExperiments:
    """Multi-modal SlotFormer experiments."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.results_dir = Path("./multimodal_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def test_text_causal_learning(self):
        """Test causal learning on text sequences."""
        print("=== Text Causal Learning Experiment ===")
        
        # Create text dataset
        texts = [
            "The cat sat on the mat and looked around carefully.",
            "A beautiful sunset painted the sky in brilliant colors.",
            "Scientists discovered a new species in the deep ocean.",
            "The ancient castle stood majestically on the hilltop.",
            "Children played happily in the sunny garden all day.",
            "The mysterious book contained secrets from another world.",
            "Fresh coffee brewing filled the kitchen with wonderful aromas.",
            "Waves crashed against the rocky shore during the storm.",
            "The old tree provided shade for travelers on hot days.",
            "Music filled the concert hall with beautiful melodies."
        ] * 10  # Repeat for more data
        
        # Create model
        model = MultiModalSlotFormer(num_slots=64, modalities=['text']).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        model.train()
        losses = []
        
        for epoch in range(20):
            epoch_loss = 0
            count = 0
            
            for i in range(0, len(texts), 4):
                batch_texts = texts[i:i+4]
                
                # Random slot count for training
                num_slots = torch.randint(1, model.num_slots + 1, (1,)).item()
                
                slots, reconstructed, original = model(batch_texts, 'text', num_slots)
                
                # Pad slots if needed
                if num_slots < model.num_slots:
                    B = slots.size(0)
                    null_padding = model.null_slots[:, :model.num_slots-num_slots]
                    null_padding = null_padding.expand(B, -1, -1).type_as(slots)
                    padded_slots = torch.cat([slots, null_padding], dim=1)
                else:
                    padded_slots = slots
                
                # Reconstruction loss
                loss = F.mse_loss(reconstructed, original)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
            
            avg_loss = epoch_loss / count
            losses.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Test causal properties
        model.eval()
        test_texts = texts[:10]  # Use first 10 for testing
        slot_counts = [1, 4, 8, 16, 32, 64]
        
        monotonic_scores = []
        
        with torch.no_grad():
            for text in test_texts:
                text_losses = []
                
                for num_slots in slot_counts:
                    slots, reconstructed, original = model([text], 'text', num_slots)
                    
                    # Pad slots
                    if num_slots < model.num_slots:
                        null_padding = model.null_slots[:, :model.num_slots-num_slots]
                        null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                        padded_slots = torch.cat([slots, null_padding], dim=1)
                    else:
                        padded_slots = slots
                    
                    loss = F.mse_loss(reconstructed, original).item()
                    text_losses.append(loss)
                
                # Check monotonic decrease
                decreases = sum(1 for i in range(1, len(text_losses)) 
                              if text_losses[i] < text_losses[i-1])
                monotonic_score = decreases / (len(text_losses) - 1)
                monotonic_scores.append(monotonic_score)
        
        avg_monotonic = np.mean(monotonic_scores)
        
        results = {
            'modality': 'text',
            'training_losses': losses,
            'monotonic_score': avg_monotonic,
            'individual_scores': monotonic_scores
        }
        
        print(f"  Text monotonic score: {avg_monotonic:.3f}")
        
        return results
    
    def test_cross_modal_alignment(self):
        """Test cross-modal slot alignment."""
        print("=== Cross-Modal Alignment Experiment ===")
        
        # Create vision-text pairs
        import torchvision
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Simple captions for Imagenette classes
        captions = {
            0: "A small dog with fluffy fur",
            1: "A colorful bird with bright feathers", 
            2: "A sleek sports car on the road",
            3: "A large fish swimming in water",
            4: "A musical instrument with strings",
            5: "A gas pump at a station",
            6: "A sharp knife for cutting",
            7: "A parachute floating in the sky",
            8: "A tennis racket for sports",
            9: "A truck carrying heavy loads"
        }
        
        try:
            dataset = torchvision.datasets.Imagenette(
                "./datasets", split="val", transform=transform, download=False
            )
            
            # Create image-text pairs
            pairs = []
            for i in range(min(50, len(dataset))):
                image, label = dataset[i]
                caption = captions[label]
                pairs.append((image, caption))
                
        except:
            print("⚠️ Using synthetic data for cross-modal test")
            pairs = [(torch.randn(3, 224, 224), "A synthetic test image") for _ in range(20)]
        
        # Create multi-modal model
        model = MultiModalSlotFormer(num_slots=32, modalities=['vision', 'text']).to(self.device)
        
        # Test slot alignment
        model.eval()
        alignment_scores = []
        
        with torch.no_grad():
            for image, text in pairs[:10]:  # Test first 10 pairs
                image = image.unsqueeze(0).to(self.device)
                
                # Get slots for both modalities
                vision_slots, _, _ = model(image, 'vision', 16)
                text_slots, _, _ = model([text], 'text', 16)
                
                # Compute alignment (cosine similarity between slot sets)
                vision_flat = vision_slots.flatten()
                text_flat = text_slots.flatten()
                
                alignment = F.cosine_similarity(vision_flat, text_flat, dim=0).item()
                alignment_scores.append(alignment)
        
        avg_alignment = np.mean(alignment_scores)
        
        results = {
            'cross_modal_alignment': avg_alignment,
            'individual_alignments': alignment_scores
        }
        
        print(f"  Cross-modal alignment: {avg_alignment:.3f}")
        
        return results
    
    def create_multimodal_analysis(self):
        """Create comprehensive multi-modal analysis."""
        print("=== Multi-Modal SlotFormer Analysis ===")
        
        all_results = {}
        
        # Test text causal learning
        text_results = self.test_text_causal_learning()
        all_results['text'] = text_results
        
        # Test cross-modal alignment
        cross_modal_results = self.test_cross_modal_alignment()
        all_results['cross_modal'] = cross_modal_results
        
        # Save results
        with open(self.results_dir / 'multimodal_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create visualization
        self.plot_multimodal_results(all_results)
        
        return all_results
    
    def plot_multimodal_results(self, results):
        """Plot multi-modal results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Text learning curve
        if 'text' in results:
            losses = results['text']['training_losses']
            axes[0, 0].plot(losses, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Reconstruction Loss')
            axes[0, 0].set_title('Text Causal Learning')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Text monotonic scores
        if 'text' in results:
            scores = results['text']['individual_scores']
            axes[0, 1].hist(scores, bins=10, alpha=0.7, color='blue')
            axes[0, 1].axvline(x=np.mean(scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(scores):.3f}')
            axes[0, 1].axvline(x=0.8, color='green', linestyle='--', label='Target: 0.8')
            axes[0, 1].set_xlabel('Monotonic Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Text Monotonic Score Distribution')
            axes[0, 1].legend()
        
        # Cross-modal alignment
        if 'cross_modal' in results:
            alignments = results['cross_modal']['individual_alignments']
            axes[1, 0].hist(alignments, bins=10, alpha=0.7, color='purple')
            axes[1, 0].axvline(x=np.mean(alignments), color='red', linestyle='--',
                              label=f'Mean: {np.mean(alignments):.3f}')
            axes[1, 0].set_xlabel('Cross-Modal Alignment')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Vision-Text Slot Alignment')
            axes[1, 0].legend()
        
        # Summary comparison
        modalities = []
        monotonic_scores = []
        
        if 'text' in results:
            modalities.append('Text')
            monotonic_scores.append(results['text']['monotonic_score'])
        
        # Add vision results if available (from previous experiments)
        try:
            with open('./cluster_fixed_visualizations/cluster_analysis_results.json', 'r') as f:
                vision_results = json.load(f)
                if 'summary' in vision_results:
                    modalities.append('Vision')
                    monotonic_scores.append(vision_results['summary']['avg_monotonic_ratio'])
        except:
            pass
        
        if modalities:
            bars = axes[1, 1].bar(modalities, monotonic_scores, 
                                 color=['blue', 'green'], alpha=0.7)
            axes[1, 1].set_ylabel('Monotonic Score')
            axes[1, 1].set_title('Multi-Modal Causal Learning')
            axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='Target')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].legend()
            
            # Add value labels
            for bar, score in zip(bars, monotonic_scores):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'multimodal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run multi-modal experiments."""
    experiments = MultiModalExperiments()
    
    print("=== Multi-Modal SlotFormer Experiments ===")
    
    results = experiments.create_multimodal_analysis()
    
    print(f"\n✅ Multi-modal experiments complete!")
    print(f"📁 Results saved to: {experiments.results_dir.absolute()}")

if __name__ == "__main__":
    main()
