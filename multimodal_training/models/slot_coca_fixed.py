import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train import SlotFormer

class SlotCoCaFixed(nn.Module):
    """
    Fixed Slot-based Contrastive Captioner
    
    Key fix: Uses GPT-2 (causal) for caption generation instead of BERT (bidirectional)
    - BERT for contrastive learning (bidirectional is fine)
    - GPT-2 for caption generation (causal/autoregressive)
    """
    def __init__(
        self,
        num_slots=128,
        num_layers=3,
        encoder_name="vit_base_patch16_dinov3",
        text_encoder_name="distilbert-base-uncased",
        caption_model_name="gpt2",
        projection_dim=256,
        max_caption_length=77,
        temperature=0.07
    ):
        super().__init__()
        
        # Vision encoder (SlotFormer)
        self.vision_model = SlotFormer(
            num_slots=num_slots,
            num_layers=num_layers,
            encoder_name=encoder_name
        )
        
        # Text encoder for contrastive learning (BERT is fine here)
        self.text_encoder = DistilBertModel.from_pretrained(text_encoder_name)
        self.text_tokenizer = DistilBertTokenizer.from_pretrained(text_encoder_name)
        
        # Caption model (GPT-2 for causal generation)
        self.caption_model = GPT2LMHeadModel.from_pretrained(caption_model_name)
        self.caption_tokenizer = GPT2Tokenizer.from_pretrained(caption_model_name)
        self.caption_tokenizer.pad_token = self.caption_tokenizer.eos_token
        
        # Get dimensions
        self.vision_dim = self.vision_model.slot_queries.shape[-1]
        self.text_dim = self.text_encoder.config.hidden_size
        self.caption_dim = self.caption_model.config.n_embd
        
        # Projection heads for contrastive learning
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Visual adapter: project slots to GPT-2 dimension for cross-attention
        self.visual_adapter = nn.Linear(self.vision_dim, self.caption_dim)
        
        # Temperature for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.max_caption_length = max_caption_length
    
    def encode_image(self, images):
        """Encode images to slot representations"""
        with torch.no_grad():
            gt_features = self.vision_model.encoder.forward_features(images)
            gt_patches = gt_features[:, self.vision_model.encoder.num_prefix_tokens:]
        
        slots = self.vision_model.generator(
            tgt=self.vision_model.slot_queries.repeat(images.shape[0], 1, 1),
            memory=gt_patches,
            tgt_mask=self.vision_model.tgt_mask.to(images.device)
        )
        return slots, gt_patches
    
    def encode_text(self, text_tokens):
        """Encode text for contrastive learning (uses BERT)"""
        outputs = self.text_encoder(**text_tokens)
        return outputs.last_hidden_state
    
    def forward(self, images, text_tokens=None, caption_tokens=None, mode='all'):
        """
        Forward pass with multiple objectives
        
        Args:
            images: Input images [B, 3, H, W]
            text_tokens: Tokenized text for contrastive learning (BERT tokens)
            caption_tokens: Tokenized captions for generation (GPT-2 tokens)
            mode: 'contrastive', 'caption', 'reconstruct', or 'all'
        """
        outputs = {}
        
        # Encode images
        slots, gt_patches = self.encode_image(images)
        
        # 1. Visual reconstruction loss (SlotFormer objective)
        if mode in ['reconstruct', 'all']:
            reconstructed = self.vision_model.decoder(slots)
            outputs['reconstruction_loss'] = F.mse_loss(reconstructed, gt_patches)
        
        # 2. Contrastive loss (CLIP-style, uses BERT)
        if mode in ['contrastive', 'all'] and text_tokens is not None:
            # Pool slots and text for contrastive learning
            vision_embeds = self.vision_projection(slots.mean(dim=1))
            text_embeds_full = self.encode_text(text_tokens)
            text_embeds = self.text_projection(text_embeds_full[:, 0])  # CLS token
            
            # Normalize
            vision_embeds = F.normalize(vision_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # Compute similarity
            logits_per_image = vision_embeds @ text_embeds.t() / self.temperature
            logits_per_text = logits_per_image.t()
            
            # Contrastive loss (symmetric)
            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=images.device)
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            outputs['contrastive_loss'] = (loss_i2t + loss_t2i) / 2
        
        # 3. Caption generation loss (uses GPT-2, which is causal!)
        if mode in ['caption', 'all'] and caption_tokens is not None:
            # Project visual slots to GPT-2 dimension
            visual_prefix = self.visual_adapter(slots)  # [B, num_slots, caption_dim]
            
            # Get caption embeddings from GPT-2
            caption_input_ids = caption_tokens['input_ids']
            caption_embeds = self.caption_model.transformer.wte(caption_input_ids)
            
            # Concatenate visual prefix + caption embeddings
            inputs_embeds = torch.cat([visual_prefix, caption_embeds], dim=1)
            
            # Create attention mask (attend to visual + caption)
            visual_attention_mask = torch.ones(
                (visual_prefix.shape[0], visual_prefix.shape[1]),
                device=images.device, dtype=torch.long
            )
            combined_attention_mask = torch.cat([
                visual_attention_mask,
                caption_tokens.get('attention_mask', torch.ones_like(caption_input_ids))
            ], dim=1)
            
            # Forward through GPT-2 (causal by default!)
            gpt_outputs = self.caption_model(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                labels=None  # We'll compute loss manually
            )
            
            logits = gpt_outputs.logits
            
            # Extract logits corresponding to caption tokens (skip visual prefix)
            caption_logits = logits[:, visual_prefix.shape[1]:, :]
            
            # Shift for next-token prediction
            shift_logits = caption_logits[:, :-1].contiguous()
            shift_labels = caption_input_ids[:, 1:].contiguous()
            
            outputs['caption_loss'] = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=self.caption_tokenizer.pad_token_id
            )
            outputs['caption_logits'] = caption_logits
        
        return outputs
    
    @torch.no_grad()
    def generate_caption(self, images, max_length=50):
        """Generate captions using GPT-2 (properly causal!)"""
        self.eval()
        batch_size = images.shape[0]
        device = images.device
        
        # Encode image to slots
        slots, _ = self.encode_image(images)
        
        # Project slots to GPT-2 dimension
        visual_prefix = self.visual_adapter(slots)  # [B, num_slots, caption_dim]
        
        # Start with BOS token
        input_ids = torch.full(
            (batch_size, 1),
            self.caption_tokenizer.bos_token_id or self.caption_tokenizer.eos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate autoregressively
        for _ in range(max_length):
            # Get embeddings for current sequence
            caption_embeds = self.caption_model.transformer.wte(input_ids)
            
            # Concatenate visual prefix + caption
            inputs_embeds = torch.cat([visual_prefix, caption_embeds], dim=1)
            
            # Forward through GPT-2
            outputs = self.caption_model(inputs_embeds=inputs_embeds)
            
            # Get logits for next token (last position)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == self.caption_tokenizer.eos_token_id).all():
                break
        
        return input_ids
