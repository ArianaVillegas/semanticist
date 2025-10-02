import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train import SlotFormer

class SlotCoCa(nn.Module):
    """
    Slot-based Contrastive Captioner (Slot-CoCa)
    
    Multi-task vision-language model that combines:
    1. Slot-based visual representation (SlotFormer)
    2. Contrastive learning (CLIP-style)
    3. Autoregressive captioning
    """
    def __init__(
        self,
        num_slots=128,
        num_layers=3,
        encoder_name="vit_base_patch16_dinov3",
        text_encoder_name="distilbert-base-uncased",
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
        
        # Text encoder
        self.text_encoder = DistilBertModel.from_pretrained(text_encoder_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_encoder_name)
        
        # Get dimensions
        self.vision_dim = self.vision_model.slot_queries.shape[-1]
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Projection heads for contrastive learning
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Caption decoder (multimodal transformer)
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.text_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Caption generation head
        self.caption_head = nn.Linear(self.text_dim, self.tokenizer.vocab_size)
        
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
        """Encode text to embeddings"""
        outputs = self.text_encoder(**text_tokens)
        return outputs.last_hidden_state  # [B, seq_len, text_dim]
    
    def forward(self, images, text_tokens=None, caption_tokens=None, mode='all'):
        """
        Forward pass with multiple objectives
        
        Args:
            images: Input images [B, 3, H, W]
            text_tokens: Tokenized text for contrastive learning
            caption_tokens: Tokenized captions for generation
            mode: 'contrastive', 'caption', 'reconstruct', or 'all'
        """
        outputs = {}
        
        # Encode images
        slots, gt_patches = self.encode_image(images)
        
        # 1. Visual reconstruction loss (SlotFormer objective)
        if mode in ['reconstruct', 'all']:
            recon_patches = self.vision_model.reconstructor(
                tgt=self.vision_model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                memory=slots
            )
            outputs['reconstruction_loss'] = F.mse_loss(recon_patches, gt_patches)
        
        # 2. Contrastive learning (CLIP objective)
        if mode in ['contrastive', 'all'] and text_tokens is not None:
            # Pool slots for global representation
            vision_features = slots.mean(dim=1)  # [B, vision_dim]
            vision_embeds = self.vision_projection(vision_features)
            
            # Encode text
            text_features = self.encode_text(text_tokens)
            text_embeds = self.text_projection(text_features[:, 0])  # Use [CLS] token
            
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
            
            outputs['vision_embeds'] = vision_embeds
            outputs['text_embeds'] = text_embeds
        
        # 3. Caption generation (autoregressive objective)
        if mode in ['caption', 'all'] and caption_tokens is not None:
            # Use slots as memory for cross-attention
            caption_embeds = self.encode_text(caption_tokens)
            
            # Create causal mask (decoder can't see future tokens)
            seq_len = caption_embeds.size(1)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=images.device
            )
            
            # Autoregressive decoding with cross-attention to slots
            decoded = self.caption_decoder(
                tgt=caption_embeds,
                memory=slots,
                tgt_mask=causal_mask  # Prevent looking ahead!
            )
            
            # Predict next tokens
            logits = self.caption_head(decoded)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = caption_tokens['input_ids'][:, 1:].contiguous()
            
            outputs['caption_loss'] = F.cross_entropy(
                shift_logits.view(-1, self.tokenizer.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            outputs['caption_logits'] = logits
        
        return outputs
    
    @torch.no_grad()
    def generate_caption(self, images, max_length=50, num_beams=3):
        """Generate captions for images using greedy decoding"""
        self.eval()
        batch_size = images.shape[0]
        device = images.device
        
        # Encode image to slots
        slots, _ = self.encode_image(images)
        
        # Start with [CLS] token
        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Greedy autoregressive decoding
        for step in range(max_length):
            # Encode current sequence
            outputs = self.text_encoder(input_ids=input_ids)
            caption_embeds = outputs.last_hidden_state  # [B, seq_len, dim]
            
            # Create causal mask to prevent looking ahead
            seq_len = caption_embeds.size(1)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            
            # Decode with cross-attention to slots (memory)
            # Use causal mask so decoder can't look at future positions
            decoded = self.caption_decoder(
                tgt=caption_embeds,
                memory=slots,
                tgt_mask=causal_mask
            )
            
            # Get logits for LAST position only (next token prediction)
            logits = self.caption_head(decoded[:, -1, :])  # [B, vocab_size]
            
            # Prevent early [SEP] - force model to generate at least 3 words
            min_length = 3
            if step < min_length:
                logits[:, self.tokenizer.sep_token_id] = -float('inf')  # Mask [SEP]
            
            # Greedy: take argmax
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences generated [SEP]
            if (next_token == self.tokenizer.sep_token_id).all():
                break
        
        return input_ids
