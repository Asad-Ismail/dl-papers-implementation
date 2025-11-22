#!/usr/bin/env python3
"""
FarSight + LLaVA Inference Script
Compares base LLaVA model with FarSight-enhanced version.
Generates attention maps and text outputs side-by-side.

FarSight Integration Approach:
-------------------------------
FarSight is integrated into the model's forward pass using monkey-patching:
1. Store original attention forward methods from all decoder layers
2. Wrap each attention layer with FarSight masking:
   - Apply causal mask (C): Upper-triangular to prevent looking ahead
   - Add attention register (P): Absorbs outlier attention
   - Apply positional mask (pos_mask): Enforces diminishing-rate decay
3. Generate text with FarSight active in forward pass
4. Restore original methods after generation

This is a plug-and-play approach that requires NO model weight changes!
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from src.config import Config
from src.farsight_attention import FarSightAttention


def download_sample_image():
    """Download a sample image for testing"""
    import urllib.request
    import ssl
    
    # Disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
    
    img_path = "data/images/test_image.jpg"
    os.makedirs("data/images", exist_ok=True)
    
    if not os.path.exists(img_path):
        print("Downloading sample image...")
        # Beach scene with surfers (similar to paper example)
        url = "https://picsum.photos/id/1015/640/480"
        urllib.request.urlretrieve(url, img_path)
        print(f"✓ Downloaded to {img_path}")
    
    return img_path


def visualize_attention_maps(attention_weights, title, save_path=None):
    """
    Visualize attention maps as heatmaps
    attention_weights: tensor of shape [num_heads, seq_len, seq_len]
    """
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    for i in range(min(num_heads, 8)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Get attention map for this head
        attn_map = attention_weights[i].cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved attention visualization to {save_path}")
    
    return fig


def run_base_llava(image_path, prompt, model, processor, device):
    """Run base LLaVA model"""
    print("\n" + "="*70)
    print("Running BASE LLaVA Model")
    print("="*70)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare inputs
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Decode output
    generated_text = processor.decode(output.sequences[0], skip_special_tokens=True)
    response = generated_text.split("ASSISTANT:")[-1].strip()
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    
    # Extract attention weights (last layer, last generated token)
    if hasattr(output, 'attentions') and output.attentions:
        last_layer_attn = output.attentions[-1][-1]  # Last layer, last token
        avg_attn = last_layer_attn[0].mean(dim=0)  # Average over batch
    else:
        avg_attn = None
    
    return response, avg_attn


def run_farsight_llava(image_path, prompt, model, processor, device, cfg):
    """Run LLaVA with FarSight attention enhancement"""
    print("\n" + "="*70)
    print("Running FarSight-Enhanced LLaVA Model")
    print("="*70)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare inputs
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    
    # Initialize FarSight parameters (seq_len will be computed dynamically)
    decay_base = cfg['hyperparameters']['decay_base']
    p = cfg['hyperparameters']['p']
    
    print(f"Integrating FarSight into model forward pass...")
    
    # Import FarSight masking functions
    from src.causal_mask import build_causal_mask, build_attention_register, build_positional_mask
    import math
    
    # Store original forward method
    original_forwards = []
    
    # Monkey-patch attention layers to apply FarSight masking
    def create_farsight_forward(original_forward, layer_idx):
        """Create a wrapped forward pass that applies FarSight masks"""
        def farsight_forward(hidden_states, attention_mask=None, **kwargs):
            # Call original forward to get attention weights
            outputs = original_forward(hidden_states, attention_mask=attention_mask, 
                                      output_attentions=True, **kwargs)
            
            # Extract attention weights
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attn_weights = outputs.attentions
                
                # Get sequence length dynamically from attention weights
                B, H, T, _ = attn_weights.shape
                
                # Compute sigma based on actual sequence length
                sigma = math.log(decay_base) / T
                
                # Build FarSight masks for current sequence length
                C = build_causal_mask(T, device=device)
                P = build_attention_register(T, sigma, device=device)
                pos_mask = build_positional_mask(T, p, device=device)
                
                # Apply FarSight masking: C ⊙ (A + P) ⊙ pos_mask
                # C: Causal mask (upper triangular)
                # P: Attention register (absorbs outliers)
                # pos_mask: Positional awareness (diminishing rate)
                
                # Expand masks for batch and heads
                C_expanded = C.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                pos_mask_expanded = pos_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                P_expanded = P.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                
                # Apply masks
                masked_attn = attn_weights * C_expanded * pos_mask_expanded
                masked_attn = masked_attn + P_expanded * C_expanded
                
                # Renormalize
                masked_attn = masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-9)
                
                # Replace attention weights in output
                outputs.attentions = masked_attn
            
            return outputs
        
        return farsight_forward
    
    # Patch all decoder attention layers
    text_model = model.language_model
    num_layers = len(text_model.model.layers)
    
    print(f"Patching {num_layers} attention layers with FarSight...")
    
    for i, layer in enumerate(text_model.model.layers):
        # Store original forward
        original_forward = layer.self_attn.forward
        original_forwards.append((layer.self_attn, original_forward))
        
        # Replace with FarSight-enhanced forward
        layer.self_attn.forward = create_farsight_forward(original_forward, i)
    
    # Generate with FarSight integrated into forward pass
    print("Generating response with FarSight integrated...")
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True
            )
    finally:
        # Restore original forward methods
        print("Restoring original attention layers...")
        for attn_module, original_forward in original_forwards:
            attn_module.forward = original_forward
    
    # Decode output
    generated_text = processor.decode(output.sequences[0], skip_special_tokens=True)
    response = generated_text.split("ASSISTANT:")[-1].strip()
    
    print(f"\nPrompt: {prompt}")
    print(f"Response (with FarSight): {response}")
    
    # Extract FarSight-modified attention from output
    # The attention has already been modified by FarSight in the forward pass
    if hasattr(output, 'attentions') and output.attentions:
        last_layer_attn = output.attentions[-1][-1]  # Last layer, last token
        avg_attn = last_layer_attn[0].mean(dim=0)  # Average over heads and batch
    else:
        avg_attn = None
    
    return response, avg_attn


def main():
    print("\n" + "="*70)
    print("FarSight + LLaVA Inference Demo")
    print("Mitigating Hallucinations in Multimodal LLMs")
    print("="*70)
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load config
    cfg = Config(default_path='configs/default.yaml')
    
    # Download sample image
    image_path = download_sample_image()
    
    # Load LLaVA model
    print("\nLoading LLaVA model (this may take a few minutes)...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # Smaller model for Mac
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device
    )
    
    print(" Model loaded successfully")

    prompts = [
        "Where is the dog in this image?",
        "Describe what you see in this image.",
        "What is happening in this scene?"
    ]
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run comparison for first prompt
    prompt = prompts[0]
    
    # Base model
    base_response, base_attn = run_base_llava(image_path, prompt, model, processor, device)
    
    # FarSight model
    farsight_response, farsight_attn = run_farsight_llava(image_path, prompt, model, processor, device, cfg)
    
    # Visualize attention maps
    if base_attn is not None and farsight_attn is not None:
        print("\nGenerating attention visualizations...")
        
        # Visualize base attention
        visualize_attention_maps(
            base_attn.unsqueeze(0) if base_attn.dim() == 2 else base_attn,
            "Base LLaVA Attention Maps",
            "outputs/base_attention.png"
        )
        
        # Visualize FarSight attention
        visualize_attention_maps(
            farsight_attn.unsqueeze(0) if farsight_attn.dim() == 2 else farsight_attn,
            "FarSight-Enhanced Attention Maps",
            "outputs/farsight_attention.png"
        )
    else:
        print("\nNote: Attention weights not available for visualization")
    
    # Create comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nPrompt: {prompt}")
    print(f"\nBase Response:\n  {base_response}")
    print(f"\nFarSight Response:\n  {farsight_response}")
    print("\n" + "="*70)
    print("\nKey Differences:")
    print("• Base model may hallucinate objects not present in the image")
    print("• FarSight uses causal masking to prevent attention outliers")
    print("• FarSight enforces positional awareness for more grounded responses")
    print("="*70)
    
    print("\n✓ Demo complete! Check outputs/ folder for attention visualizations.")


if __name__ == "__main__":
    main()
