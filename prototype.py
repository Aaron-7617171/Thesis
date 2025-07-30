import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

import matplotlib.pyplot as plt  # requires: pip install matplotlib
import seaborn as sns  # requires: pip install seaborn
import numpy as np

# Original forecasting plot
forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()

# ==========================
# ATTENTION WEIGHTS VISUALIZATION
# ==========================

def visualize_attention_weights(model, tokenized_inputs, layer_idx=0, head_idx=0):
    """
    Visualize attention weights for a specific layer and head
    """
    model.eval()
    with torch.no_grad():
        # Get model outputs with attention weights
        outputs = model(**tokenized_inputs, output_attentions=True, return_dict=True)
        
        # Extract attention weights (encoder and decoder if available)
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None:
            encoder_attentions = outputs.encoder_attentions
            attention_weights = encoder_attentions[layer_idx][0, head_idx].cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention_weights, 
                       cmap='Blues', 
                       cbar=True,
                       square=True,
                       xticklabels=False,
                       yticklabels=False)
            plt.title(f'Encoder Attention Weights - Layer {layer_idx}, Head {head_idx}')
            plt.xlabel('Key/Value Positions')
            plt.ylabel('Query Positions')
            plt.tight_layout()
            plt.show()
            
        if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions is not None:
            decoder_attentions = outputs.decoder_attentions
            if len(decoder_attentions) > layer_idx:
                attention_weights = decoder_attentions[layer_idx][0, head_idx].cpu().numpy()
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(attention_weights, 
                           cmap='Reds', 
                           cbar=True,
                           square=True,
                           xticklabels=False,
                           yticklabels=False)
                plt.title(f'Decoder Attention Weights - Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Key/Value Positions')
                plt.ylabel('Query Positions')
                plt.tight_layout()
                plt.show()

def visualize_attention_patterns(model, tokenized_inputs, max_layers=3, max_heads=4):
    """
    Visualize attention patterns across multiple layers and heads
    """
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_inputs, output_attentions=True, return_dict=True)
        
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None:
            encoder_attentions = outputs.encoder_attentions
            num_layers = min(len(encoder_attentions), max_layers)
            
            fig, axes = plt.subplots(num_layers, max_heads, figsize=(16, 4*num_layers))
            if num_layers == 1:
                axes = axes.reshape(1, -1)
            
            for layer in range(num_layers):
                attention = encoder_attentions[layer][0]  # First batch item
                num_heads = min(attention.shape[0], max_heads)
                
                for head in range(num_heads):
                    attention_weights = attention[head].cpu().numpy()
                    
                    sns.heatmap(attention_weights, 
                               cmap='viridis', 
                               cbar=False,
                               square=True,
                               ax=axes[layer, head],
                               xticklabels=False,
                               yticklabels=False)
                    axes[layer, head].set_title(f'L{layer}H{head}', fontsize=10)
                    
                # Hide unused subplots
                for head in range(num_heads, max_heads):
                    axes[layer, head].axis('off')
            
            plt.suptitle('Encoder Attention Patterns Across Layers and Heads', fontsize=14)
            plt.tight_layout()
            plt.show()

def get_attention_statistics(model, tokenized_inputs):
    """
    Get statistics about attention patterns
    """
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_inputs, output_attentions=True, return_dict=True)
        
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None:
            encoder_attentions = outputs.encoder_attentions
            
            print("=== ATTENTION STATISTICS ===")
            print(f"Number of encoder layers: {len(encoder_attentions)}")
            
            for layer_idx, attention in enumerate(encoder_attentions):
                batch_size, num_heads, seq_len, _ = attention.shape
                print(f"Layer {layer_idx}: {num_heads} heads, sequence length: {seq_len}")
                
                # Calculate attention entropy for each head
                attention_np = attention[0].cpu().numpy()  # First batch
                entropies = []
                
                for head in range(num_heads):
                    head_attention = attention_np[head]
                    # Calculate entropy for each query position
                    head_entropy = -np.sum(head_attention * np.log(head_attention + 1e-8), axis=1)
                    entropies.append(np.mean(head_entropy))
                
                print(f"  Average attention entropy per head: {[f'{e:.3f}' for e in entropies]}")

# Example usage with your Chronos model
print("Setting up attention visualization for Chronos model...")

# Prepare a sample input for attention visualization
# For time series models, we need to tokenize the input properly
sample_context = torch.tensor(df["#Passengers"][:24]).unsqueeze(0)  # Use first 24 months

# Get the underlying model from the pipeline
underlying_model = pipeline.model

# Create tokenized inputs for the model
# Note: For Chronos, we need to prepare inputs in the expected format
try:
    # Prepare inputs similar to how Chronos processes them
    with torch.no_grad():
        # This is a simplified approach - you might need to adjust based on Chronos internals
        sample_inputs = {
            'input_ids': sample_context.long(),
            'attention_mask': torch.ones_like(sample_context).long()
        }
        
        print("Visualizing attention weights...")
        
        # Get attention statistics
        get_attention_statistics(underlying_model, sample_inputs)
        
        # Visualize specific layer and head
        print("Generating attention heatmap for layer 0, head 0...")
        visualize_attention_weights(underlying_model, sample_inputs, layer_idx=0, head_idx=0)
        
        # Visualize patterns across multiple layers and heads
        print("Generating attention patterns across layers and heads...")
        visualize_attention_patterns(underlying_model, sample_inputs, max_layers=2, max_heads=4)
        
except Exception as e:
    print(f"Note: Direct attention visualization failed: {e}")
    print("This might be due to Chronos model's specific architecture.")
    print("Alternative approach: Using model hooks to capture attention...")
    
    # Alternative approach using hooks specifically for Chronos T5-based models
    attention_weights = []
    
    def attention_hook(module, input, output):
        # For T5 models, attention outputs are typically in output[1] if present
        if isinstance(output, tuple) and len(output) > 1:
            # Check if this looks like attention weights (should be 4D: batch, heads, seq, seq)
            attention = output[1] if len(output) > 1 else output[0]
            if hasattr(attention, 'shape') and len(attention.shape) == 4:
                attention_weights.append({
                    'module_name': module.__class__.__name__,
                    'attention': attention.detach().cpu().float()  # Convert to float32 immediately
                })
        elif hasattr(output, 'attentions') and output.attentions is not None:
            attention_weights.append({
                'module_name': module.__class__.__name__,
                'attention': output.attentions.detach().cpu().float()  # Convert to float32 immediately
            })
    
    # Register hooks on attention layers - more specific targeting for T5
    hooks = []
    for name, module in underlying_model.named_modules():
        # Target T5 attention modules more specifically
        if ('attention' in name.lower() and ('SelfAttention' in str(type(module)) or 'MultiHeadAttention' in str(type(module)))) or \
           (hasattr(module, 'o') and hasattr(module, 'q') and hasattr(module, 'k') and hasattr(module, 'v')):
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)
            print(f"Registered hook on: {name} ({type(module).__name__})")
    
    try:
        # Run a forward pass to capture attention
        with torch.no_grad():
            _ = pipeline.predict_quantiles(
                context=torch.tensor(df["#Passengers"][:24]),
                prediction_length=6,
                quantile_levels=[0.5],
            )
        
        print(f"Captured {len(attention_weights)} attention outputs via hooks")
        
        # Process and visualize the captured attention weights
        if attention_weights:
            print("Processing captured attention weights...")
            
            # Visualize the first few attention layers
            max_visualize = min(4, len(attention_weights))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i in range(max_visualize):
                attention_data = attention_weights[i]
                attention_matrix = attention_data['attention']
                module_name = attention_data['module_name']
                
                # Get the first batch and first head (already converted to float32)
                if len(attention_matrix.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    attention_to_plot = attention_matrix[0, 0].numpy()  # First batch, first head
                elif len(attention_matrix.shape) == 3:  # [heads, seq_len, seq_len]
                    attention_to_plot = attention_matrix[0].numpy()  # First head
                else:
                    attention_to_plot = attention_matrix.numpy()
                
                # Create heatmap
                sns.heatmap(attention_to_plot, 
                           cmap='Blues', 
                           cbar=True,
                           square=True,
                           ax=axes[i],
                           xticklabels=False,
                           yticklabels=False)
                axes[i].set_title(f'{module_name}\nAttention Layer {i+1}', fontsize=10)
                axes[i].set_xlabel('Key Positions')
                axes[i].set_ylabel('Query Positions')
            
            # Hide unused subplots
            for i in range(max_visualize, 4):
                axes[i].axis('off')
            
            plt.suptitle('Chronos Model Attention Weights (Captured via Hooks)', fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # Create a detailed view of the first attention layer
            if len(attention_weights) > 0:
                first_attention = attention_weights[0]['attention']
                if len(first_attention.shape) == 4:
                    first_attention_matrix = first_attention[0, 0].numpy()
                elif len(first_attention.shape) == 3:
                    first_attention_matrix = first_attention[0].numpy()
                else:
                    first_attention_matrix = first_attention.numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(first_attention_matrix, 
                           cmap='viridis', 
                           cbar=True,
                           square=True,
                           xticklabels=False,
                           yticklabels=False)
                plt.title(f'Detailed View: {attention_weights[0]["module_name"]} Attention Weights')
                plt.xlabel('Key/Value Positions (Time Steps)')
                plt.ylabel('Query Positions (Time Steps)')
                plt.tight_layout()
                plt.show()
                
                # Print attention statistics
                print("\n=== ATTENTION ANALYSIS ===")
                for i, attn_data in enumerate(attention_weights[:3]):  # Show first 3 layers
                    attention_matrix = attn_data['attention']
                    module_name = attn_data['module_name']
                    
                    if len(attention_matrix.shape) == 4:
                        batch_size, num_heads, seq_len, _ = attention_matrix.shape
                        print(f"Layer {i+1} ({module_name}): {num_heads} heads, sequence length: {seq_len}")
                        
                        # Calculate entropy for each head (diversity of attention)
                        attention_np = attention_matrix[0].numpy()  # First batch (already float32)
                        for head in range(min(num_heads, 4)):  # Show first 4 heads
                            head_attention = attention_np[head]
                            entropy = -np.sum(head_attention * np.log(head_attention + 1e-8), axis=1)
                            avg_entropy = np.mean(entropy)
                            print(f"  Head {head}: Average attention entropy = {avg_entropy:.3f}")
                    else:
                        print(f"Layer {i+1} ({module_name}): Shape = {attention_matrix.shape}")
        else:
            print("No attention weights were captured. The model might have a different architecture.")
            
            # Try accessing T5 model directly
            print("\nTrying to access T5 model components directly...")
            try:
                # Access the underlying T5 model
                if hasattr(underlying_model, 'model'):
                    t5_model = underlying_model.model
                    print(f"Found T5 model: {type(t5_model)}")
                    
                    # Try to get attention from a direct forward pass
                    if hasattr(t5_model, 'encoder'):
                        print("T5 encoder found - attempting direct attention extraction...")
                        sample_input_ids = torch.randint(0, 4096, (1, 24))  # Chronos uses 4096 vocab size
                        
                        with torch.no_grad():
                            encoder_outputs = t5_model.encoder(
                                input_ids=sample_input_ids,
                                output_attentions=True,
                                return_dict=True
                            )
                            
                            if hasattr(encoder_outputs, 'attentions') and encoder_outputs.attentions:
                                print(f"Successfully extracted {len(encoder_outputs.attentions)} encoder attention layers!")
                                
                                # Visualize encoder attentions
                                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                                axes = axes.flatten()
                                
                                for i, attention in enumerate(encoder_outputs.attentions[:4]):
                                    attention_matrix = attention[0, 0].cpu().numpy()  # First batch, first head
                                    
                                    sns.heatmap(attention_matrix,
                                               cmap='Reds',
                                               cbar=True,
                                               square=True,
                                               ax=axes[i],
                                               xticklabels=False,
                                               yticklabels=False)
                                    axes[i].set_title(f'T5 Encoder Layer {i+1}')
                                
                                plt.suptitle('T5 Encoder Attention Weights (Direct Access)')
                                plt.tight_layout()
                                plt.show()
                            
            except Exception as e3:
                print(f"Direct T5 access failed: {e3}")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
    except Exception as e2:
        print(f"Hook-based approach also failed: {e2}")
        print("For advanced attention visualization with Chronos, you may need to:")
        print("1. Access the model's internal T5 components directly")
        print("2. Use the transformers library's attention visualization tools")
        print("3. Modify the Chronos pipeline to output attention weights")
