# Pre-Layer Normalization Transformer for Toxicity Classification

## Overview
A custom, lightweight Transformer model with pre-layer normalization (PreLN) architecture for binary toxicity classification. Designed to serve as a drop-in backbone replacement in the existing `ToxicityClassifier` wrapper.

## Architecture Design

### Tokenizer

We will use the existing BERT tokenizer (`distilbert-base-uncased`) from HuggingFace. The tokenizer is completely decoupled from model architecture and provides a proven vocabulary of 30k tokens that covers 99% of English text.

### Core Components

1. **Token Embedding Layer**
   - Vocabulary size: 30k (matches BERT tokenizer vocabulary)
   - Embedding dimension: 256
   - Learnable embeddings from scratch

2. **Positional Encoding**
   - Sinusoidal positional encoding (original Transformer approach)
   - Max sequence length: 256 (configured to match model config max_length)
   - Non-trainable, deterministic encoding
   - Generalizes to unseen sequence lengths

3. **Pre-Layer Normalization Transformer Encoder**
   - Number of layers: 6
   - Hidden dimension (FFN): 1024
   - Number of attention heads: 4
   - Head dimension: 64 (embedding_dim / num_heads = 256 / 4)
   - Dropout: 0.1 (regularization)
   - Layer normalization before each sub-layer (pre-LN)

4. **Pooling & Classification Head**
   - Use [CLS] token representation (first token) for classification
   - Two fully-connected layers for classification:
     - Hidden layer: 512 units with ReLU and dropout
     - Output layer: 2 units (logits for binary classification)

### Pre-Layer Normalization Rationale

**Traditional Post-LN (Original Transformer):**
```
x' = LayerNorm(x + Attention(x))      # LN AFTER residual
x'' = LayerNorm(x' + FFN(x'))         # LN AFTER residual
```

**Pre-LN (Our Approach - Modern Standard):**
```
x' = x + Attention(LayerNorm(x))      # LN BEFORE attention
x'' = x' + FFN(LayerNorm(x'))         # LN BEFORE FFN
```

**Benefits of Pre-LN:**
- Better gradient flow and training stability
- Easier to scale to deeper models
- Prevents gradient explosion in early layers
- No need for careful learning rate scheduling
- Standard in modern transformers (GPT-3, LLaMA)

## Integration with Existing Pipeline

### Compatibility Analysis

✅ **Compatible Design:**

The `ToxicityClassifier` expects backbone to:
```python
outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits if hasattr(outputs, 'logits') else outputs
```

Our custom model will implement a **flexible output interface**:

1. **Option A: Return object with `.logits` attribute** (mimics HuggingFace)
   ```python
   class ModelOutput:
       def __init__(self, logits):
           self.logits = logits
   ```
   
2. **Option B: Return raw logits** (fallback)
   - ToxicityClassifier detects raw tensor and uses directly

Both approaches work with existing `ToxicityClassifier` without modification.

### Data Flow
```
input_ids (batch, seq_len) 
    ↓
Token Embedding + Positional Encoding
    ↓
Pre-LN Transformer Blocks (6 layers)
    ↓
Extract [CLS] Token (position 0)
    ↓
Classification Head
    ↓
Logits (batch, 2)
    ↓
ToxicityClassifier → loss/predictions
```

## Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding Dim | 256 | Balance between capacity and efficiency; common in mobile/lightweight models |
| Vocab Size | 30k | Covers most English + special tokens; reduces embedding parameter overhead |
| Num Layers | 6 | Moderate depth for expressiveness without excessive computation |
| Num Heads | 4 | Works well with 256 dim (256/4 = 64 per head); 8 heads would be overkill for small dim |
| FFN Hidden | 1024 | ~4x embedding dim (standard transformer ratio) |
| Max Seq Len | 256 | Matches model config; shorter than BERT's 512 but sufficient for comments |
| Dropout | 0.1 | Conservative dropout; task is simpler than multi-task learning |
| Position Encoding | Sinusoidal | Generalizes to unseen lengths; zero extra parameters; proven in original Transformer |

## Loss Function Choice

**Cross-Entropy Loss:**
- Binary classification (2 classes: toxic/non-toxic)
- Logits output (2 values per sample)
- Numerically stable (handled by PyTorch)

**Weight Handling (for class imbalance):**
- Can pass `class_weights` to CrossEntropyLoss if needed
- Flexible design: weights can be computed from data imbalance ratio
- Option to enable via config in training pipeline

## Implementation Strategy

### PyTorch Built-in Components (No Reinventing!)

We leverage PyTorch's battle-tested modules:

1. **`nn.TransformerEncoderLayer`** - Complete transformer block with attention + FFN
   - Set `norm_first=True` for Pre-LN architecture (default is Post-LN)
   - Handles multi-head attention, residuals, layer norms automatically

2. **`nn.TransformerEncoder`** - Stack of N encoder layers
   - Automatically broadcasts operations across layers
   - Built-in gradient checkpointing support

3. **`nn.Embedding`** - For token embeddings
   - Efficient sparse gradient updates
   - Automatic padding_idx handling

4. **`nn.MultiheadAttention`** - Used internally by TransformerEncoderLayer
   - Optimized attention computation
   - Proper masking support

**Why this matters:** These components are highly optimized, GPU-accelerated, and battle-tested. No need to reimplement attention mechanics or transformer blocks from scratch.

### File Structure
```
src/models/
├── toxicity_classifier.py (existing - no changes)
└── custom_transformer/
    ├── __init__.py
    └── preln_transformer.py (main model using PyTorch built-ins)
```

### Key Implementation Details

1. **Embedding & Position Encoding Module**
   - `nn.Embedding` for token embeddings (vocab_size=30k, embed_dim=256)
   - Sinusoidal positional encoding (registered as buffer, non-trainable)
   - Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
   - Add embeddings and apply dropout

2. **Pre-LN Transformer Block**
   - Use `nn.TransformerEncoderLayer` with `norm_first=True` for Pre-LN
   - Parameters: d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1
   - Stack N=6 layers using `nn.TransformerEncoder`
   - Final LayerNorm after all layers (`norm=nn.LayerNorm(d_model)`) for stability

3. **Custom Transformer Model**
   - Token embeddings + sinusoidal positional encoding (registered as buffer)
   - `nn.TransformerEncoder` (handles all N layers, masking, and final norm)
   - [CLS] token extraction for classification
   - Classification head: `nn.Sequential` with Linear(256→512), ReLU, Dropout, Linear(512→2)
   - Returns raw logits tensor (compatible with ToxicityClassifier fallback)

4. **Forward Pass**
   ```
   Input: input_ids, attention_mask
   1. Token embeddings + sinusoidal positional encoding
   2. Convert mask (1→attend, 0→mask) to src_key_padding_mask format
   3. Pass to nn.TransformerEncoder (applies all layers + final norm)
   4. Extract [CLS] token representation (index 0)
   5. Pass through classification head
   Return: logits tensor (batch, 2)
   ```

## Why This Approach is Optimal

### ✅ Advantages

1. **Plug-and-Play with Existing Code**
   - Works with ToxicityClassifier without modification
   - Compatible with training pipeline (Trainer API)
   - Uses existing data loading and preprocessing

2. **Leverages PyTorch Built-ins**
   - `nn.TransformerEncoder` with `norm_first=True` handles all complexity
   - No manual attention/FFN/residual implementation needed
   - Optimized, GPU-accelerated, battle-tested components
   - Less error-prone than reimplementing from scratch

3. **Lightweight & Efficient**
   - 6M-8M parameters (vs 66M for DistilBERT)
   - Faster inference and training
   - Lower memory footprint
   - Can run on CPU or single GPU

4. **Full Control**
   - No dependency on pre-trained weights
   - Customizable at every layer
   - Suitable for future optimizations (quantization, distillation)

5. **Modern Architecture**
   - Pre-LN proven superior for stability
   - Multi-head attention for multi-perspective learning
   - Proper gradient flow with residual connections

6. **Small Model, Big Ideas**
   - Demonstrates understanding of transformer internals
   - Shows principled design choices (not magic)
   - Easier to debug and modify

## Next Steps

1. Implement `PreLNTransformer` class with all components
2. Test backward compatibility with ToxicityClassifier
3. Run one full training epoch to verify end-to-end pipeline
4. Compare metrics with DistilBERT baseline
5. Document architectural decisions in report.md

## Notes for Future Improvements

- Could add layer-wise learning rate decay
- Position encoding could be made learnable vs. fixed
- FFN dimension could be configurable in config
- Attention heads could be adjusted based on embedding dim
- Could implement rotary position embeddings (RoPE) for better generalization
