# Toxicity Classification Report

**GitHub Repository:** https://github.com/siddique-d1/jigsaw

---

## 1. Architecture Choice: BERT-like Encoder with Pre-LN

**Pre-LN Transformer Architecture:**

```
Input Sequence
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────────┐
│  Transformer Block (×6 layers)          │
│  ┌───────────────────────────────────┐  │
│  │ LayerNorm                         │  │
│  │    ↓                              │  │
│  │ MultiheadAttention (4 heads)      │  │
│  │    ↓                              │  │
│  │ Residual Add with Input           │  │
│  │    ↓                              │  │
│  │ LayerNorm                         │  │
│  │    ↓                              │  │
│  │ FeedForward (d_model→d_ff→d_model)│  │
│  │    ↓                              │  │
│  │ Residual Add with Previous        │  │
│  └───────────────────────────────────┘  │
│  (Repeat 6 times)                       │
└─────────────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
[CLS] Token Hidden State
    ↓
Classification Head (Linear + Sigmoid)
    ↓
Output: Toxic/Non-toxic (0-1)
```

**Key difference from Post-LN:**
- Pre-LN: LayerNorm → Attention → Residual
- Post-LN: Attention → LayerNorm → Residual

**Why [CLS] token for classification?**
- Standard BERT approach: [CLS] token at sequence start aggregates full comment semantics.


---

## 2. Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---|
| vocab_size | 30,522 | distilbert-base-uncased tokenizer (good for English text) |
| d_model | 256 | Simple and small model |
| num_heads | 4 | Standard for 256-dim: 256/4 = 64 per head |
| num_layers | 6 | Simple and small model|
| d_ff | 1024 | 4x d_model (standard Transformer ratio) |
| max_length | 256 | Covers 98% of Civil Comments (typical comment ~50 tokens) |
| dropout | 0.1 | Standard regularization for Transformers |

**Training Config:**
- Learning rate: 2e-5 
- Warmup steps: 500 
- Batch size: 128 
- Num epochs: 1 
- Optimizer: AdamW (standard in transformer training)

---

## 3. Loss Function Design

Implemented three loss functions:

**Option 1: Binary Cross-Entropy (BCE) - Default**
- Standard balanced loss
- Baseline for comparison

**Option 2: Weighted BCE**
- Weights positive class: `pos_weight = 11.50` (class imbalance ratio)
- Equally weights both classes in gradient computation
- Simple and interpretable

**Option 3: Focal Loss**
- Down-weights easy negatives, focuses on hard examples
- Most robust if class distribution changes
- Less tuning required than weighted BCE

---

## 4. Class Imbalance Handling

**Dataset composition:**
- Toxic: 141K (8%)
- Non-toxic: 1.66M (92%)
- Imbalance ratio: **11.5x**

**Handling strategies:**

1. **Loss Function Selection** 
   - Implemented three loss options: BCE (baseline), Weighted BCE (pos_weight=11.50), and Focal Loss
   - Experiments needed to determine which performs best for this dataset

2. **Data-Level Strategies**
   - Current: Train/Val/Test splits (70/15/15) based on percentages
   - Next step: Implement stratified splits to preserve class distribution (8% toxic, 92% non-toxic across all splits)
   - Stratified approach ensures validation metrics are truly representative of test performance

3. **Evaluation with Appropriate Metrics**
   - Accuracy alone is misleading (92% if predicting all non-toxic)
   - Use balanced metrics: **F1, Balanced Accuracy, AUROC, Precision, Recall**
   - Balanced Accuracy = (recall_toxic + recall_nontoxic) / 2

4. **Potential Future Enhancements**
   - **Oversampling**: Duplicate minority class samples for better representation
   - **Label Refinement**: Re-label boundary cases (0.4-0.6) using LLM for cleaner training data
   - **Label Consistency**: Aggregate rows with duplicate comments but conflicting labels by averaging scores
---

## 5. Post-Mortem: 10k Requests/Second Deployment

**Challenge:** Process 10,000 requests/second with low latency. 

**Key Strategy: Dynamic Batching with Token-Length Bucketing**

Since we already have a small model, the focus shifts to maximizing batch efficiency:

1. **Request Queue with Accumulation**
   - Accumulate incoming requests for 5-10ms into a queue
   - Tradeoff: small latency increase (5-10ms) for 10-100x throughput gain
   - Enables fuller GPU utilization

2. **Token-Count Bucketing**
   - Group requests by input length (e.g., bins: <50 tokens, 50-100, 100-150, etc.)
   - Process each bucket separately with dynamic max_length
   - Benefit: Reduces padding waste, more efficient tensor computation
   - Example: 100-token request doesn't pad to 256; instead uses 128-length batch

3. **Inference Server**
   - Use specialized inference server (e.g., **Triton Inference Server**) for automatic batching
   - Triton handles request scheduling, dynamic batching, and GPU memory optimization automatically
   - Alternative: Custom queue service using Redis/RabbitMQ with batching logic
   - Note: I have no experience building something like this.
