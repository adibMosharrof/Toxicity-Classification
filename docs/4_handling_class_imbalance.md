# Handling Severe Class Imbalance in Toxicity Classification

## Problem Statement

Current model performance indicates severe class imbalance issues:
- **Accuracy**: 95.93% (misleadingly high - predicting majority class)
- **F1 Score**: 0.0000 (no positive class predictions)
- **Precision**: 0.0000
- **Recall**: 0.0000
- **AUROC**: 0.4370 (worse than random)
- **Balanced Accuracy**: 0.5000

The model is predicting almost entirely the majority class (non-toxic). We need strategies to force the model to learn the minority class (toxic comments).

---

## Strategy 1: Class-Weighted Loss Function

### Description
Assign higher weights to the minority class in the loss function. PyTorch's `CrossEntropyLoss` supports class weights that penalize misclassifying the minority class more heavily.

### Implementation Approach
- Calculate class weights inversely proportional to class frequencies
- Pass `weight` parameter to `CrossEntropyLoss`
- Formula: `weight[i] = total_samples / (num_classes * class_count[i])`

### Pros
- ✅ Simple to implement (1-line change in loss function)
- ✅ No data manipulation required
- ✅ Works well with imbalanced datasets
- ✅ Integrates seamlessly with existing training loop
- ✅ Computationally efficient (no extra overhead)
- ✅ Supported natively by HuggingFace Trainer

### Cons
- ❌ Requires hyperparameter tuning for optimal weights
- ❌ Can lead to overfitting on minority class if weights too high
- ❌ May increase false positives
- ❌ Doesn't directly address data scarcity of minority class

### When to Use
- First line of defense for class imbalance
- When you want minimal code changes
- When you have sufficient minority class examples (even if imbalanced)

### Recommended for This Project
**Priority: HIGH** - Start here as baseline approach.

---

## Strategy 2: Focal Loss

### Description
Focal Loss (from RetinaNet paper) down-weights easy examples and focuses training on hard examples. It adds a modulating factor `(1 - p_t)^γ` to cross-entropy loss.

Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

### Implementation Approach
- Replace `CrossEntropyLoss` with custom `FocalLoss` implementation
- Tune focusing parameter `γ` (typically 2.0)
- Tune balancing parameter `α` (class weights)

### Pros
- ✅ Handles extreme class imbalance better than weighted loss
- ✅ Reduces contribution from easy examples
- ✅ Forces model to focus on hard-to-classify examples
- ✅ Works well in practice for dense object detection (proven)
- ✅ Can combine with class weights for further control

### Cons
- ❌ Requires custom loss implementation (not in PyTorch by default)
- ❌ More hyperparameters to tune (γ, α)
- ❌ May not integrate cleanly with HuggingFace Trainer
- ❌ Training can be unstable with wrong parameters
- ❌ Computationally slightly more expensive

### When to Use
- When class weights alone don't work
- When you have many "easy" negative examples
- When willing to implement custom training loop modifications

### Recommended for This Project
**Priority: MEDIUM** - Try if class weights insufficient.

---

## Strategy 3: Data Resampling (Oversampling Minority Class)

### Description
Increase the number of minority class samples by duplicating or synthesizing them. Random oversampling repeats examples; SMOTE creates synthetic examples.

### Implementation Approaches
**A. Random Oversampling**
- Duplicate minority class examples to balance classes
- Can use `RandomSampler` with replacement weights

**B. SMOTE (Synthetic Minority Over-sampling Technique)**
- Generate synthetic examples by interpolating between similar minority examples
- Works on feature space (harder for text)

### Pros
- ✅ Directly addresses data scarcity
- ✅ Simple to implement (especially random oversampling)
- ✅ No loss function changes needed
- ✅ Can be combined with other strategies
- ✅ Works well for tabular data

### Cons
- ❌ Random oversampling leads to overfitting (memorization of repeated examples)
- ❌ SMOTE difficult for text data (interpolating embeddings is tricky)
- ❌ Increases training time (more samples per epoch)
- ❌ Doesn't add new information (just repeats existing data)
- ❌ May need to adjust batch size to see balanced batches

### When to Use
- When you have very few minority class examples
- When combined with regularization to prevent overfitting
- When training time is not a concern

### Recommended for This Project
**Priority: LOW** - Text data makes SMOTE impractical; random oversampling risks overfitting.

---

## Strategy 4: Data Resampling (Undersampling Majority Class)

### Description
Reduce the number of majority class samples to balance classes. Randomly discard majority class examples until balanced.

### Implementation Approach
- Sample a subset of majority class (e.g., keep only 2x minority class size)
- Use stratified sampling to ensure diversity
- Can implement via custom `Sampler` in DataLoader

### Pros
- ✅ Fast training (fewer samples)
- ✅ Simple to implement
- ✅ Reduces computational cost
- ✅ Forces model to see minority class more frequently
- ✅ No overfitting risk on minority class

### Cons
- ❌ **Throws away valuable data** (majority class examples)
- ❌ May lose important negative examples (diverse non-toxic text)
- ❌ Can hurt generalization
- ❌ Wasteful when you have large datasets
- ❌ Model sees less total diversity

### When to Use
- When majority class is extremely large and redundant
- When computational resources are limited
- When combined with ensemble methods (bootstrap aggregating)

### Recommended for This Project
**Priority: VERY LOW** - We have limited data already; throwing away examples is wasteful.

---

## Strategy 5: Threshold Tuning (Post-Training)

### Description
Instead of using default 0.5 threshold for binary classification, tune the decision threshold to optimize for desired metric (F1, recall, precision).

### Implementation Approach
- Train model normally
- On validation set, sweep threshold values (0.1 to 0.9)
- Select threshold that maximizes target metric
- Use precision-recall curve or ROC curve analysis

### Pros
- ✅ **No retraining required** (fastest approach)
- ✅ Can optimize for specific business metric
- ✅ Works with any model
- ✅ Easy to explain to stakeholders
- ✅ Can adjust threshold per use case
- ✅ Reversible (can always change back)

### Cons
- ❌ Doesn't improve model's learned representations
- ❌ May not be enough if model outputs are poorly calibrated
- ❌ Requires validation set to tune properly
- ❌ Band-aid solution (doesn't fix root cause)

### When to Use
- **Always** - as a final step after any training approach
- When model probabilities are well-calibrated
- When you need different operating points (high recall vs high precision)

### Recommended for This Project
**Priority: HIGH** - Try this first with existing model before retraining. Quick diagnostic.

---

## Strategy 6: Hybrid Sampling (Over + Under)

### Description
Combine oversampling minority class with undersampling majority class to find middle ground.

### Implementation Approach
- Oversample minority class by 2-3x
- Undersample majority class by 50%
- Achieve ratio like 1:2 or 1:3 instead of 1:20+

### Pros
- ✅ Balances benefits of both approaches
- ✅ Less data waste than pure undersampling
- ✅ Less overfitting than pure oversampling
- ✅ Can achieve desired ratio flexibility

### Cons
- ❌ Still has drawbacks of both methods (just reduced)
- ❌ More complex to implement
- ❌ Requires tuning the balance ratio

### When to Use
- When pure oversampling causes overfitting
- When pure undersampling discards too much data
- When you want more control over class ratios

### Recommended for This Project
**Priority: LOW-MEDIUM** - Consider if class weights don't work and we want to modify data.

---

## Strategy 7: Ensemble Methods

### Description
Train multiple models with different sampling strategies or parameters and combine predictions.

### Implementation Approaches
**A. Bootstrap Aggregating (Bagging)**
- Train N models on different random samples of data
- Each model sees balanced data (via undersampling)
- Average predictions

**B. Boosting (e.g., AdaBoost concept)**
- Train models sequentially, focusing on misclassified examples
- Weight difficult examples higher in subsequent rounds

### Pros
- ✅ Often best performance (wisdom of crowds)
- ✅ Reduces variance
- ✅ Can use undersampling without data waste (different samples per model)
- ✅ Robust to outliers

### Cons
- ❌ Computationally expensive (N models to train)
- ❌ Inference time increased by N
- ❌ Complex to implement and maintain
- ❌ Storage requirements (N model checkpoints)
- ❌ Diminishing returns after certain N

### When to Use
- When you need absolute best performance
- When computational resources are abundant
- For production systems with multiple models already

### Recommended for This Project
**Priority: LOW** - Too early. Try simpler approaches first.

---

## Strategy 8: Data Augmentation for Text

### Description
Generate synthetic toxic examples through text augmentation techniques to increase minority class data.

### Implementation Approaches
**A. Back-translation**
- Translate toxic text to another language and back
- Creates paraphrases

**B. Paraphrasing with LLMs**
- Use GPT/other LLMs to rephrase toxic comments
- Maintain toxicity while varying language

**C. Synonym replacement (simple)**
- Replace words with synonyms using WordNet

**D. Token perturbations**
- Random insertion, deletion, swap of tokens

### Pros
- ✅ Actually creates new training data
- ✅ Increases diversity of minority class
- ✅ Can be very effective for text
- ✅ No overfitting on exact same examples

### Cons
- ❌ Time-intensive to implement
- ❌ Risk of changing label (toxic → non-toxic or vice versa)
- ❌ Requires external APIs (for back-translation) or models (LLMs)
- ❌ Quality control needed for synthetic data
- ❌ May introduce artifacts or unrealistic examples

### When to Use
- When you have very few minority examples (<1000)
- When you have resources for data generation pipeline
- When domain-specific augmentation is possible

### Recommended for This Project
**Priority: MEDIUM-LOW** - Worth exploring if we have time, but complex.

---

## Strategy 9: Two-Stage Training (Pre-training + Fine-tuning)

### Description
First train on a larger, more balanced external dataset, then fine-tune on our imbalanced data.

### Implementation Approach
- Find related dataset (e.g., other toxicity datasets: Civil Comments, Hate Speech, etc.)
- Pre-train model on balanced external data
- Fine-tune on jigsaw data with class weights

### Pros
- ✅ Leverages external data
- ✅ Better initialization than random weights
- ✅ Model learns general toxicity patterns first
- ✅ Can improve performance significantly

### Cons
- ❌ Requires finding and preprocessing external datasets
- ❌ Time-consuming (train twice)
- ❌ External data may have different definitions of toxicity
- ❌ Risk of negative transfer if domains too different

### When to Use
- When you have access to related datasets
- When current dataset is very small
- When domain shift is minimal

### Recommended for This Project
**Priority: MEDIUM** - DistilBERT already pre-trained on general text. Could try toxicity-specific pre-training.

---

## Strategy 10: Cost-Sensitive Learning (False Positive vs False Negative Trade-off)

### Description
Explicitly define costs for different types of errors and optimize for total cost instead of accuracy.

Example costs:
- False Positive (flagging non-toxic as toxic): Cost = 1
- False Negative (missing toxic content): Cost = 10

### Implementation Approach
- Define cost matrix
- Modify loss function to incorporate costs
- Or use threshold tuning to achieve desired FP/FN ratio

### Pros
- ✅ Aligns model with business objectives
- ✅ Flexible to changing requirements
- ✅ Can be combined with class weights
- ✅ Makes trade-offs explicit

### Cons
- ❌ Requires domain knowledge to set costs
- ❌ Costs may be subjective or change over time
- ❌ Similar to class weights (not fundamentally different)

### When to Use
- When you have clear business costs for errors
- When false negatives are much worse than false positives (or vice versa)
- In production systems with real cost implications

### Recommended for This Project
**Priority: MEDIUM** - Similar to class weights but with explicit business logic.

---

## Strategy 11: Modify Model Architecture

### Description
Use architectures designed for imbalanced data or add auxiliary losses.

### Implementation Approaches
**A. Class-Balanced Loss (CB Loss)**
- Uses effective number of samples instead of raw counts
- Paper: "Class-Balanced Loss Based on Effective Number of Samples"

**B. Re-weighting + Re-sampling combined**
- Loss re-weighting + Hard example mining

**C. Auxiliary tasks**
- Multi-task learning (e.g., predict toxicity + toxic type)

### Pros
- ✅ Theoretically principled approaches
- ✅ Can achieve state-of-art results
- ✅ Combines multiple strategies

### Cons
- ❌ Complex to implement
- ❌ Requires significant code changes
- ❌ May not integrate with HuggingFace Trainer easily
- ❌ Research-level approaches (less battle-tested)

### When to Use
- When pursuing state-of-art performance
- When willing to invest in implementation
- For research projects

### Recommended for This Project
**Priority: LOW** - Too complex for initial iterations.

---

## Recommended Action Plan

### Phase 1: Quick Wins (Do First)
1. **Threshold Tuning** - Test current model with different thresholds
   - Sweep thresholds on validation set
   - Plot precision-recall curve
   - Identify if model can separate classes at all
   - **Time investment**: 1 hour

2. **Class-Weighted Loss** - Retrain with weighted loss function
   - Calculate inverse frequency weights
   - Pass to `CrossEntropyLoss`
   - Train for 1 epoch, evaluate
   - **Time investment**: 2 hours (implementation + training)

### Phase 2: If Phase 1 Insufficient
3. **Focal Loss** - Implement and test focal loss
   - Custom loss function
   - Tune γ and α parameters
   - **Time investment**: 4 hours

4. **Cost-Sensitive Learning** - Define explicit costs for FN vs FP
   - Use domain knowledge
   - Adjust weights accordingly
   - **Time investment**: 2 hours

### Phase 3: Advanced Techniques (If Needed)
5. **Data Augmentation** - Generate synthetic toxic examples
   - Start with simple methods (synonym replacement)
   - Move to LLM-based if needed
   - **Time investment**: 8+ hours

6. **External Data** - Find and incorporate related datasets
   - Search for toxicity datasets
   - Pre-train or mix with current data
   - **Time investment**: 12+ hours

### Phase 4: Production Optimization
7. **Ensemble** - If single model insufficient
   - Train multiple models with different seeds/strategies
   - Combine predictions
   - **Time investment**: 6+ hours

---

## Key Metrics to Track

For each approach, monitor:
- **Balanced Accuracy** (avg of recall per class)
- **F1 Score** (harmonic mean of precision and recall)
- **AUROC** (area under ROC curve)
- **Precision** (of positive class)
- **Recall** (of positive class)
- **Confusion Matrix** (to see FP/FN breakdown)

**Don't rely on accuracy alone** - it's misleading for imbalanced datasets.

---

## Expected Outcomes

### Realistic Targets (After Implementing Phase 1-2)
- **F1 Score**: 0.60 - 0.75
- **Precision**: 0.55 - 0.70
- **Recall**: 0.65 - 0.80
- **AUROC**: 0.85 - 0.92
- **Balanced Accuracy**: 0.75 - 0.85

### Stretch Goals (After Phase 3-4)
- **F1 Score**: 0.75 - 0.85
- **AUROC**: 0.92 - 0.96

---

## Decision Matrix

| Strategy | Implementation Effort | Expected Impact | Should Do First | Risk |
|----------|----------------------|-----------------|-----------------|------|
| Threshold Tuning | ⭐ Very Low | ⭐⭐ Medium | ✅ Yes | Low |
| Class Weights | ⭐ Very Low | ⭐⭐⭐ High | ✅ Yes | Low |
| Focal Loss | ⭐⭐ Medium | ⭐⭐⭐ High | 🔄 If needed | Medium |
| Oversampling | ⭐ Low | ⭐ Low-Medium | ❌ No | High (overfitting) |
| Undersampling | ⭐ Low | ⭐ Low | ❌ No | High (data loss) |
| Hybrid Sampling | ⭐⭐ Medium | ⭐⭐ Medium | 🔄 Maybe | Medium |
| Ensemble | ⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐ Very High | ❌ No | Low |
| Data Augmentation | ⭐⭐⭐ High | ⭐⭐⭐ High | 🔄 Maybe | Medium |
| External Pre-training | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ High | 🔄 Maybe | Medium |
| Cost-Sensitive | ⭐⭐ Medium | ⭐⭐⭐ High | 🔄 If needed | Low |
| Architecture Changes | ⭐⭐⭐⭐ Very High | ⭐⭐ Medium | ❌ No | High |

---

## Conclusion

**Start simple, iterate quickly:**
1. Threshold tuning (diagnostic)
2. Class-weighted loss (baseline fix)
3. Evaluate and decide next steps based on results

Most class imbalance problems are solved with strategies 1-2. Only move to complex approaches if simpler ones fail.
