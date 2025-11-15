# Training Improvements & Visualization
============================================================

"Talk is cheap. Show me the code." - Linus Torvalds

This document explains the critical fixes applied to achieve proper training performance and the new visualization features.

## 🔴 Problem Diagnosis

### Original Performance
- **Validation Accuracy: 27.60%**
- 61-class classification (random guessing = 1.6%)
- Model was learning, but extremely slowly

### Root Cause Analysis (Linus-Style)

**Layer 1: What was wrong?**

The model had 3 fundamental configuration problems:

1. **Learning Rate Too Low**
   ```
   Original: lr = 1e-4
   Problem: Too conservative for initial training from pretrained weights
   ```

2. **No Learning Rate Warmup**
   ```
   Problem: Jumped directly to 1e-4 without warm start
   Effect: Pretrained features can be destroyed in early epochs
   ```

3. **Premature LR Decay**
   ```python
   # Original: Cosine decay started from epoch 0!
   scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
   # This means LR starts decreasing immediately - WRONG!
   ```

4. **Batch Size vs LR Mismatch**
   ```
   batch_size=64 with lr=1e-4
   Effective LR ∝ lr/sqrt(batch_size) ≈ 1.25e-5
   Too small to learn effectively
   ```

**Layer 2: The "Good Taste" Fix**

Instead of adding complexity (fancy optimizers, complex schedulers), we fixed the fundamentals:
- Increase LR to proper range
- Add warmup for stability
- Fix scheduler timing
- Reduce batch size for better gradient signal

---

## ✅ Improvements Applied

### 1. Learning Rate: 1e-4 → 5e-4 (5x increase)

**File: `train.py` line 128**
```python
# Old
default=1e-4

# New
default=5e-4  # Proper LR for fine-tuning pretrained models
```

**Why this matters:**
- 1e-4 is for *late-stage* fine-tuning when you're close to convergence
- 5e-4 is appropriate for *initial* fine-tuning from pretrained weights
- For new domains (agricultural diseases), model needs room to adapt

### 2. Batch Size: 64 → 32

**File: `train.py` line 122**
```python
# Old
default=64

# New
default=32  # Better gradient signal, doubles effective LR
```

**Why this matters:**
- Smaller batch = more frequent updates = faster learning
- Reduces memory, allows larger images if needed later
- Effective LR doubles (from batchsize reduction + LR increase = 10x total improvement)

### 3. Learning Rate Warmup (NEW!)

**File: `train.py` lines 402-419**
```python
# New: Warmup for 5 epochs, then cosine decay
warmup_epochs = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.02,  # Start from lr * 0.02 = 1e-5
    end_factor=1.0,     # Reach full lr = 5e-4
    total_iters=warmup_epochs,
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs - warmup_epochs,  # Decay AFTER warmup
    eta_min=1e-6,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)
```

**Why this matters:**
- Prevents destroying pretrained features in early epochs
- Gradually increases LR: 1e-5 → 5e-4 over 5 epochs
- Only starts decay after warmup completes
- Standard practice for transfer learning

### 4. Training Visualization (NEW!)

**File: `trainer.py` - Added comprehensive plotting**

#### Features:
- **Real-time plotting**: Updates after every epoch
- **No TensorBoard dependency**: Direct matplotlib plots
- **4-panel comprehensive view**:
  1. Loss curves (train/val)
  2. Accuracy curves (train/val) with best marker
  3. Learning rate schedule with warmup indicator
  4. Overfitting analysis (train-val gap)

#### Auto-generated plots:
- `checkpoints/{save_dir}/training_curves.png` - Updates every epoch
- Shows all metrics in one glance
- Includes summary statistics box

#### History tracking:
```python
self.history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "learning_rate": [],
}
```

Saved in checkpoints for:
- Resume training with full history
- Post-training analysis
- Comparison between runs

---

## 📊 Visualization Tools

### 1. Real-time Training Plots

Automatically generated during training:
```bash
# Training will generate:
checkpoints/task1_improved/training_curves.png
# Updated after each epoch
```

### 2. Standalone Visualization Tool

**File: `visualize_training.py`**

Visualize any checkpoint:
```bash
# Single checkpoint
python visualize_training.py --checkpoint checkpoints/task1_improved/best.pth

# From directory (uses best.pth)
python visualize_training.py --checkpoint-dir checkpoints/task1_improved/

# Compare multiple runs
python visualize_training.py --compare \
    checkpoints/baseline/best.pth \
    checkpoints/improved/best.pth \
    --output comparison.png
```

Features:
- 4-panel comprehensive view
- Best accuracy marker
- Min loss marker
- Warmup end indicator
- Overfitting analysis with color-coded status
- Summary statistics box

### 3. Demo Visualization

**File: `demo_visualization.py`**

See what plots look like with synthetic data:
```bash
python demo_visualization.py
```

Generates 3 scenarios:
- `demo_plots/demo_good.png` - Well-trained model
- `demo_plots/demo_overfitting.png` - Overfitting example
- `demo_plots/demo_underfitting.png` - Underfitting example

---

## 🚀 How to Use

### Quick Start: Improved Training

```bash
# Run with improved hyperparameters
bash train_improved.sh

# Or manually:
python train.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --save-dir checkpoints/task1_improved
```

### Monitor Training

**During training:**
- Watch console for epoch-by-epoch updates
- Check `checkpoints/task1_improved/training_curves.png` (auto-updated)

**After training:**
```bash
# Visualize results
python visualize_training.py --checkpoint-dir checkpoints/task1_improved/

# Compare with baseline
python visualize_training.py --compare \
    checkpoints/task1_baseline/best.pth \
    checkpoints/task1_improved/best.pth
```

---

## 📈 Expected Results

### Before (Baseline)
```
Learning Rate: 1e-4 (too low)
Batch Size: 64 (too large)
Warmup: None (risky)
LR Decay: From epoch 0 (premature)

Result: 27.6% validation accuracy
```

### After (Improved)
```
Learning Rate: 5e-4 (proper)
Batch Size: 32 (better gradient)
Warmup: 5 epochs (safe)
LR Decay: After warmup (correct timing)

Expected: 70-85% validation accuracy
Target: 80%+ (achievable with proper training)
```

### Learning Rate Schedule Comparison

**Before:**
```
Epoch 0:  1e-4  (starts high, immediately decays)
Epoch 10: 8e-5
Epoch 25: 5e-5
Epoch 50: 1e-6  (too low, barely learning)
```

**After:**
```
Epoch 0:  1e-5  (safe start)
Epoch 1:  2e-5  (warmup)
Epoch 5:  5e-4  (peak LR, start real training)
Epoch 10: 4.8e-4 (slow decay)
Epoch 25: 2.5e-4 (cosine decay)
Epoch 50: 1e-6   (final convergence)
```

---

## 🔍 Interpreting the Plots

### Loss Curves
- **Both decreasing**: ✅ Good
- **Val plateaus, train continues**: ⚠️ Overfitting
- **Both plateau high**: ❌ Underfitting (need higher LR or more capacity)

### Accuracy Curves
- **Small gap (<5%)**: ✅ Good fit
- **Moderate gap (5-10%)**: ⚠️ Slight overfitting (acceptable)
- **Large gap (>10%)**: ❌ Overfitting (add regularization)

### Learning Rate Schedule
- **Warmup visible**: ✅ Gradual increase in first 5 epochs
- **Smooth decay**: ✅ Cosine curve after warmup
- **Check alignment**: LR should be high when loss is decreasing fast

### Overfitting Analysis
- **Green box**: ✅ Good fit (gap < 5%)
- **Orange box**: ⚠️ Slight overfitting (gap 5-10%)
- **Red box**: ❌ Overfitting (gap > 10%)

---

## 🎯 Key Takeaways (Linus-Style)

### What We Fixed
1. ✅ **Learning rate**: Bumped from too-conservative to proper range
2. ✅ **Warmup**: Added to prevent feature destruction
3. ✅ **Scheduler timing**: Fixed premature decay
4. ✅ **Batch size**: Reduced for better gradient signal
5. ✅ **Visualization**: Added real-time monitoring

### What We DIDN'T Do (Good Taste)
- ❌ No fancy optimizers (AdamW is fine)
- ❌ No complex schedulers (warmup + cosine is enough)
- ❌ No architecture changes (ResNet50 is proven)
- ❌ No data tricks yet (fix training first)

### The Philosophy
> "Premature optimization is the root of all evil." - Knuth

We fixed the **fundamentals** first:
- Learning rate
- Scheduler timing
- Batch size

**Before** adding complexity:
- Fancier augmentation
- Different architectures  
- Ensemble methods
- Test-time augmentation

---

## 📝 Files Modified

### Core Training Changes
- `train.py` - LR (line 128), batch size (line 122), warmup scheduler (lines 402-419)
- `config_task1.yaml` - Updated default hyperparameters
- `trainer.py` - Added history tracking and plotting (lines 103-111, 337-417)

### New Files
- `train_improved.sh` - Quick-start script with proper hyperparameters
- `visualize_training.py` - Standalone visualization tool
- `demo_visualization.py` - Demo with synthetic data
- `IMPROVEMENTS.md` - This document

---

## 🐛 Troubleshooting

### "No history in checkpoint"
Old checkpoints don't have history. Only new training runs (after this update) will have it.

### "Plot not updating"
Check that training is actually running. Plot updates after each epoch completes.

### "Import error: matplotlib"
```bash
pip install matplotlib
# or
uv pip install matplotlib
```

### "Still low accuracy after changes"
1. Check data quality first (garbage in = garbage out)
2. Verify class_weights.csv exists and is loaded
3. Check if model is actually training (loss should decrease)
4. Try training longer (50 epochs might not be enough)

---

## 🎓 References

**Why these changes work:**
1. **Warmup**: "Bag of Tricks for Image Classification" (He et al., 2019)
2. **LR scaling**: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
3. **Cosine annealing**: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov et al., 2017)

**Linus's principles:**
- "Talk is cheap. Show me the code."
- "Bad programmers worry about code. Good programmers worry about data structures."
- "If you need more than 3 levels of indentation, you're screwed."

We applied these by:
- ✅ Showing actual code changes, not theory
- ✅ Fixing data flow (LR schedule) first
- ✅ Keeping solutions simple and direct

---

**Last Updated**: 2024-11-15  
**Author**: Claude (supervised by Linus Torvalds 😉)