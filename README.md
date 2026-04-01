# Transformer-Based Product Price Predictor 💰
### NLP Price Regression from Catalog Text using DistilBERT

A deep learning pipeline that predicts the price of a product purely from its catalog text description. Built using a fine-tuned DistilBERT encoder with a custom regression head, log-space training, and post-hoc calibration.

---

## Problem Statement

E-commerce catalogs often contain thousands of products with inconsistent or missing pricing. This model learns the relationship between how a product is described and what it should cost — enabling automated price suggestion from text alone.

Key challenge addressed: price distributions are heavily skewed (many cheap items, few expensive ones), so the model uses log-transformed targets and a custom weighted loss to handle this imbalance.

---

## How It Works

### 1. Data Preparation
- Loads a CSV (`Transformer_train.csv`) with `catalog_content` (product description) and `price` columns
- Applies `log1p` transformation to price → trains in log-space to handle skew
- 80/20 train-validation split

### 2. Tokenization
- Uses **DistilBERT** (`distilbert-base-uncased`) tokenizer
- Truncates/pads all descriptions to 128 tokens

### 3. Model Architecture
```
DistilBERT Encoder
       ↓
  CLS Token Embedding (768-dim)
       ↓
  Linear(768 → 384) → ReLU → Dropout(0.1)
       ↓
  Linear(384 → 1)  [log-price output]
```

### 4. Custom Weighted Loss
```python
loss = mean(exp(-target) × (pred - target)²)
```
Cheaper items (lower log-price) receive higher weight — making the model more accurate in the price range where most products fall.

### 5. Training
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: Linear warmup (10%) + linear decay
- Gradient clipping at 1.0
- 20 epochs, batch size 16

### 6. Calibration
After training, a **LinearRegression calibrator** is fit on validation predictions to correct any systematic bias in the model's raw output before final price conversion.

### 7. Inference
```
Input text → DistilBERT → log-price → calibrator → expm1 → final price ($)
```

---

## Sample Output

```
Target Description: Item Name: Deep Authentic Indian Spice Mix Blend
Optimized Predicted Price: $X.XX
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language Model | DistilBERT (HuggingFace Transformers) |
| Deep Learning | PyTorch |
| Calibration | scikit-learn LinearRegression |
| Data Handling | pandas, NumPy |
| Training Env | Google Colab (GPU recommended) |

---

## Setup & Usage

```bash
pip install transformers scikit-learn torch pandas numpy
```

1. Upload `Transformer_train.csv` to `/content/` in Colab
2. Run all cells sequentially
3. Use `predict_price_calibrated(text)` for inference:

```python
predict_price_calibrated("Item Name: Organic Whole Wheat Pasta 500g")
# → $3.49
```

---

## Design Decisions

- **Log-space training** — prevents the model from being dominated by a few high-price outliers
- **CLS pooling** — uses the `[CLS]` token embedding as a fixed-size representation of the full description
- **Weighted MSE loss** — gives more training signal to the common (cheaper) price range
- **Post-hoc calibration** — corrects systematic over/under-prediction without retraining

---

## Dataset

Expected CSV format:

| catalog_content | price |
|----------------|-------|
| "Item Name: ..." | 4.99 |
| "Item Name: ..." | 12.50 |
